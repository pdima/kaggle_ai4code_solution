import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import defaultdict
import copy
import config
import metrics
import madgrad
import pickle

import common_utils
import code_dataset
import models_l2
import restore_order
from common_utils import DotDict, timeit_context
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


from train import output_names

def build_model(cfg):
    model_params = copy.copy(cfg['model_params'])
    # if model_params['model_type'] == 'models_segmentation':
    cls = models_l2.__dict__[model_params['model_cls']]
    del model_params['model_cls']
    del model_params['model_type']
    model: nn.Module = cls(**model_params)
    return model


def no_collate(batch):
    return batch[0]


def train(experiment_name: str, fold: str, continue_epoch: int = -3):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)

    model_params = DotDict(cfg["model_params"])
    model_type = model_params.model_type
    train_params = DotDict(cfg["train_params"])

    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    tensorboard_dir = f"{config.OUTPUT_DIR}/tensorboard/{model_type}/{model_str}_{fold}"
    oof_dir = f"{config.OUTPUT_DIR}/oof/{model_str}_{fold}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)
    print("\n", experiment_name, "\n")

    logger = SummaryWriter(log_dir=tensorboard_dir)

    scaler = torch.cuda.amp.GradScaler()
    dataset_train = code_dataset.CodeDatasetL2(
        is_training=True,
        fold=fold,
        **cfg['dataset_params']
    )

    dataset_valid = code_dataset.CodeDatasetL2(
        is_training=False,
        fold=fold,
        **cfg['dataset_params']
    )
    dataset_valid.items = dataset_valid.items[:train_params.epoch_size//2]

    batch_size = 1
    use_weighted_sampler = cfg['train_data_loader'].get('use_weighted_sampler', False)
    weighted_sampler_epoch = cfg['train_data_loader'].get('weighted_sampler_epoch', 0)

    def create_train_loader(is_weighted: bool):
        if is_weighted:
            print('Create weighted sampler')

            notebook_size = pickle.load(open(f'{config.DATA_DIR}/notebook_size.pkl', 'rb'))
            notebook_size_ds = [notebook_size[sample_id][0] for sample_id in dataset_train.items]
            p = np.array(notebook_size_ds) + float(cfg['train_data_loader']['weighed_sampler_min_size'])
            p = p/p.sum()

            nb_extra_items = len(dataset_train.extra_items)
            if nb_extra_items:
                p = np.concatenate(
                    [p*(2.0/3.0), np.ones((nb_extra_items,), dtype=np.float64)/(3.0*nb_extra_items)]
                )
            sampler = torch.utils.data.WeightedRandomSampler(weights=p, replacement=True, num_samples=train_params.epoch_size)
        else:
            sampler = torch.utils.data.RandomSampler(data_source=dataset_train, replacement=True, num_samples=train_params.epoch_size)

        return DataLoader(
            dataset_train,
            num_workers=cfg['train_data_loader']['num_workers'],
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=no_collate
        )

    data_loaders = {
        "train": create_train_loader(is_weighted=use_weighted_sampler and continue_epoch >= weighted_sampler_epoch),
        "val": DataLoader(
            dataset_valid,
            num_workers=cfg['val_data_loader']['num_workers'],
            shuffle=False,
            batch_size=cfg['val_data_loader']['batch_size'],
            collate_fn=no_collate
        ),
    }

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.train()

    initial_lr = float(train_params.initial_lr)
    if train_params.optimizer == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "madgrad":
        optimizer = madgrad.MADGRAD(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, nesterov=True)
    else:
        raise RuntimeError("Invalid optimiser" + train_params.optimizer)

    nb_epochs = train_params.nb_epochs
    if train_params.scheduler == "steps":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=train_params.optimiser_milestones,
            gamma=0.2,
            last_epoch=continue_epoch
        )
    elif train_params.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = common_utils.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=train_params.scheduler_period,
            T_mult=train_params.get('scheduler_t_mult', 1),
            eta_min=initial_lr / 100.0,
            first_epoch_lr_scale=0.1,
            min_first_steps_scale_epoch=16,  # don't ramp up the first initial not pretrained steps
            first_steps_scale=train_params.get('first_steps_scale', None),
            last_epoch=-1,
        )
        for i in range(continue_epoch + 1):
            scheduler.step()
    else:
        raise RuntimeError("Invalid scheduler name")

    if continue_epoch > -1:
        print(f"{checkpoints_dir}/{continue_epoch:03}.pt")
        checkpoint = torch.load(f"{checkpoints_dir}/{continue_epoch:03}.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        del checkpoint

    grad_clip_value = train_params.get("grad_clip", 8.0)
    freeze_backbone_steps = train_params.get("freeze_backbone_steps", 0)
    grad_accumulation_steps = train_params.grad_accumulation_steps
    print(f"grad clip: {grad_clip_value} freeze_backbone_steps {freeze_backbone_steps}")
    print(f"Num training samples: {len(dataset_train)} val {len(dataset_valid)}")

    cr_cls = torch.nn.BCEWithLogitsLoss()
    cr_ce_no_reduce = torch.nn.CrossEntropyLoss(label_smoothing=1e-4, reduction='none')
    cr_cls_no_reduce = torch.nn.BCEWithLogitsLoss(reduction='none')
    cr_ce = torch.nn.CrossEntropyLoss(label_smoothing=1e-4)

    loss_scale = train_params.loss_scale

    metric_names = ['loss', 'mp_pos_ce', 'md_after_code_max', 'md_after_code_sum', 'md_after_code_mean_sum'] + output_names

    loss_scale_mp_pos_ce = loss_scale.get('mp_pos_ce', 0.0)
    loss_scale_md_after_code = loss_scale['md_after_code']
    loss_scale_md_after_md = loss_scale['md_after_md']

    batch_md_after_code_size = train_params.get('batch_md_after_code_size', 64*64)
    loss_sum_mul = 100.0 / batch_md_after_code_size

    total_md_after_code_size = 0
    cur_batch_size = 0

    for epoch_num in range(continue_epoch + 1, nb_epochs + 1):
        if use_weighted_sampler and epoch_num == weighted_sampler_epoch:
            data_loaders["train"] = create_train_loader(is_weighted=True)

        for phase in ["train", "val"]:
            model.train(phase == "train")

            epoch_loss = {k: common_utils.AverageMeter() for k in metric_names}
            order_gt = []
            order_pred = []
            batch_sizes = [0]
            data_loader = data_loaders[phase]
            optimizer.zero_grad()

            data_iter = tqdm(data_loader, disable=False)
            for step, data in enumerate(data_iter):
                with torch.set_grad_enabled(phase == "train"):
                    keys_all_sorted = data['keys_all_sorted']
                    keys_code = data['keys_code']
                    keys_md = data['keys_md']
                    activations_code = torch.from_numpy(data['activations_code']).float().cuda()
                    activations_md = torch.from_numpy(data['activations_md']).float().cuda()

                    label = {
                        l: torch.from_numpy(data[l]).float().cuda()
                        for l in output_names
                    }

                    # optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        pred = model(activations_code, activations_md)

                        total_md_after_code_size += np.prod(data['md_after_code'].shape)
                        # total_md_after_md_size += np.prod(data['md_after_md'].shape)
                        # total_md_pos_size += np.prod(data['md_between_code'].shape)
                        cur_batch_size += 1

                        clip = 1e-4
                        loss_md_after_code = cr_cls_no_reduce(pred['md_after_code'], clip + label['md_after_code'] * (1.0 - clip*2)).sum() * loss_sum_mul
                        loss_md_after_md = cr_cls_no_reduce(pred['md_after_md'], clip + label['md_after_md'] * (
                                    1.0 - clip * 2)).sum() * loss_sum_mul

                        loss_md_pos = cr_ce_no_reduce(
                            pred['md_between_code'],
                            torch.from_numpy(data['md_between_code']).float().cuda()
                        ).sum() * loss_sum_mul

                        epoch_loss['md_after_code'].update(loss_md_after_code.detach().item())
                        epoch_loss['md_after_md'].update(loss_md_after_md.detach().item())
                        epoch_loss['mp_pos_ce'].update(loss_md_pos.detach().item())

                        loss = (loss_md_after_code * loss_scale_md_after_code +
                                loss_md_after_md * loss_scale_md_after_md +
                                loss_md_pos * loss_scale_mp_pos_ce)

                        epoch_loss['loss'].update(loss.detach().item())

                    if phase == "train":
                        try:
                            scaler.scale(loss / grad_accumulation_steps).backward()
                        except RuntimeError as err:
                            print(err)
                            print(data['item'], data['item_id'])
                            print(len(keys_all_sorted), len(keys_code), len(keys_md))
                            optimizer.zero_grad()
                            continue

                        # if grad_accumulation_steps < 2 or (step + 1) % grad_accumulation_steps == 0 or step == train_params.epoch_size - 1 or total_md_after_code_size >= batch_md_after_code_size:
                        if step == train_params.epoch_size - 1 or total_md_after_code_size >= batch_md_after_code_size:
                            # Unscales the gradients of optimizer's assigned params in-place
                            scaler.unscale_(optimizer)
                            if grad_clip_value > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                            batch_sizes.append(cur_batch_size)
                            cur_batch_size = 0
                            total_md_after_code_size = 0

                    data_iter.set_description(f"{epoch_num} {phase[0]} " + ' '.join([f"{m} {epoch_loss[m].avg:1.4f}" for m in metric_names]) + f" bs {np.mean(batch_sizes):0.1f} {batch_sizes[-1]}")

                    pred_md_after_code = torch.sigmoid(pred['md_after_code'].detach().cpu().float()).numpy()
                    pred_md_after_md = torch.sigmoid(pred['md_after_md'].detach().cpu().float()).numpy()
                    pred_md_between_code = torch.softmax(pred['md_between_code'].detach().cpu().float(), dim=0).numpy()

                    keys_all_sorted_pred = restore_order.restore_order_sm(
                        keys_code=keys_code,
                        keys_md=keys_md,
                        md_after_code=pred_md_after_code,
                        md_after_md=pred_md_after_md,
                        md_between_code=pred_md_between_code
                    )

                    order_gt.append(keys_all_sorted)
                    order_pred.append(keys_all_sorted_pred)

                    del pred
                    del loss

            if epoch_num > 1:
                tau = metrics.kendall_tau(ground_truth=order_gt, predictions=order_pred)
                logger.add_scalar(f"tau_{phase}", tau, epoch_num)

                logger.add_scalar(f"cur_batch_size_{phase}", np.mean(batch_sizes), epoch_num)
                batch_sizes = [0]

                for metric_name in metric_names:
                    logger.add_scalar(f"{metric_name}_{phase}", epoch_loss[metric_name].avg, epoch_num)

                if phase == "train":
                    logger.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_num)
                logger.flush()

            if phase == "train":
                if epoch_num >= 0:
                    scheduler.step()
                if (epoch_num % train_params.save_period == 0) and (epoch_num > 0) or (epoch_num == nb_epochs):
                    try:
                        torch.save(
                            {
                                "epoch": epoch_num,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                            },
                            f"{checkpoints_dir}/{epoch_num:03}.pt",
                        )
                    except OSError as err:
                        print("Failed to save checkpoint")
                        print(err)


def check(experiment_name: str, fold: int, epoch: int):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    oof_dir = f"{config.OUTPUT_DIR}/oof/{model_str}"
    print("\n", experiment_name, "\n")

    cfg['dataset_params']['nb_code_cells'] = 1024
    cfg['dataset_params']['nb_md_cells'] = 1024
    cfg['dataset_params']['max_size2'] = 262144//8

    dataset_valid = code_dataset.CodeDatasetL2(
        is_training=False,
        fold=fold,
        **cfg['dataset_params']
    )

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()

    data_loader = DataLoader(
        dataset_valid,
        num_workers=0,
        shuffle=False,
        batch_size=1,
        collate_fn=no_collate
    )

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # output_names += ['md_between_code']

    with torch.set_grad_enabled(False):
        data_iter = tqdm(data_loader, disable=True)
        for data in data_iter:
            activations_code = data['activations_code'].float().cuda()
            activations_md = data['activations_md'].float().cuda()

            with torch.cuda.amp.autocast():
                pred = model(activations_code, activations_md)

            fix, ax = plt.subplots(2, 7)

            for i, output_name in enumerate(output_names):
                ax[0, i].set_title(output_name.replace('_', ' '))
                ax[0, i].imshow(data[output_name], vmin=0, vmax=1)
                ax[1, i].imshow(torch.sigmoid(pred[output_name].detach()).cpu().numpy(), vmin=0, vmax=1)

            ax[0, 6].set_title('md between code')
            ax[0, 6].imshow(data['md_between_code'], vmin=0, vmax=1)
            ax[1, 6].imshow(torch.softmax(pred['md_between_code'].detach(), dim=1).cpu().numpy(), vmin=0, vmax=1)
            plt.show()


def check_oof_score(experiment_name: str, fold: int, epoch: int):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    oof_dir = f"{config.OUTPUT_DIR}/oof/{model_str}"
    print("\n", experiment_name, "\n")

    dataset_valid = code_dataset.CodeDatasetL2(
        is_training=False,
        fold=fold,
        **cfg['dataset_params']
    )

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()

    data_loader = DataLoader(
        dataset_valid,
        num_workers=0,
        shuffle=False,
        batch_size=1,
        collate_fn=no_collate
    )

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    order_gt = []
    order_pred = []

    torch.set_grad_enabled(False)

    data_iter = tqdm(data_loader)
    for data in data_iter:
        activations_code = torch.from_numpy(data['activations_code']).float().cuda()
        activations_md = torch.from_numpy(data['activations_md']).float().cuda()

        with torch.cuda.amp.autocast():
            pred = model(activations_code, activations_md)

        pred_md_after_code = torch.sigmoid(pred['md_after_code'].detach()).cpu().numpy()
        pred_md_after_md = torch.sigmoid(pred['md_after_md'].detach()).cpu().numpy()
        pred_md_between_code = torch.softmax(pred['md_between_code'].detach().cpu().float(), dim=0).numpy()
        # print(data['keys_all_sorted'])
        keys_all_sorted = data['keys_all_sorted']
        keys_code = data['keys_code']
        keys_md = data['keys_md']

        keys_all_sorted_pred = restore_order.restore_order_sm(
            keys_code=keys_code,
            keys_md=keys_md,
            md_after_code=pred_md_after_code,
            md_after_md=pred_md_after_md,
            md_between_code=pred_md_between_code
        )

        order_gt.append(keys_all_sorted)
        order_pred.append(keys_all_sorted_pred)

        # print(keys_all_sorted)
        # print(keys_all_sorted_pred)
        # print(keys_code)
        # print(keys_md)

        if len(order_gt) % 10000 == 0:
            print(metrics.kendall_tau(ground_truth=order_gt, predictions=order_pred))

    score = metrics.kendall_tau(ground_truth=order_gt, predictions=order_pred)
    print(score)
    print(f'{score:0.4f}  {experiment_name} {fold} {epoch}')


def export_model(experiment_name: str, fold: int, epoch: int):
    model_str = experiment_name

    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    output_dir = f"{config.OUTPUT_DIR}/models/{model_str}_{fold}_{epoch}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"{checkpoints_dir}/{epoch:03}.pt")
    print(output_dir)

    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    torch.save(
        {
            "model_state_dict": checkpoint["model_state_dict"],
        },
        f"{output_dir}/l2.pt",
    )


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    # check_score()

    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default="check")
    parser.add_argument("experiment", type=str, default="")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=-1)

    args = parser.parse_args()
    action = args.action
    experiment_name = common_utils.normalize_experiment_name(args.experiment)

    if action == "train":
        train(
            experiment_name=experiment_name,
            continue_epoch=args.epoch,
            fold=args.fold
        )

    if action == "check":
        check(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )

    if action == "oof_score":
        check_oof_score(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )

    if action == "export":
        export_model(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )
