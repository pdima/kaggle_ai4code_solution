import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import enable_grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import defaultdict
import copy
import config
import metrics
import gc
import madgrad


import common_utils
import code_dataset
import models_bert2
import restore_order
from common_utils import DotDict, timeit_context
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# seed_everything(seed=42)


def build_model(cfg):
    model_params = copy.copy(cfg['model_params'])
    # if model_params['model_type'] == 'models_segmentation':
    cls = models_bert2.__dict__[model_params['model_cls']]
    del model_params['model_cls']
    del model_params['model_type']
    model: nn.Module = cls(**model_params)
    return model


output_names = [
    'md_after_code',
    # 'md_between_code',
    'md_after_md'
]

# set of large notebook ids training failed due to OOM error
large_notebooks = set()


def no_collate(batch):
    return batch[0]


def train(experiment_name: str, fold: int, continue_epoch: int = -3):
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
    dataset_train = code_dataset.CodeDataset(
        is_training=True,
        fold=fold,
        **cfg['dataset_params']
    )

    dataset_params_val = cfg['dataset_params'].copy()
    dataset_params_val.update(cfg.get('dataset_params_val', {}))
    dataset_valid = code_dataset.CodeDataset(
        is_training=False,
        fold=fold,
        **dataset_params_val
    )
    dataset_valid.items = dataset_valid.items[:train_params.epoch_size//2]

    batch_size = cfg['train_data_loader']['batch_size']
    use_weighted_sampler = cfg['train_data_loader'].get('use_weighted_sampler', False)
    weighted_sampler_epoch = cfg['train_data_loader'].get('weighted_sampler_epoch', 0)

    def create_train_loader(is_weighted: bool):
        if is_weighted:
            print('Create weighted sampler')

            notebook_size = pickle.load(open(f'{config.DATA_DIR}/notebook_size.pkl', 'rb'))
            notebook_size_ds = [notebook_size[sample_id][0] for sample_id in dataset_train.items]
            p = np.array(notebook_size_ds) + float(cfg['train_data_loader']['weighed_sampler_min_size'])
            p = p/p.sum()
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
            last_epoch=-1,
            first_epoch_lr_scale=0.1,
            first_steps_scale=train_params.get('first_steps_scale', None),  # {0: 1. / 30, 1: 1. / 10, 2: 1. / 3, 3: 1. / 2},
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
    cr_ce = torch.nn.CrossEntropyLoss(label_smoothing=1e-4)
    cr_cls_no_reduce = torch.nn.BCEWithLogitsLoss(reduction='none')

    loss_scale = train_params.loss_scale
    autocast_enabled = train_params.get('enable_fp16', True)

    metric_names = ['loss', 'mp_pos_ce', 'md_after_code_sum', 'md_after_code_mean_sum', 'md_after_code_max'] + output_names

    loss_scale_mp_pos_ce = loss_scale.get('mp_pos_ce', 0.0)
    if loss_scale_mp_pos_ce > 0:
        metric_names.append('mp_pos_ce')

    loss_scale_md_after_code_sum = loss_scale.get('md_after_code_sum', 0.0)
    loss_scale_md_after_code_mean_sum = loss_scale.get('md_after_code_mean_sum', 0.0)
    loss_scale_md_after_code_max = loss_scale.get('md_after_code_max', 0.0)

    is_frozen = False

    total_inputs_size_threshold = 1e6 * train_params.get('total_inputs_size_threshold', 24.0)

    if freeze_backbone_steps > 0 or continue_epoch < -1:
        is_frozen = not (continue_epoch + 1 > freeze_backbone_steps)
        model.set_backbone_trainable(not is_frozen)

    for epoch_num in range(continue_epoch + 1, nb_epochs + 1):
        if use_weighted_sampler and epoch_num == weighted_sampler_epoch:
            data_loaders["train"] = create_train_loader(is_weighted=True)

        for phase in ["train", "val"]:
            model.train(phase == "train")

            epoch_loss = {k: common_utils.AverageMeter() for k in metric_names}
            order_gt = []
            order_pred = []
            data_loader = data_loaders[phase]
            optimizer.zero_grad()

            if freeze_backbone_steps > continue_epoch and phase == 'train' and epoch_num == freeze_backbone_steps:
                model.set_backbone_trainable(True)
                is_frozen = False

            frozen_steps = 0  # display some stats to have an idea how many samples are frozen
            data_iter = tqdm(data_loader, disable=False)
            for step, data in enumerate(data_iter):
                with torch.set_grad_enabled(phase == "train"):
                    keys_all_sorted = data['keys_all_sorted']
                    keys_code = data['keys_code']
                    keys_md = data['keys_md']

                    label = {
                        l: torch.from_numpy(data[l]).float().cuda()
                        for l in output_names
                    }

                    # optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=autocast_enabled):
                        try:
                            tokens_code = [d.cuda() for d in data['tokens_code']]
                            tokens_md = [d.cuda() for d in data['tokens_md']]

                            total_inputs_size = sum([b.size * b.max_length * b.max_length for b in tokens_code] +
                                                    [b.size * b.max_length * b.max_length for b in tokens_md])

                            if phase == 'train' and epoch_num >= freeze_backbone_steps - 1:
                                should_be_frozen = total_inputs_size > total_inputs_size_threshold or \
                                                          data['item_id'] in large_notebooks or \
                                                          epoch_num < freeze_backbone_steps

                                if is_frozen != should_be_frozen:
                                    is_frozen = should_be_frozen
                                    model.set_backbone_trainable(not is_frozen)

                                if is_frozen:
                                    frozen_steps += 1

                            pred = model(
                                code_ids=tokens_code,
                                md_ids=tokens_md,
                            )
                        except RuntimeError as err:
                            print(err)
                            print(data['item'], data['item_id'])
                            print(len(keys_all_sorted), len(keys_code), len(keys_md))
                            print('code:', [(b.size, b.max_length) for b in tokens_code])
                            print('md:', [(b.size, b.max_length) for b in tokens_md])
                            print(f'total inputs size: {total_inputs_size} {total_inputs_size/1e6:0.1f}')

                            large_notebooks.add(data['item_id'])
                            frozen_steps += 1
                            # print(large_notebooks)

                            if not is_frozen:
                                # try again with backbone frozen
                                optimizer.zero_grad(set_to_none=True)
                                model.set_backbone_trainable(False)
                                is_frozen = True

                                try:
                                    pred = model(
                                        code_ids=tokens_code,
                                        md_ids=tokens_md,
                                    )
                                except RuntimeError as err:
                                    print('Still failed......................')
                                    print(err)
                                    for param in model.parameters():
                                        param.grad = None

                                    gc.collect()
                                    optimizer.zero_grad()
                                    torch.cuda.empty_cache()
                                    continue

                            else:
                                for param in model.parameters():
                                    param.grad = None
                                optimizer.zero_grad(set_to_none=True)
                                gc.collect()
                                torch.cuda.empty_cache()
                                continue
                                # raise

                        loss = 0.0
                        clip = 1e-4
                        for output_name in output_names:
                            cur_loss = cr_cls(pred[output_name], clip + label[output_name] * (1.0 - clip*2))
                            epoch_loss[output_name].update(cur_loss.detach().item())
                            loss = loss + cur_loss * loss_scale[output_name]
                            del cur_loss

                        if loss_scale_mp_pos_ce > 0:
                            cur_loss = cr_ce(
                                pred['md_between_code'],
                                torch.from_numpy(data['md_between_code']).float().cuda()
                            )
                            epoch_loss['mp_pos_ce'].update(cur_loss.detach().item())
                            loss = loss + cur_loss * loss_scale_mp_pos_ce
                            del cur_loss

                        if loss_scale_md_after_code_sum > 0:
                            cur_loss = cr_cls_no_reduce(pred['md_after_code'], clip + label['md_after_code'] * (1.0 - clip*2))
                            cur_loss = cur_loss.sum()
                            epoch_loss['md_after_code_sum'].update(cur_loss.detach().item())
                            loss = loss + cur_loss * loss_scale_md_after_code_sum

                        if loss_scale_md_after_code_mean_sum > 0:
                            cur_loss = cr_cls_no_reduce(pred['md_after_code'], clip + label['md_after_code'] * (1.0 - clip*2))
                            cur_loss = cur_loss.sum(dim=1).mean(dim=0)  # sum over code, mean over md

                            epoch_loss['md_after_code_mean_sum'].update(cur_loss.detach().item())
                            loss = loss + cur_loss * loss_scale_md_after_code_mean_sum

                        if loss_scale_md_after_code_max > 0:
                            cur_loss = cr_cls_no_reduce(pred['md_after_code'], clip + label['md_after_code'] * (1.0 - clip * 2))
                            cur_loss = cur_loss.max()
                            epoch_loss['md_after_code_max'].update(cur_loss.detach().item())
                            loss = loss + cur_loss * loss_scale_md_after_code_max

                        epoch_loss['loss'].update(loss.detach().item())

                    if phase == "train":
                        try:
                            scaler.scale(loss / grad_accumulation_steps).backward()
                        except RuntimeError as err:
                            print(err)
                            large_notebooks.add(data['item_id'])
                            print(data['item'], data['item_id'])
                            print(len(keys_all_sorted), len(keys_code), len(keys_md))
                            print('code:', [(b.size, b.max_length) for b in tokens_code])
                            print('md:', [(b.size, b.max_length) for b in tokens_md])
                            optimizer.zero_grad()
                            frozen_steps += 1
                            continue

                        if grad_accumulation_steps < 2 or (step + 1) % grad_accumulation_steps == 0 or step == train_params.epoch_size - 1:
                            # Unscales the gradients of optimizer's assigned params in-place
                            try:
                                scaler.unscale_(optimizer)
                                if grad_clip_value > 0:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                                # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                                scaler.step(optimizer)
                            except RuntimeError as err:
                                print(err)
                                large_notebooks.add(data['item_id'])
                                print('code:', [(b.size, b.max_length) for b in tokens_code])
                                print('md:', [(b.size, b.max_length) for b in tokens_md])
                                optimizer.zero_grad()
                                frozen_steps += 1
                                continue
                            scaler.update()
                            optimizer.zero_grad()

                    data_iter.set_description(f"{epoch_num} {phase[0]} fr:{frozen_steps} " + ' '.join([f"{m} {epoch_loss[m].avg:1.4f}" for m in metric_names]))

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

    dataset_valid = code_dataset.CodeDataset(
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
            tokens_code = [d.cuda() for d in data['tokens_code']]
            tokens_md = [d.cuda() for d in data['tokens_md']]

            with torch.cuda.amp.autocast():
                pred = model(
                    code_ids=tokens_code,
                    md_ids=tokens_md
                )

            #
            #
            # tokens_code = data['tokens_code'].cuda().long()
            # tokens_code_mask = data['tokens_code_mask'].cuda().long()
            # tokens_md = data['tokens_md'].cuda().long()
            # tokens_md_mask = data['tokens_md_mask'].cuda().long()
            #
            # # if tokens_code.shape[1] < 96:
            # #     continue
            #
            # print(tokens_code.shape, tokens_md.shape)

            fix, ax = plt.subplots(2, 3)

            for i, output_name in enumerate(output_names):
                ax[0, i].set_title(output_name.replace('_', ' '))
                ax[0, i].imshow(data[output_name], vmin=0, vmax=1)
                ax[1, i].imshow(torch.sigmoid(pred[output_name].detach()).cpu().numpy(), vmin=0, vmax=1)

            ax[0, 2].set_title('md between code')
            ax[0, 2].imshow(data['md_between_code'], vmin=0, vmax=1)
            ax[1, 2].imshow(torch.softmax(pred['md_between_code'].detach(), dim=1).cpu().numpy(), vmin=0, vmax=1)
            plt.show()


def check_oof_score(experiment_name: str, fold: int, epoch: int):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    oof_dir = f"{config.OUTPUT_DIR}/oof/{model_str}"
    print("\n", experiment_name, "\n")

    cfg['dataset_params']['nb_code_cells'] = 1024
    cfg['dataset_params']['nb_md_cells'] = 1024
    # cfg['dataset_params']['batch_cost'] = 1024
    # cfg['dataset_params']['max_size2'] = 256 * 256

    dataset_valid = code_dataset.CodeDataset(
        is_training=False,
        fold=fold,
        **cfg['dataset_params']
    )

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    state_dict = checkpoint["model_state_dict"]
    del checkpoint

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict)

    data_loader = DataLoader(
        dataset_valid,
        num_workers=0,
        shuffle=False,
        batch_size=1,
        collate_fn=no_collate
    )

    train_params = DotDict(cfg["train_params"])
    autocast_enabled = train_params.get('enable_fp16', True)

    order_gt = []
    order_pred = []

    torch.set_grad_enabled(False)

    data_iter = tqdm(data_loader)
    for data in data_iter:
        tokens_code = [d.cuda() for d in data['tokens_code']]
        tokens_md = [d.cuda() for d in data['tokens_md']]

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            pred = model(
                code_ids=tokens_code,
                md_ids=tokens_md
            )

        pred_md_after_code = torch.sigmoid(pred['md_after_code'].detach()).cpu().numpy()
        pred_md_after_md = torch.sigmoid(pred['md_after_md'].detach()).cpu().numpy()
        pred_md_between_code = torch.softmax(pred['md_between_code'].detach(), dim=0).cpu().numpy()
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

        if len(order_gt) % 1000 == 0:
            print(metrics.kendall_tau(ground_truth=order_gt, predictions=order_pred))

    print(metrics.kendall_tau(ground_truth=order_gt, predictions=order_pred))


def check_oof_score_cached(experiment_name: str, fold: int, epoch: int, other_experiments=()):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)
    cache_dir = f"{config.DATA_DIR}/decoder/{model_str}_{epoch:03}_f{fold}"

    other_cache_dirs = [
        f"{config.DATA_DIR}/decoder/{other_experiment_name}_{other_epoch:03}_f{fold}"
        for other_experiment_name, other_epoch in other_experiments
    ]

    print("\n", experiment_name, "\n")
    print(other_experiments)

    languages = pd.read_csv(f'{config.DATA_DIR}/train_languages_v2.csv', index_col='item_id')

    folds = pd.read_csv(f'{config.DATA_DIR}/folds.csv')
    folds = folds[folds.fold == fold]
    items = list(sorted(folds['id'].values))

    order_gt = []
    order_pred = []

    order_gt_en = []
    order_pred_en = []

    order_gt_non_en = []
    order_pred_non_en = []

    torch.set_grad_enabled(False)

    for item_id in tqdm(items):
        data = np.load(f'{cache_dir}/{item_id}.npz')

        pred_md_after_code = torch.sigmoid(torch.from_numpy(data['md_after_code'])).numpy()
        pred_md_after_md = torch.sigmoid(torch.from_numpy(data['md_after_md'])).numpy()
        pred_md_between_code = torch.softmax(torch.from_numpy(data['md_between_code']), dim=0).numpy()

        for other_cache_dir in other_cache_dirs:
            data2 = np.load(f'{other_cache_dir}/{item_id}.npz')
            pred_md_after_code2 = torch.sigmoid(torch.from_numpy(data2['md_after_code'])).numpy()
            pred_md_after_md2 = torch.sigmoid(torch.from_numpy(data2['md_after_md'])).numpy()
            pred_md_between_code2 = torch.softmax(torch.from_numpy(data2['md_between_code']), dim=0).numpy()

            pred_md_after_code = pred_md_after_code + pred_md_after_code2
            pred_md_after_md = pred_md_after_md + pred_md_after_md2
            pred_md_between_code = pred_md_between_code + pred_md_between_code2

        pred_md_after_code /= (1.0 + len(other_cache_dirs))
        pred_md_after_md /= (1.0 + len(other_cache_dirs))
        pred_md_between_code /= (1.0 + len(other_cache_dirs))

        keys_all_sorted = [str(item) for item in data['keys_all_sorted']]
        keys_code = [str(item) for item in data['keys_code']]
        keys_md = [str(item) for item in data['keys_md']]

        keys_all_sorted_pred = restore_order.restore_order_sm(
            keys_code=keys_code,
            keys_md=keys_md,
            md_after_code=pred_md_after_code,
            md_after_md=pred_md_after_md,
            md_between_code=pred_md_between_code
        )

        order_gt.append(keys_all_sorted)
        order_pred.append(keys_all_sorted_pred)

        lan = languages.loc[item_id, 'language']
        if lan == 'en':
            order_gt_en.append(keys_all_sorted)
            order_pred_en.append(keys_all_sorted_pred)
        else:
            order_gt_non_en.append(keys_all_sorted)
            order_pred_non_en.append(keys_all_sorted_pred)

        # print(keys_all_sorted)
        # print(keys_all_sorted_pred)
        # print(keys_code)
        # print(keys_md)

        if len(order_gt) % 10000 == 0:
            score = metrics.kendall_tau(ground_truth=order_gt, predictions=order_pred)
            score_en = metrics.kendall_tau(ground_truth=order_gt_en, predictions=order_pred_en)
            score_non_en = metrics.kendall_tau(ground_truth=order_gt_non_en, predictions=order_pred_non_en)
            print(f'{score:0.4f}  en {score_en:0.4f} non en {score_non_en:0.4f}')

    score = metrics.kendall_tau(ground_truth=order_gt, predictions=order_pred)
    score_en = metrics.kendall_tau(ground_truth=order_gt_en, predictions=order_pred_en)
    score_non_en = metrics.kendall_tau(ground_truth=order_gt_non_en, predictions=order_pred_non_en)
    print(f'{score:0.4f}  en {score_en:0.4f} non en {score_non_en:0.4f}    {experiment_name} {fold} {epoch}')


def predict_encoders(experiment_name: str, fold: int, epoch: int, is_training: bool):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    dst_dir = f"{config.DATA_DIR}/decoder/{model_str}_{epoch:03}_f{fold}"
    os.makedirs(dst_dir, exist_ok=True)
    print(dst_dir)

    print("\n", experiment_name, "\n")
    print(dst_dir)

    cfg['dataset_params']['nb_code_cells'] = 2048
    cfg['dataset_params']['nb_md_cells'] = 2048

    dataset_valid = code_dataset.CodeDataset(
        is_training=is_training,
        fold=fold,
        **cfg['dataset_params']
    )

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt", map_location='cpu')
    model_state_dict = checkpoint["model_state_dict"]
    del checkpoint

    model = build_model(cfg)
    print(model.__class__.__name__)
    model.load_state_dict(model_state_dict)
    model = model.cuda()
    model.eval()

    data_loader = DataLoader(
        dataset_valid,
        num_workers=4,
        shuffle=False,
        batch_size=1,
        collate_fn=no_collate
    )

    torch.set_grad_enabled(False)

    data_iter = tqdm(data_loader)
    for data in data_iter:
        tokens_code = [d.cuda() for d in data['tokens_code']]
        tokens_md = [d.cuda() for d in data['tokens_md']]

        with torch.cuda.amp.autocast():
            pred_code = model.combine_predictions(model.get_code_model(), tokens_code)
            pred_md = model.combine_predictions(model.get_md_model(), tokens_md)

            pred_combined = model.forward_combined(pred_code, pred_md)

        pred_code = pred_code.float().detach().cpu().numpy()
        pred_md = pred_md.float().detach().cpu().numpy()
        pred_combined = {k: v.float().detach().cpu().numpy() for k, v in pred_combined.items()}

        keys_all_sorted = data['keys_all_sorted']
        keys_code = data['keys_code']
        keys_md = data['keys_md']
        item_id = data['item_id']

        np.savez(
            f'{dst_dir}/{item_id}.npz',
            item_id=item_id,
            pred_code=pred_code,
            pred_md=pred_md,
            keys_all_sorted=keys_all_sorted,
            keys_code=keys_code,
            keys_md=keys_md,
            **pred_combined
        )


def predict_extra_data(experiment_name: str, fold: int, epoch: int, is_training: bool, step: int):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    dst_dir = f"{config.EXTRA_DATA_DIR}/extra_data/{model_str}_{epoch:03}_f{fold}"
    os.makedirs(dst_dir, exist_ok=True)
    print(dst_dir)

    print("\n", experiment_name, "\n")
    print(dst_dir)

    cfg['dataset_params']['nb_code_cells'] = 4096
    cfg['dataset_params']['nb_md_cells'] = 4096

    dataset_valid = code_dataset.CodeDataset(
        is_training=is_training,
        fold=fold,
        dataset_type='extra_data',
        **cfg['dataset_params']
    )

    dataset_valid.items = dataset_valid.items[step::2]

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt", map_location='cpu')
    model_state_dict = checkpoint["model_state_dict"]
    del checkpoint

    model = build_model(cfg)
    print(model.__class__.__name__)
    model.load_state_dict(model_state_dict)
    model = model.cuda()
    model.eval()

    data_loader = DataLoader(
        dataset_valid,
        num_workers=16,
        shuffle=False,
        batch_size=1,
        collate_fn=no_collate
    )

    torch.set_grad_enabled(False)

    data_iter = tqdm(data_loader)
    for data in data_iter:
        item_id = data['item_id']
        if os.path.exists(f'{dst_dir}/{item_id}.npz'):
            continue

        tokens_code = [d.cuda() for d in data['tokens_code']]
        tokens_md = [d.cuda() for d in data['tokens_md']]

        with torch.cuda.amp.autocast():
            pred_code = model.combine_predictions(model.get_code_model(), tokens_code)
            pred_md = model.combine_predictions(model.get_md_model(), tokens_md)

            pred_combined = model.forward_combined(pred_code, pred_md)

        pred_code = pred_code.float().detach().cpu().numpy()
        pred_md = pred_md.float().detach().cpu().numpy()
        pred_combined = {k: v.float().detach().cpu().numpy() for k, v in pred_combined.items()}

        keys_all_sorted = data['keys_all_sorted']
        keys_code = data['keys_code']
        keys_md = data['keys_md']


        np.savez(
            f'{dst_dir}/{item_id}.npz',
            item_id=item_id,
            pred_code=pred_code,
            pred_md=pred_md,
            keys_all_sorted=keys_all_sorted,
            keys_code=keys_code,
            keys_md=keys_md,
            **pred_combined
        )



def save_empty_code(experiment_name: str, fold: int, epoch: int):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    dst_dir = f"{config.DATA_DIR}/decoder/{model_str}_{epoch:03}_f{fold}"
    os.makedirs(dst_dir, exist_ok=True)
    print(dst_dir)

    print("\n", experiment_name, "\n")
    print(dst_dir)

    cfg['dataset_params']['nb_code_cells'] = 2048
    cfg['dataset_params']['nb_md_cells'] = 2048

    dataset_valid = code_dataset.CodeDataset(
        is_training=False,
        fold=fold,
        **cfg['dataset_params']
    )

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt", map_location='cpu')
    model_state_dict = checkpoint["model_state_dict"]
    del checkpoint

    model = build_model(cfg)
    print(model.__class__.__name__)
    model.load_state_dict(model_state_dict)
    model = model.cuda()
    model.eval()

    input_ids = dataset_valid.code_tokenizer.encode_plus(
                    dataset_valid.cell_prefix + dataset_valid.preprocess_code_src(''),
                    max_length=dataset_valid.max_code_tokens_number,
                    truncation=True,
                    add_special_tokens=True
                )['input_ids']

    input_ids = torch.tensor([input_ids], dtype=torch.long).cuda()

    torch.set_grad_enabled(False)

    x = model.get_code_model()(input_ids=input_ids)
    pred_code = model.pool(x, torch.ones_like(x.last_hidden_state)[:, :, 0])[0].float().detach().cpu().numpy()
    print(input_ids.shape, pred_code.shape)

    np.savez(
        f'{dst_dir}/empty.npz',
        pred_code=pred_code
    )



def export_model(experiment_name: str, fold: int, epoch: int):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"{config.OUTPUT_DIR}/checkpoints/{model_str}_{fold}"
    dst_dir = f"{config.OUTPUT_DIR}/models/{model_str}_{fold}_{epoch:03}"
    dst_dir_md = f"{config.OUTPUT_DIR}/models/{model_str}_{fold}_{epoch:03}/l1_md"
    dst_dir_code = f"{config.OUTPUT_DIR}/models/{model_str}_{fold}_{epoch:03}/l1_code"
    dst_dir_l2 = f"{config.OUTPUT_DIR}/models/{model_str}_{fold}_{epoch:03}/l2"
    dst_dir_md_tokenizer = f"{config.OUTPUT_DIR}/models/{model_str}_{fold}_{epoch:03}/l1_md_tokenizer"
    dst_dir_code_tokenizer = f"{config.OUTPUT_DIR}/models/{model_str}_{fold}_{epoch:03}/l1_code_tokenizer"
    os.makedirs(dst_dir_md, exist_ok=True)
    os.makedirs(dst_dir_code, exist_ok=True)
    os.makedirs(dst_dir_l2, exist_ok=True)
    os.makedirs(dst_dir_md_tokenizer, exist_ok=True)
    os.makedirs(dst_dir_code_tokenizer, exist_ok=True)
    print(dst_dir)

    print("\n", experiment_name, "\n")
    print(dst_dir)

    cfg['dataset_params']['nb_code_cells'] = 2048
    cfg['dataset_params']['nb_md_cells'] = 2048

    dataset_valid = code_dataset.CodeDataset(
        is_training=False,
        fold=fold,
        **cfg['dataset_params']
    )

    dataset_valid.md_tokenizer.save_pretrained(dst_dir_md_tokenizer)
    dataset_valid.code_tokenizer.save_pretrained(dst_dir_code_tokenizer)

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model_state_dict = checkpoint["model_state_dict"]
    del checkpoint

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()
    model.load_state_dict(model_state_dict)

    if cfg['model_params']['model_cls'] == 'SingleBertWithL2':
        model.bert_code.save_pretrained(dst_dir_code)
    else:
        model.bert_md.save_pretrained(dst_dir_md)
        model.bert_code.save_pretrained(dst_dir_code)

    torch.save(
        {"model_state_dict": model.l2.state_dict()},
        f"{dst_dir_l2}/l2.pt",
    )


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # check_score()

    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default="check")
    parser.add_argument("experiment", type=str, default="")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=-8)
    parser.add_argument("--step", type=int, default=0)

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

    if action == "oof_score_cached":
        check_oof_score_cached(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold,
            other_experiments=[]
            # other_experiments=[('344_single_bert_l2_max_loss_0.01', 536)]
        )

    if action == "predict":
        predict_encoders(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold,
            is_training=False
        )

        check_oof_score_cached(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )

        predict_encoders(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold,
            is_training=True
        )

    if action == "predict_encoders_train":
        predict_encoders(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold,
            is_training=True
        )

    if action == "predict_encoders_val":
        predict_encoders(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold,
            is_training=False
        )

        check_oof_score_cached(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )

    if action == "predict_extra":
        predict_extra_data(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold,
            is_training=False,
            step=args.step
        )

    if action == "save_empty_code":
        save_empty_code(
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
