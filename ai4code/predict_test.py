import copy
import gc
import os
import random
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import code_dataset
import config
import models_bert2
import models_l2
import restore_order
import json

model1_f0_dir = '../input/ai4code-model344-f0-770/342_single_bert_l2_scaled_att_0_770'
model1_f1_dir = '../input/ai4code-model342-f1-770/342_single_bert_l2_scaled_att_1_770'
model1_f2_dir = '../input/ai4code-model342-f2-770/342_single_bert_l2_scaled_att_2_770'
model1_f3_dir = '../input/ai4code-model342-f3-770/342_single_bert_l2_scaled_att_3_770'

model2_f0_dir = "../input/ai4code-model356-f0-798/356_bert_mpnet_l2_madgrad_0_798"
model2_f1_dir = "../input/ai4code-model356-f1-798/356_bert_mpnet_l2_madgrad_1_798"
model2_f2_dir = "../input/ai4code-model356-f2-798/356_bert_mpnet_l2_madgrad_2_798"

model1_l2_dirs = [
    "../input/ai4code-l2-770/l2_500_l6_b64_w64_0_770",
    "../input/ai4code-l2-770/l2_501_l6_b64_w64_1_770",
    "../input/ai4code-l2-770/l2_502_l6_b64_w64_2_770",
    "../input/ai4code-l2-770/l2_503_l6_b64_w64_3_770",
]

model1_l2_dir_fallback = "../input/ai4code-l2-770/l2_700_l2_light_0_770"

model2_l2_dirs = [
    "../input/ai4code-l2-770/l2_510_l6_b64_w64_0_770",
    "../input/ai4code-l2-770/l2_511_l6_b64_w64_1_770",
    "../input/ai4code-l2-770/l2_512_l6_b64_w64_2_770"
]

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


seed_everything(seed=42)


def build_model_l2(cfg):
    model_params = copy.copy(cfg['model_params'])
    # if model_params['model_type'] == 'models_segmentation':
    cls = models_l2.__dict__[model_params['model_cls']]
    del model_params['model_cls']
    del model_params['model_type']
    model: nn.Module = cls(**model_params)
    return model


def no_collate(batch):
    return batch[0]


@dataclass
class L1ModelInfo:
    name: str
    config: dict
    is_separate: bool
    model_path_code: str
    model_path_code_tokenizer: str
    model_path_md: str
    model_path_md_tokenizer: str

    def __init__(self, name, cfg, model_dir):
        self.name = name
        self.config = cfg
        model_cls = cfg['model_params']['model_cls']

        if model_cls == 'DualBertWithL2':
            self.is_separate = True
        elif model_cls == 'SingleBertWithL2':
            self.is_separate = False
        else:
            raise RuntimeError(f"Invalid model cls {model_cls}")

        self.model_path_code = f'{model_dir}/l1_code'
        self.model_path_code_tokenizer = f'{model_dir}/l1_code_tokenizer'
        self.model_path_md = f'{model_dir}/l1_md' if self.is_separate else f'{model_dir}/l1_code'
        self.model_path_md_tokenizer = f'{model_dir}/l1_md_tokenizer'


@dataclass
class L2ModelInfo:
    name: str
    config: dict
    model_path_code: str
    weight: float = 1.0


def load_config(fn) -> dict:
    return yaml.load(open(fn), Loader=yaml.FullLoader)


def masked_avg(t, mask):
    return (t * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1)[:, None]


def combine_predictions(model: nn.Module, tokens: [code_dataset.TokenIdsBatch]):
    predictions = []

    for token in tokens:
        # print(token.input_ids.shape, token.input_ids.dtype, token.attention_mask.shape, token.attention_mask.dtype)
        x = model(input_ids=token.input_ids, attention_mask=token.attention_mask)
        x = masked_avg(x.last_hidden_state, token.attention_mask)
        predictions.append(x)

    predictions = torch.cat(predictions, dim=0)
    return predictions[code_dataset.restore_order_idx(tokens), :]

#
# def combine_predictions_onnx(model, tokens: [code_dataset.TokenIdsBatch]):
#     predictions = []
#
#     for token in tokens:
#         # print(token.input_ids.shape, token.input_ids.dtype, token.attention_mask.shape, token.attention_mask.dtype)
#         x = model.run(None, dict(input_ids=token.input_ids.numpy(), attention_mask=token.attention_mask.numpy()))[0]
#         # print(x)
#         # x = model(input_ids=token.input_ids, attention_mask=token.attention_mask)
#         predictions.append(x)
#
#     predictions = np.concatenate(predictions, axis=0)
#     return predictions[code_dataset.restore_order_idx(tokens), :]


def predict_l1(data_loader, model_l1, model_l1_name, tmp_dir, start_time, max_duration_s):
    for data in tqdm(data_loader):
        elapsed = time.time() - start_time
        if elapsed > max_duration_s:
            print(f'Stop prediction due to execution time {elapsed / 60.0:0.1f}m exceeded time limit {max_duration_s / 60:0.1f}m')
            break

        tokens_code = [d.cuda() for d in data['tokens_code']]
        tokens_md = [d.cuda() for d in data['tokens_md']]

        # with torch.cuda.amp.autocast():
        try:
            pred_code = combine_predictions(model_l1.get_code_model(), tokens_code)
            pred_md = combine_predictions(model_l1.get_md_model(), tokens_md)
        except:
            with torch.cuda.amp.autocast():
                pred_code = combine_predictions(model_l1.get_code_model(), tokens_code)
                pred_md = combine_predictions(model_l1.get_md_model(), tokens_md)

        pred_code = pred_code.float().detach().cpu().numpy()
        pred_md = pred_md.float().detach().cpu().numpy()

        keys_all_sorted = data['keys_all_sorted']
        keys_code = data['keys_code']
        keys_md = data['keys_md']
        item_id = data['item_id']

        np.savez(
            f'{tmp_dir}/l1/{model_l1_name}/{item_id}.npz',
            activations_code=pred_code,
            activations_md=pred_md,
            keys_all_sorted=keys_all_sorted,
            keys_code=keys_code,
            keys_md=keys_md
        )


def predict_l2(model2, test_ids, tmp_dir, l1_model_info, l2_model_info):
    predicted_ids = set()

    for item_id in tqdm(test_ids):
        pred_fn = f'{tmp_dir}/l1/{l1_model_info.name}/{item_id}.npz'

        if not os.path.exists(pred_fn):
            continue

        l1_pred = np.load(pred_fn)

        activations_code = torch.from_numpy(l1_pred['activations_code']).float().cuda()
        activations_md = torch.from_numpy(l1_pred['activations_md']).float().cuda()

        weight = l2_model_info.weight

        with torch.cuda.amp.autocast():
            try:
                pred = model2(activations_code, activations_md)
            except:
                # if run out of VRAM, lets home at least one simple model is successful
                continue

        pred_md_after_code = torch.sigmoid(pred['md_after_code'].detach()).cpu().numpy()
        pred_md_after_md = torch.sigmoid(pred['md_after_md'].detach()).cpu().numpy()
        pred_md_between_code = torch.softmax(pred['md_between_code'].detach().cpu().float(), dim=0).numpy()

        keys_code = l1_pred['keys_code']
        keys_md = l1_pred['keys_md']

        np.savez(
            f'{tmp_dir}/l2/{l2_model_info.name}/{item_id}.npz',
            keys_code=keys_code,
            keys_md=keys_md,
            md_after_code=pred_md_after_code,
            md_after_md=pred_md_after_md,
            md_between_code=pred_md_between_code,
            weight=weight
        )

        predicted_ids.add(item_id)
    return predicted_ids


def predict(l1_model_info: L1ModelInfo, l2_model_infos: [L2ModelInfo], l2_model_fallback, test_dir, tmp_dir, test_ids, start_time, max_duration_s):
    torch.set_grad_enabled(False)

    os.makedirs(f'{tmp_dir}/l1/{l1_model_info.name}', exist_ok=True)

    elapsed = time.time() - start_time
    if elapsed > max_duration_s:
        print(f'Skip model prediction due to execution time {elapsed / 60.0:0.1f}m exceeded time limit {max_duration_s / 60:0.1f}m')
        return

    l1_cfg = l1_model_info.config

    dataset_valid = code_dataset.CodeDatasetTest(
        test_data_dir=test_dir,
        test_ids=test_ids,
        code_tokenizer_name=l1_model_info.model_path_code_tokenizer,
        md_tokenizer_name=l1_model_info.model_path_md_tokenizer
    )

    if l1_model_info.is_separate:
        model_l1 = models_bert2.DualBertWithL2(
            code_model_name=l1_model_info.model_path_code,
            md_model_name=l1_model_info.model_path_md,
            l2_name='None',
            l2_params=None,
            pool_mode=l1_cfg['model_params']['pool_mode']
        )
    else:
        model_l1 = models_bert2.SingleBertWithL2(
            code_model_name=l1_model_info.model_path_code,
            l2_name='None',
            l2_params=None,
            pool_mode=l1_cfg['model_params']['pool_mode']
        )
    model_l1 = model_l1.cuda()
    model_l1.eval()

    data_loader = DataLoader(
        dataset_valid,
        num_workers=1,
        shuffle=False,
        batch_size=1,
        collate_fn=no_collate
    )

    predict_l1(
        data_loader=data_loader,
        model_l1=model_l1,
        model_l1_name=l1_model_info.name,
        tmp_dir=tmp_dir,
        start_time=start_time,
        max_duration_s=max_duration_s)

    del model_l1
    del data_loader
    del dataset_valid
    gc.collect()

    predicted_ids = set()

    for l2_model_info in l2_model_infos:
        os.makedirs(f'{tmp_dir}/l2/{l2_model_info.name}', exist_ok=True)

        model2 = build_model_l2(l2_model_info.config)
        model2 = model2.cuda()
        model2.eval()

        checkpoint = torch.load(f"{l2_model_info.model_path_code}/l2.pt")
        model2.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint

        cur_predicted_ids = predict_l2(model2, test_ids, tmp_dir, l1_model_info, l2_model_info)
        del model2
        gc.collect()

        predicted_ids.update(cur_predicted_ids)

    if l2_model_fallback is not None and len(predicted_ids) != len(test_ids):
        l2_model_info = l2_model_fallback
        failed_ids = list(set(test_ids).difference(predicted_ids))
        print(f'Using fallback model to predict {len(failed_ids)} items')

        os.makedirs(f'{tmp_dir}/l2/{l2_model_info.name}', exist_ok=True)

        model2 = build_model_l2(l2_model_info.config)
        model2 = model2.cuda()
        model2.eval()

        checkpoint = torch.load(f"{l2_model_info.model_path_code}/l2.pt")
        model2.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint

        predict_l2(model2, failed_ids, tmp_dir, l1_model_info, l2_model_info)
        del model2
        gc.collect()




def combine_l2_predictions(l2_model_names, tmp_dir, test_ids):
    l2_predictions = {}

    # for item_id, l1_pred in tqdm(l1_predictions.items()):
    for item_id in tqdm(test_ids):
        predictions = []
        total_weight = 0.0
        for l2_model_name in l2_model_names:
            fn = f'{tmp_dir}/l2/{l2_model_name}/{item_id}.npz'
            if os.path.exists(fn):
                data = np.load(fn)
                predictions.append(data)
                total_weight += float(data['weight'])

        if total_weight == 0:
            continue  # TODO: add fallback solution

        keys_code = predictions[0]['keys_code']
        keys_md = predictions[0]['keys_md']
        md_after_code = predictions[0]['md_after_code'] * predictions[0]['weight'] / total_weight
        md_after_md = predictions[0]['md_after_md'] * predictions[0]['weight'] / total_weight
        md_between_code = predictions[0]['md_between_code'] * predictions[0]['weight'] / total_weight

        for p in predictions[1:]:
            md_after_code += p['md_after_code'] * p['weight'] / total_weight
            md_after_md += p['md_after_md'] * p['weight'] / total_weight
            md_between_code += p['md_between_code'] * p['weight'] / total_weight

        keys_all_sorted_pred = restore_order.restore_order_sm(
            keys_code=keys_code,
            keys_md=keys_md,
            md_after_code=md_after_code,
            md_after_md=md_after_md,
            md_between_code=md_between_code
        )

        l2_predictions[item_id] = keys_all_sorted_pred

    return l2_predictions


def split_small_large_notebooks(test_dir: str, test_ids, large_ratio=0.1):
    test_ids_with_size = []
    for item_id in test_ids:
        data = json.load(open(f'{test_dir}/{item_id}.json'))
        nb_code = 0
        nb_md = 0
        for cell_type in data['cell_type'].values():
            if cell_type == 'code':
                nb_code += 1
            else:
                nb_md += 1

        test_ids_with_size.append(((nb_code+1)*(nb_md+1), item_id))

    test_ids_with_size = list(sorted(test_ids_with_size, reverse=True))
    test_ids_sorted = [item[1] for item in test_ids_with_size]

    nb_large = int(len(test_ids) * large_ratio + 0.999)

    return test_ids_sorted[nb_large:], test_ids_sorted[:nb_large]




def predict_all(test_data_path):
    all_samples = [s[:-5] for s in sorted(os.listdir(test_data_path)) if s.endswith('.json')]

    models_dir = f"{config.OUTPUT_DIR}/models"

    batch_size = 4096
    nb_steps = (len(all_samples) + batch_size - 1) // batch_size

    prediction = {}
    l1_cfg = load_config('experiments/344_single_bert_l2_max_loss_0.01.yaml')
    l2_cfg = dict(
        model_params=dict(
            model_type="models_l2",
            model_cls="L2Transformer",
            **l1_cfg['model_params']['l2_params']
        )
    )

    for step in tqdm(range(nb_steps)):
        predict(
            l1_model_info=L1ModelInfo(
                name='model_l1_f0',
                cfg=l1_cfg,
                model_dir=model1_f0_dir
            ),
            l2_model_info=L2ModelInfo(
                name='l2_model_l1_f0',
                config=l2_cfg,
                model_path_code=f'{model1_f0_dir}/l2',
                weight=1.0
            ),
            test_dir=test_data_path,
            test_ids=all_samples[step * batch_size:(step + 1) * batch_size],
            tmp_dir='../../ai4code_output/tmp'
        )


    item_ids = []
    cell_order = []

    for item_id, pred in prediction.items():
        item_ids.append(item_id)
        cell_order.append(' '.join(pred))

    res_df = pd.DataFrame(data={
        'id': item_ids,
        'cell_order': cell_order
    })
    res_df.to_csv('submission.csv', index=False)


def predict_oof(test_data_path):
    start_time = time.time()
    max_duration_s = 5.0*60

    folds = pd.read_csv(f'{config.DATA_DIR}/folds.csv')
    fold = 0
    folds = folds[folds.fold == fold]
    all_samples = list(sorted(folds['id'].values))[:1024]

    batch_size = 512

    prediction = {}

    m1_l1_cfg = load_config('experiments/344_single_bert_l2_max_loss_0.01.yaml')
    m1_l2_cfg = dict(
        model_params=dict(
            model_type="models_l2",
            model_cls=m1_l1_cfg['model_params']['l2_name'],
            **m1_l1_cfg['model_params']['l2_params']
        )
    )

    m2_l1_cfg = load_config('experiments/356_bert_mpnet_l2_madgrad.yaml')
    m2_l2_cfg = dict(
        model_params=dict(
            model_type="models_l2",
            model_cls=m2_l1_cfg['model_params']['l2_name'],
            **m2_l1_cfg['model_params']['l2_params']
        )
    )

    extra_l2_cfg = load_config('experiments/l2_500_l6_b64_w64.yaml')
    extra_l2_cfg_fallback = load_config('experiments/l2_700_l2_light.yaml')

    tmp_dir = '../../ai4code_output/tmp'

    def predict_model1(fold, all_samples):
        model_dir = [
            model1_f0_dir,
            model1_f1_dir,
            model1_f2_dir,
            model1_f3_dir,
        ][fold]

        model_extra_dir = model1_l2_dirs[fold]

        if fold == 0:
            l2_model_fallback = L2ModelInfo(
                        name=f'l2_model1_f0',  # re-use the model name
                        config=extra_l2_cfg_fallback,
                        model_path_code=model1_l2_dir_fallback,
                        weight=0.01
                    )
        else:
            l2_model_fallback = None

        nb_steps = (len(all_samples) + batch_size - 1) // batch_size

        for step in tqdm(range(nb_steps)):
            predict(
                l1_model_info=L1ModelInfo(
                    name=f'model1_f{fold}',
                    cfg=m1_l1_cfg,
                    model_dir=model_dir
                ),
                l2_model_infos=[
                    L2ModelInfo(
                        name=f'l2_model1_f{fold}',
                        config=m1_l2_cfg,
                        model_path_code=f'{model_dir}/l2',
                        weight=1.0
                    ),
                    L2ModelInfo(
                        name=f'l2_model1_extra_f{fold}',
                        config=extra_l2_cfg,
                        model_path_code=f'{model_extra_dir}',
                        weight=1.6
                    ),
                ],
                l2_model_fallback=l2_model_fallback,
                test_dir=test_data_path,
                test_ids=all_samples[step * batch_size:(step + 1) * batch_size],
                tmp_dir=tmp_dir,
                start_time=start_time,
                max_duration_s=max_duration_s
            )
            gc.collect()

    def predict_model2(fold, all_samples):
        model_dir = [
            model2_f0_dir,
            model2_f1_dir,
            model2_f2_dir,
        ][fold]

        model_extra_dir = model2_l2_dirs[fold]

        nb_steps = (len(all_samples) + batch_size - 1) // batch_size

        for step in tqdm(range(nb_steps)):
            predict(
                l1_model_info=L1ModelInfo(
                    name=f'model2_f{fold}',
                    cfg=m2_l1_cfg,
                    model_dir=model_dir
                ),
                l2_model_infos=[
                    L2ModelInfo(
                        name=f'l2_model2_f{fold}',
                        config=m2_l2_cfg,
                        model_path_code=f'{model_dir}/l2',
                        weight=1.0
                    ),
                    L2ModelInfo(
                        name=f'l2_model2_extra_f{fold}',
                        config=extra_l2_cfg,
                        model_path_code=f'{model_extra_dir}',
                        weight=1.6
                    ),
                ],
                l2_model_fallback=None,
                test_dir=test_data_path,
                test_ids=all_samples[step * batch_size:(step + 1) * batch_size],
                tmp_dir=tmp_dir,
                start_time=start_time,
                max_duration_s=max_duration_s
            )
            gc.collect()

    all_samples_small, all_samples_large = split_small_large_notebooks(test_dir=test_data_path, test_ids=all_samples, large_ratio=0.1)

    predict_model1(0, all_samples_large)
    predict_model2(0, all_samples_large)

    predict_model1(0, all_samples_small)
    predict_model2(0, all_samples_small)


    step_prediction = combine_l2_predictions(
        ['l2_model1_f0', 'l2_model2_f0', 'l2_model1_extra_f0', 'l2_model2_extra_f0'],
        test_ids=all_samples,
        tmp_dir='../../ai4code_output/tmp')

    prediction.update(step_prediction)

    item_ids = []
    cell_order = []

    for item_id, pred in prediction.items():
        item_ids.append(item_id)
        cell_order.append(' '.join([p for p in pred if p != '']))

    res_df = pd.DataFrame(data={
        'id': item_ids,
        'cell_order': cell_order
    })
    res_df.to_csv('submission.csv', index=False)


def check_submission():
    import metrics

    sub = pd.read_csv('submission.csv', index_col='id')
    gt = pd.read_csv('../input/train_orders.csv', index_col='id')

    keys = [str(s) for s in sub.index.values]
    sub_item = [sub.loc[k, 'cell_order'].split() for k in keys]
    gt_items = [gt.loc[k, 'cell_order'].split() for k in keys]

    score = metrics.kendall_tau(ground_truth=gt_items, predictions=sub_item)
    print(score)


if __name__ == "__main__":
    # predict_all('../input/test')
    predict_oof('/opt/data2/ai4code/train')
    check_submission()
