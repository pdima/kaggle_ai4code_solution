import io
import json
import os
import random
import re
import sys
import tokenize
from dataclasses import dataclass

import matplotlib.pyplot as plt
# import nltk
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm

import common_utils as utils
import config
from restore_order import restore_order_sm

# nltk.download('wordnet')
# nltk.download('omw-1.4')


# stemmer = nltk.stem.WordNetLemmatizer()


def preprocess_text_v1(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()
    # return document

    return document

    # # Lemmatization
    # tokens = document.split()
    # tokens = [stemmer.lemmatize(word) for word in tokens]
    # tokens = [word for word in tokens if len(word) > 3]
    #
    # preprocessed_text = ' '.join(tokens)
    # return preprocessed_text


def preprocess_text_v2(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()
    # return document

    # Lemmatization
    tokens = document.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


def pad_to_shape(a, dst_shape):
    return np.pad(
        a,
        pad_width=((0, d - s) for d, s in zip(dst_shape, a.shape))
    )


def t5_python_tokenizer(line):
    result = []
    line_io = io.StringIO(line)
    nb_valid_lines = 0

    try:
        for toktype, tok, start, end, cur_line in tokenize.generate_tokens(line_io.readline):
            # print(toktype, tok, start, end, cur_line)
            nb_valid_lines = end[0] + 1
            if toktype != tokenize.COMMENT:
                if toktype == tokenize.STRING:
                    result.append("CODE_STRING")
                elif toktype == tokenize.NUMBER:
                    result.append("CODE_INTEGER")
                elif (not tok == "\n") and (not tok == "    "):
                    result.append(str(tok))
        return ' '.join(result)
    except Exception as e:
        # print(e)
        # print(line)

        if len(result) > 0 and nb_valid_lines > len(line.split('\n')) // 2:
            return ' '.join(result)
        else:
            line = line.replace("    ", " ")
            line = line.replace("\n", " ")
            line = line.replace("  ", " ")
            return line

@dataclass
class TokenIdsBatch:
    size: int
    max_length: int

    input_ids: torch.tensor
    attention_mask: torch.tensor
    src_list_pos: [int]

    def cuda(self):
        return TokenIdsBatch(
            size=self.size,
            max_length=self.max_length,
            input_ids=self.input_ids.cuda(),
            attention_mask=self.attention_mask.cuda(),
            src_list_pos=self.src_list_pos
        )

    def clip_max_len(self, max_length):
        if max_length < self.max_length:
            return TokenIdsBatch(
                size=self.size,
                max_length=max_length,
                input_ids=self.input_ids[:, :max_length],
                attention_mask=self.attention_mask[:, :max_length],
                src_list_pos=self.src_list_pos
            )
        else:
            return self


def split_token_ids_to_batches(input_ids: [[int]], batch_cost: int, max_len: int, max_size2: int, len_round_up: int = 4) -> [TokenIdsBatch]:
    res = []

    input_ids = [ids[:max_len] for ids in input_ids]
    size_with_pos = [((len(ids) + len_round_up - 1) // len_round_up * len_round_up, pos) for pos, ids in enumerate(input_ids)]
    size_with_pos_sorted = list(sorted(size_with_pos))

    cur_batch = TokenIdsBatch(size=0, max_length=0, input_ids=None, attention_mask=None, src_list_pos=[])

    for len_ids_up, pos in size_with_pos_sorted:
        # len_ids = (len_ids + len_round_up - 1) // len_round_up * len_round_up
        # cur_cost = cur_batch.size * cur_batch.max_length * cur_batch.max_length
        next_cost = (cur_batch.size + 1) * len_ids_up * len_ids_up
        extra_cost = (cur_batch.size + 1) * ((len_ids_up - cur_batch.max_length) ** 2)

        if cur_batch.size > 0 and (next_cost > max_size2 or extra_cost > batch_cost):
            res.append(cur_batch)
            cur_batch = TokenIdsBatch(size=0, max_length=0, input_ids=None, attention_mask=None, src_list_pos=[])

        cur_batch.src_list_pos.append(pos)
        cur_batch.size += 1
        cur_batch.max_length = len_ids_up

    res.append(cur_batch)

    for move_border_iter in range(64):
        for batch_idx in range(len(res) - 1):
            # try move item from batch_i to batch_i+1
            batch = res[batch_idx]
            next_batch = res[batch_idx + 1]

            if batch.size > 1:
                item_pos = batch.src_list_pos[-1]
                item_len = size_with_pos[item_pos][0]

                cur_cost = batch.size * batch.max_length + next_batch.size * next_batch.max_length
                changed_cost = (batch.size - 1) * item_len + (next_batch.size + 1) * next_batch.max_length

                if changed_cost < cur_cost:
                    next_batch.src_list_pos = [item_pos] + next_batch.src_list_pos
                    next_batch.size += 1

                    batch.src_list_pos = batch.src_list_pos[:-1]
                    batch.size -= 1
                    batch.max_length = item_len

            # try moving itm from next to cur batch:
            if next_batch.size > 1:
                item_pos = next_batch.src_list_pos[0]
                item_len = size_with_pos[item_pos][0]

                cur_cost = batch.size * batch.max_length + next_batch.size * next_batch.max_length
                changed_cost = (batch.size + 1) * item_len + (next_batch.size - 1) * next_batch.max_length

                if changed_cost < cur_cost:
                    batch.src_list_pos.append(item_pos)
                    batch.size += 1
                    batch.max_length = item_len

                    next_batch.src_list_pos = next_batch.src_list_pos[1:]
                    next_batch.size -= 1

    for batch in res:
        cur_attention_mask = []
        cur_input_ids = []

        for i, pos in enumerate(batch.src_list_pos):
            ids = input_ids[pos]
            cur_len = len(ids)
            pad_len = batch.max_length - cur_len
            assert pad_len >= 0

            cur_attention_mask.append([1] * cur_len + [0] * pad_len)
            cur_input_ids.append(ids + [1] * pad_len)

        batch.input_ids = torch.tensor(cur_input_ids, dtype=torch.long)
        batch.attention_mask = torch.tensor(cur_attention_mask, dtype=torch.long)

    return res


def restore_order_idx(batches: [TokenIdsBatch]):
    total_size = sum([b.size for b in batches])
    res = [-1] * total_size
    b_offset = 0
    for batch in batches:
        for i, pos in enumerate(batch.src_list_pos):
            res[pos] = b_offset + i
        b_offset += batch.size

    return res


def test_split_token_ids_to_batches():
    input_ids = [list([k] * k) for k in range(18, 2, -1)]

    batches = split_token_ids_to_batches(input_ids=input_ids, batch_cost=64, max_len=16, max_size2=16 * 16 * 2, len_round_up=2)

    ids_count = 0
    positions = set()

    for batch in batches:
        for i in range(batch.size):
            assert len(batch.input_ids[i]) == batch.max_length
            assert len(batch.attention_mask[i]) == batch.max_length
        ids_count += len(batch.src_list_pos)
        positions.update(batch.src_list_pos)

    assert ids_count == len(input_ids)
    assert positions == set(range(len(input_ids)))

    combined_pred = torch.cat([batch.input_ids[:, :2] for batch in batches], dim=0)

    order_idx = restore_order_idx(batches)
    assert -1 not in order_idx
    assert len(order_idx) == combined_pred.shape[0]

    combined_pred_restored_order = combined_pred[order_idx, :]
    print(combined_pred_restored_order)

    for i, input_id in enumerate(input_ids):
        assert combined_pred_restored_order[i, 0] == input_id[0]
    # print(batches)


def load_cell_order(fn):
    if not os.path.exists(fn):
        pd.read_csv(fn[:-3] + 'csv', dtype=str).set_index('id', drop=True).to_pickle(fn)

    return pd.read_pickle(fn)['cell_order'].to_dict()


class CodeDataset(Dataset):
    def __init__(self,
                 fold:int,
                 is_training:bool,
                 code_tokenizer_name:str,
                 md_tokenizer_name:str,
                 return_order_only=False,
                 max_code_tokens_number=128,
                 max_md_tokens_number=128,
                 nb_code_cells=64,
                 nb_md_cells=64,
                 dataset_type='train',
                 test_data_dir=None,
                 test_ids=None,
                 cell_prefix='',
                 preprocess_md='',
                 preprocess_code='',
                 use_relaxed_md_right_before_code=False,
                 use_pos_between_code=True,
                 batch_cost=128 * 128,
                 max_size2=128 * 128 * 64,
                 low_case_md=True,
                 low_case_code=True
                 ):

        self.is_training = is_training
        self.fold = fold
        self.dataset_type = dataset_type
        self.cell_prefix = cell_prefix
        self.use_relaxed_md_right_before_code = use_relaxed_md_right_before_code
        self.use_pos_between_code = use_pos_between_code
        self.low_case_md = low_case_md
        self.low_case_code = low_case_code

        self.batch_cost = batch_cost
        self.max_size2 = max_size2

        self.nb_code_cells = nb_code_cells
        self.nb_mb_cells = nb_md_cells

        self.preprocess_md = preprocess_md
        self.preprocess_code = preprocess_code

        self.max_md_tokens_number = max_md_tokens_number
        self.max_code_tokens_number = max_code_tokens_number

        self.return_order_only = return_order_only

        if not return_order_only:
            if code_tokenizer_name.startswith('cubert'):
                sys.path.append("../input/cubert/tokenizer")
                import full_tokenizer as full_tokenizer_cubert
                self.code_tokenizer = full_tokenizer_cubert.CuBertHugTokenizer(vocab_file=f"../input/cubert/{code_tokenizer_name}/vocab.txt")
            else:
                self.code_tokenizer = transformers.AutoTokenizer.from_pretrained(code_tokenizer_name)

            if md_tokenizer_name.startswith('cubert'):
                sys.path.append("../input/cubert/tokenizer")
                import full_tokenizer as full_tokenizer_cubert
                self.md_tokenizer = full_tokenizer_cubert.CuBertHugTokenizer(vocab_file=f"../input/cubert/{md_tokenizer_name}/vocab.txt")
            else:
                self.md_tokenizer = transformers.AutoTokenizer.from_pretrained(md_tokenizer_name)

        if self.dataset_type == 'test':
            self.items = test_ids
            self.train_orders = None
            self.samples_dir = test_data_dir
        elif self.dataset_type == 'extra_data':
            self.train_orders = load_cell_order(f'{config.DATA_DIR}/extra_data/train_orders.pkl')
            self.items = list(sorted(list(self.train_orders.keys())))
            self.samples_dir = f'{config.DATA_DIR}/extra_data/notebooks'
        else:
            folds = pd.read_csv(f'{config.DATA_DIR}/folds.csv')

            if is_training:
                folds = folds[folds.fold != fold]
            else:
                folds = folds[folds.fold == fold]

            self.train_orders = load_cell_order(f'{config.DATA_DIR}/train_orders.pkl')
            self.items = list(sorted(folds['id'].values))
            self.samples_dir = f'{config.DATA_DIR}/{dataset_type}'

        print(f'Fold {fold} training: {self.is_training} items {len(self.items)}')

    def preprocess_md_src(self, text):
        if self.preprocess_md == '':
            return text
        elif self.preprocess_md == 'v1':
            return preprocess_text_v1(text)
        elif self.preprocess_md == 'v2':
            return preprocess_text_v2(text)
        else:
            raise Exception(f'Invalid text preprocess method:{self.preprocess_md}')

    def preprocess_code_src(self, text):
        if self.preprocess_code == '':
            return text
        elif self.preprocess_code == 't5_python_tokenizer':
            return t5_python_tokenizer(text)
        else:
            raise Exception(f'Invalid text preprocess method:{self.preprocess_code}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        item_id = self.items[item]
        data = json.load(open(f'{self.samples_dir}/{item_id}.json'))

        if self.dataset_type == 'test':
            train_orders = list(data['source'])
        else:
            train_orders = self.train_orders[item_id].split()

        cells_code = []
        cells_md = []
        for rank, key in enumerate(train_orders):
            source = data['source'][key]
            cell_type = data['cell_type'][key]

            if cell_type == 'code':
                cells_code.append([source, rank, key])
            else:
                cells_md.append([source, rank, key])

        if len(cells_code) > self.nb_code_cells:
            if self.is_training:
                offset = np.random.randint(0, len(cells_code) - self.nb_code_cells)
            else:
                offset = 0
            cells_code = cells_code[offset:offset + self.nb_code_cells]

        if len(cells_md) > self.nb_mb_cells:
            if self.is_training:
                random.shuffle(cells_md)
            cells_md = cells_md[:self.nb_mb_cells]

        nb_code = len(cells_code)
        nb_md = len(cells_md)

        rank_code = np.array([c[1] for c in cells_code])
        rank_md = np.array([c[1] for c in cells_md])

        md_after_md = (rank_md[:, None] > rank_md[None, :]).astype(np.float32)

        md_right_before_md = (rank_md[None, :] - rank_md[:, None] == 1).astype(np.float32)
        md_right_after_md = (rank_md[:, None] - rank_md[None, :] == 1).astype(np.float32)

        if self.use_pos_between_code:
            rank_between_code = np.array([-1] + [c[1] for c in cells_code])

            md_after_code = (rank_md[:, None] > rank_between_code[None, :]).astype(np.float32)
            md_between_code = np.clip(np.diff(md_after_code, axis=1, append=0) * -1, 0, 1)

            md_right_after_code = md_between_code
            md_right_before_code = md_between_code
        else:
            md_after_code = (rank_md[:, None] > rank_code[None, :]).astype(np.float32)
            md_between_code = None

            if self.use_relaxed_md_right_before_code:
                # md exactly before/after code, NOT taking into account other md
                md_right_before_code = np.clip(np.diff(md_after_code, axis=1, prepend=1) * -1, 0, 1)
                md_right_after_code = np.clip(np.diff(md_after_code, axis=1, append=0) * -1, 0, 1)

                # md_right_before_md = np.clip(np.diff(md_after_md, axis=1, prepend=1) * -1, 0, 1)
                # md_right_after_md = np.clip(np.diff(md_after_md, axis=1, append=0) * -1, 0, 1)
            else:
                # md exactly before/after code, taking into account other md
                md_right_before_code = (rank_code[None, :] - rank_md[:, None] == 1).astype(np.float32)
                md_right_after_code = (rank_md[:, None] - rank_code[None, :] == 1).astype(np.float32)

        tokens_code = []
        # tokens_code_mask = []
        tokens_md = []
        tokens_md_mask = []

        keys_code = []
        keys_md = []

        if self.return_order_only:
            for cell_source, rank, key in cells_code:
                keys_code.append(key)

            for cell_source, rank, key in cells_md:
                keys_md.append(key)

            return dict(
                item=item,
                item_id=item_id,

                md_after_code=md_after_code,
                md_right_before_code=md_right_before_code,
                md_right_after_code=md_right_after_code,

                md_between_code=md_between_code,

                md_after_md=md_after_md,
                md_right_before_md=md_right_before_md,
                md_right_after_md=md_right_after_md,

                rank_code=rank_code,
                rank_md=rank_md,
                keys_code=keys_code,
                keys_md=keys_md,
                keys_all_sorted=[k for k in train_orders if (k in keys_code) or (k in keys_md)]
            )

        for cell_source, rank, key in cells_code:
            if self.low_case_code:
                cell_source = cell_source.lower()

            try:
                tokens = self.code_tokenizer.encode_plus(
                    self.cell_prefix + self.preprocess_code_src(cell_source),
                    max_length=self.max_code_tokens_number,
                    truncation=True,
                    add_special_tokens=True
                )
            except:
                tokens = self.code_tokenizer.encode_plus(
                    self.cell_prefix + '',
                    max_length=self.max_code_tokens_number,
                    truncation=True,
                    add_special_tokens=True
                )
            tokens_code.append(tokens['input_ids'])
            # tokens_code_mask.append(tokens['attention_mask'])
            keys_code.append(key)

        code_batches = split_token_ids_to_batches(
            input_ids=tokens_code,
            max_len=self.max_code_tokens_number,
            batch_cost=self.batch_cost,
            max_size2=self.max_size2
        )

        for cell_source, rank, key in cells_md:
            if self.low_case_md:
                cell_source = cell_source.lower()

            try:
                tokens = self.md_tokenizer.encode_plus(
                    self.cell_prefix + self.preprocess_md_src(cell_source),
                    max_length=self.max_md_tokens_number,
                    truncation=True,
                    add_special_tokens=True
                )
            except:
                tokens = self.md_tokenizer.encode_plus(
                    self.cell_prefix + '',
                    max_length=self.max_md_tokens_number,
                    truncation=True,
                    add_special_tokens=True
                )
            tokens_md.append(tokens['input_ids'])
            # tokens_md_mask.append(tokens['attention_mask'])
            keys_md.append(key)

        md_batches = split_token_ids_to_batches(
            input_ids=tokens_md,
            max_len=self.max_md_tokens_number,
            batch_cost=self.batch_cost,
            max_size2=self.max_size2
        )

        res = dict(
            item=item,
            item_id=item_id,

            tokens_code=code_batches,
            tokens_md=md_batches,

            md_after_code=md_after_code,
            md_right_before_code=md_right_before_code,
            md_right_after_code=md_right_after_code,

            md_between_code=md_between_code,

            md_after_md=md_after_md,
            md_right_before_md=md_right_before_md,
            md_right_after_md=md_right_after_md,

            rank_code=rank_code,
            rank_md=rank_md,
            keys_code=keys_code,
            keys_md=keys_md,
            keys_all_sorted=[k for k in train_orders if (k in keys_code) or (k in keys_md)]
        )

        return res


class CodeDatasetTest(Dataset):
    def __init__(self,
                 code_tokenizer_name: str,
                 md_tokenizer_name: str,
                 max_code_tokens_number=256,
                 max_md_tokens_number=256,
                 test_data_dir=None,
                 test_ids=None,
                 batch_cost=32768,
                 max_size2=256 * 256 * 8,
                 low_case_md=False,
                 low_case_code=False
                 ):

        self.low_case_md = low_case_md
        self.low_case_code = low_case_code

        self.batch_cost = batch_cost
        self.max_size2 = max_size2

        self.max_md_tokens_number = max_md_tokens_number
        self.max_code_tokens_number = max_code_tokens_number

        self.code_tokenizer = transformers.AutoTokenizer.from_pretrained(code_tokenizer_name)
        self.md_tokenizer = transformers.AutoTokenizer.from_pretrained(md_tokenizer_name)

        self.items = test_ids
        self.train_orders = None
        self.samples_dir = test_data_dir

        print(f'Items {len(self.items)}')

    def preprocess_md_src(self, text):
        return text

    def preprocess_code_src(self, text):
        return text

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        item_id = self.items[item]
        data = json.load(open(f'{self.samples_dir}/{item_id}.json'))
        train_orders = list(data['source'])

        cells_code = []
        cells_md = []
        for rank, key in enumerate(train_orders):
            source = data['source'][key]
            cell_type = data['cell_type'][key]

            if cell_type == 'code':
                cells_code.append([source, rank, key])
            else:
                cells_md.append([source, rank, key])

        tokens_code = []
        tokens_md = []

        # make sure we have at least one code or md cell
        if len(cells_code) == 0:
            cells_code.append(['', len(cells_md), ''])

        if len(cells_md) == 0:
            cells_md.append(['', len(cells_code), ''])

        keys_code = []
        keys_md = []

        for cell_source, rank, key in cells_code:
            if self.low_case_code:
                cell_source = cell_source.lower()
            try:
                tokens = self.code_tokenizer.encode_plus(
                    self.preprocess_code_src(cell_source),
                    max_length=self.max_code_tokens_number,
                    truncation=True,
                    add_special_tokens=True
                )
            except:
                tokens = self.code_tokenizer.encode_plus(
                    '',
                    max_length=self.max_code_tokens_number,
                    truncation=True,
                    add_special_tokens=True
                )
            tokens_code.append(tokens['input_ids'])
            keys_code.append(key)

        code_batches = split_token_ids_to_batches(
            input_ids=tokens_code,
            max_len=self.max_code_tokens_number,
            batch_cost=self.batch_cost,
            max_size2=self.max_size2
        )

        for cell_source, rank, key in cells_md:
            if self.low_case_md:
                cell_source = cell_source.lower()

            try:
                tokens = self.md_tokenizer.encode_plus(
                    self.preprocess_md_src(cell_source),
                    max_length=self.max_md_tokens_number,
                    truncation=True,
                    add_special_tokens=True
                )
            except:
                tokens = self.md_tokenizer.encode_plus(
                    '',
                    max_length=self.max_md_tokens_number,
                    truncation=True,
                    add_special_tokens=True
                )
            tokens_md.append(tokens['input_ids'])
            keys_md.append(key)

        md_batches = split_token_ids_to_batches(
            input_ids=tokens_md,
            max_len=self.max_md_tokens_number,
            batch_cost=self.batch_cost,
            max_size2=self.max_size2
        )

        res = dict(
            item=item,
            item_id=item_id,

            tokens_code=code_batches,
            tokens_md=md_batches,

            keys_code=keys_code,
            keys_md=keys_md,
            keys_all_sorted=[k for k in train_orders if (k in keys_code) or (k in keys_md)]
        )

        return res


class CodeDatasetL2(Dataset):
    def __init__(
            self,
            fold,
            is_training,
            data_dir,
            extra_data_dir='',
            dataset_type='train',
            add_empty_cells_p=0.0,
            add_empty_cells_norm=0.33,
    ):

        self.is_training = is_training
        self.fold = fold
        self.dataset_type = dataset_type
        self.add_empty_cells_p = add_empty_cells_p if is_training else 0.0
        self.add_empty_cells_norm = add_empty_cells_norm

        folds = pd.read_csv(f'{config.DATA_DIR}/folds.csv')

        if is_training:
            folds = folds[folds.fold != fold]
        else:
            folds = folds[folds.fold == fold]

        self.train_orders = load_cell_order(f'{config.DATA_DIR}/train_orders.pkl')
        self.items = list(sorted(folds['id'].values))
        self.data_dir = data_dir

        self.extra_data_dir = extra_data_dir
        self.extra_items = []

        if is_training and extra_data_dir != '':
            self.extra_items = [
                item[:-4] for item in sorted(os.listdir(extra_data_dir)) if item.endswith('.npz')
            ]

        print(f'Fold {fold} training: {self.is_training} items {len(self.items)} extra items {len(self.extra_items)}')

    def __len__(self):
        return len(self.items) + len(self.extra_items)

    def __getitem__(self, item):
        if item < len(self.items):
            item_id = self.items[item]
            data = np.load(f'{self.data_dir}/{item_id}.npz')
            # train_orders = data['keys_all_sorted']
        else:
            item_id = self.extra_items[item - len(self.items)]
            data = np.load(f'{self.extra_data_dir}/{item_id}.npz')

        pred_code = data['pred_code']
        pred_md = data['pred_md']
        keys_all_sorted = list(data['keys_all_sorted'])
        train_orders = keys_all_sorted
        keys_code = list(data['keys_code'])
        keys_md = list(data['keys_md'])

        if item < len(self.items) and self.add_empty_cells_p > 0.0 and np.random.rand() < self.add_empty_cells_p:
            all_items = []
            for k in keys_all_sorted:
                if k in keys_code:
                    all_items.append(['c', k, pred_code[keys_code.index(k)]])
                else:
                    all_items.append(['m', k, pred_md[keys_md.index(k)]])

            empty_cell = np.load(f'{self.data_dir}/empty.npz')['pred_code']
            nb_empty_cells = int(len(keys_code) * self.add_empty_cells_norm * np.clip(np.abs(np.random.normal()), 0, 3))
            for i in range(nb_empty_cells):
                all_items.insert(np.random.randint(len(all_items)),
                                 ['c', f'empty_{i}', empty_cell])

            pred_code = []
            pred_md = []
            keys_all_sorted = []
            keys_code = []
            keys_md = []

            for item_type, key, item_pred in all_items:
                keys_all_sorted.append(key)
                if item_type == 'c':
                    pred_code.append(item_pred)
                    keys_code.append(key)
                else:
                    pred_md.append(item_pred)
                    keys_md.append(key)

            pred_code = np.array(pred_code)
            pred_md = np.array(pred_md)
            train_orders = keys_all_sorted

        rank_code = np.array([keys_all_sorted.index(c) for c in keys_code])
        rank_md = np.array([keys_all_sorted.index(c) for c in keys_md])

        md_after_md = (rank_md[:, None] > rank_md[None, :]).astype(np.float32)

        md_right_before_md = (rank_md[None, :] - rank_md[:, None] == 1).astype(np.float32)
        md_right_after_md = (rank_md[:, None] - rank_md[None, :] == 1).astype(np.float32)

        rank_between_code = np.array([-1] + [keys_all_sorted.index(c) for c in keys_code])

        md_after_code = (rank_md[:, None] > rank_between_code[None, :]).astype(np.float32)
        md_between_code = np.clip(np.diff(md_after_code, axis=1, append=0) * -1, 0, 1)

        # max_rank = max(max(rank_code), max(rank_md))
        # rank_md_relative = rank_md * 1.0 / max_rank

        res = dict(
            item=item,
            item_id=item_id,

            activations_code=pred_code,
            activations_md=pred_md,

            md_after_code=md_after_code,
            md_between_code=md_between_code,
            md_right_before_code=md_between_code,
            md_right_after_code=md_between_code,

            md_after_md=md_after_md,
            md_right_before_md=md_right_before_md,
            md_right_after_md=md_right_after_md,

            rank_code=rank_code,
            rank_md=rank_md,
            # rank_md_relative=rank_md_relative,
            keys_code=keys_code,
            keys_md=keys_md,
            keys_all_sorted=[k for k in train_orders if (k in keys_code) or (k in keys_md)]
        )

        return res


def check_dataset():
    ds = CodeDataset(
        fold=0, is_training=False,
        code_tokenizer_name='microsoft/codebert-base',
        md_tokenizer_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        max_code_tokens_number=256,
        max_md_tokens_number=256,
        nb_code_cells=1024,
        nb_md_cells=1024,
        batch_cost=32768,
        max_size2=524288,
        cell_prefix='',
        preprocess_md='',
        use_pos_between_code=True,
        low_case_md=False,
        low_case_code=False
    )

    for i, sample in enumerate(ds):
        orig_keys = sample['keys_all_sorted']
        # restored_keys = restore_order(
        #     keys_code=sample['keys_code'],
        #     keys_md=sample['keys_md'],
        #     md_after_code=sample['md_after_code'],
        #     md_after_md=sample['md_after_md']
        # )

        restored_keys = restore_order_sm(
            keys_code=sample['keys_code'],
            keys_md=sample['keys_md'],
            md_after_code=sample['md_after_code'],
            md_after_md=sample['md_after_md'],
            md_between_code=sample['md_between_code']
        )

        # restored_keys = restore_order_v2(
        #     keys_code=sample['keys_code'],
        #     keys_md=sample['keys_md'],
        #     md_after_code=sample['md_after_code'],
        #     md_after_md=sample['md_after_md'],
        #     md_between_code=sample['md_between_code']
        # )

        print(orig_keys)
        print(restored_keys)
        print(orig_keys == restored_keys)
        print(sample['rank_code'])
        print(sample['rank_md'])

        fix, ax = plt.subplots(1, 3)

        for j, output_name in enumerate(['md_after_code', 'md_between_code', 'md_after_md']):
            ax[j].set_title(output_name.replace('_', ' '))
            ax[j].set(xlabel='md' if output_name == 'md_after_md' else 'code',
                      ylabel='md')
            ax[j].imshow(sample[output_name], vmin=0, vmax=1)

        plt.show()


def profile_dataset_cubert():
    ds = CodeDataset(
        fold=0, is_training=False,
        code_tokenizer_name='cubert-512',
        md_tokenizer_name='cubert-512',
        max_code_tokens_number=256,
        max_md_tokens_number=256,
        nb_code_cells=1024,
        nb_md_cells=1024,
        batch_cost=32768,
        max_size2=524288,
        cell_prefix='',
        preprocess_md='',
        use_pos_between_code=True,
        low_case_md=False,
        low_case_code=False
    )

    for i, sample in tqdm(enumerate(ds)):
        orig_keys = sample['keys_all_sorted']
        if i > 256:
            break



def check_dataset_l2():
    ds = CodeDatasetL2(
        fold=0, is_training=False,
        dataset_type='train',
        data_dir='../../ai4code_output/decoder/180_bert_loss_between_code'
    )

    for i, sample in enumerate(ds):
        orig_keys = sample['keys_all_sorted']
        # restored_keys = restore_order(
        #     keys_code=sample['keys_code'],
        #     keys_md=sample['keys_md'],
        #     md_after_code=sample['md_after_code'],
        #     md_after_md=sample['md_after_md']
        # )

        restored_keys = restore_order_sm(
            keys_code=sample['keys_code'],
            keys_md=sample['keys_md'],
            md_after_code=sample['md_after_code'],
            md_after_md=sample['md_after_md'],
            md_between_code=sample['md_between_code']
        )

        # restored_keys = restore_order_v2(
        #     keys_code=sample['keys_code'],
        #     keys_md=sample['keys_md'],
        #     md_after_code=sample['md_after_code'],
        #     md_after_md=sample['md_after_md'],
        #     md_between_code=sample['md_between_code']
        # )

        print(orig_keys)
        print(restored_keys)
        print(orig_keys == restored_keys)
        print(sample['rank_code'])
        print(sample['rank_md'])

        fix, ax = plt.subplots(2, 3)

        for j, output_name in enumerate(['md_after_code', 'md_right_before_code', 'md_right_after_code',
                                         'md_after_md', 'md_right_before_md', 'md_right_after_md']):
            ax[j % 2, j // 2].set_title(output_name.replace('_', ' '))
            ax[j % 2, j // 2].imshow(sample[output_name], vmin=0, vmax=1)

        plt.show()
        # print(sample['item'])
        # if i > 3:
        #     break


def check_performance():
    ds = CodeDataset(
        fold=0, is_training=False,

        code_tokenizer_name= 'microsoft/codebert-base',
        md_tokenizer_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        max_code_tokens_number=256,
        max_md_tokens_number=256,
        nb_code_cells=1024,
        nb_md_cells=1024,
        batch_cost=32768,
        max_size2=524288,
        cell_prefix='',
        preprocess_md='',
        use_pos_between_code=True,
        low_case_md=False,
        low_case_code=False
    )
        
    #
    #     code_tokenizer_name='microsoft/codebert-base',
    #     md_tokenizer_name='microsoft/codebert-base'
    # )
    print()
    with utils.timeit_context('run'):
        for sample in tqdm(ds, total=len(ds)):
            pass


if __name__ == '__main__':
    # test_split_token_ids_to_batches()
    check_dataset()
    # profile_dataset_cubert()
    pass
