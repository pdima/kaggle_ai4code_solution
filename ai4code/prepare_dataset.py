import numpy as np
import pandas as pd
import json
import re
import os
from tqdm import tqdm
import config
import pickle
import matplotlib.pyplot as plt


def split_to_folds():
    if os.path.exists(f'{config.DATA_DIR}/folds.csv'):
        print('folds.csv already exists')
        return

    ancestors = pd.read_csv(f'{config.DATA_DIR}/train_ancestors.csv')

    unique_ancestors = ancestors.ancestor_id.unique().copy()
    np.random.shuffle(unique_ancestors)
    ancestors['fold'] = -1

    for fold in range(config.NB_FOLDS):
        ancestors.loc[ancestors.ancestor_id.isin(unique_ancestors[fold::config.NB_FOLDS]), 'fold'] = fold

    ancestors[['id', 'fold']].to_csv(f'{config.DATA_DIR}/folds.csv', index=False)


def find_number_of_cells_for_files(data_dir):
    document_sizes = {}

    for fn in tqdm(sorted(os.listdir(data_dir))):
        if fn.endswith('.json'):
            item_id = fn[:-5]
            data = json.load(open(f'{data_dir}/{item_id}.json'))

            total_cells = len(data['cell_type'])
            code_cells = 0
            md_cells = 0

            for cell_type in data['cell_type'].values():
                if cell_type == 'code':
                    code_cells += 1
                else:
                    md_cells += 1

            document_sizes[item_id] = (total_cells, code_cells, md_cells)

    return document_sizes


if __name__ == '__main__':
    split_to_folds()

    document_sizes = find_number_of_cells_for_files(f'{config.DATA_DIR}/train')
    dst_fn = f'{config.DATA_DIR}/notebook_size.pkl'
    pickle.dump(document_sizes, open(dst_fn, 'wb'))
    print(dst_fn)

    plt.hist(np.array(list(document_sizes.values()))[:, 0], bins=1024)
    plt.show()

