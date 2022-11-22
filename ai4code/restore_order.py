import matplotlib.pyplot as plt
import numpy as np


def restore_order_old(keys_code, keys_md, md_after_code, md_after_md, md_between_code):
    nb_code = len(keys_code)
    nb_md = len(keys_md)
    soft_pred = 0.0001

    md_after_code_soft = np.clip(md_after_code * (1 - 2 * soft_pred) + soft_pred, soft_pred, 1.0 - soft_pred).astype(np.float64)
    md_after_code_cost_pos = -1 * np.log(md_after_code_soft)
    md_after_code_cost_neg = -1 * np.log(1 - md_after_code_soft)

    nb_bins = nb_code + 1
    md_bins = [[] for _ in range(nb_bins)]  # places to put md from before the first code to after the last code
    md_after_md = np.mean(md_after_md, axis=1)

    for md_idx in range(nb_md):
        pos_costs = [
            md_after_code_cost_pos[md_idx, :i+1].sum() + md_after_code_cost_neg[md_idx, i+1:].sum()
            for i in range(nb_bins)
        ]
        md_bins[np.argmin(pos_costs)].append(md_idx)

    for bin_idx in range(nb_bins):
        bin = md_bins[bin_idx]
        if len(bin) > 1:
            items_with_order = [(md_after_md[b], b) for b in bin]
            items_with_order = list(sorted(items_with_order))
            md_bins[bin_idx] = [b[1] for b in items_with_order]

    res_order = []
    for md_bin_idx, bin in enumerate(md_bins):
        for md_idx in bin:
            res_order.append(keys_md[md_idx])

        if md_bin_idx < nb_code:
            res_order.append(keys_code[md_bin_idx])

    return res_order


def restore_order_sm(keys_code, keys_md, md_after_code, md_after_md, md_between_code):
    nb_code = len(keys_code)
    nb_md = len(keys_md)

    md_after_code_soft = md_after_code.astype(np.float64)
    md_after_code_cost_pos = -1 * md_after_code_soft
    md_after_code_cost_neg = -1 * (1 - md_after_code_soft)

    nb_bins = nb_code + 1
    md_bins = [[] for _ in range(nb_bins)]  # places to put md from before the first code to after the last code
    # md_after_md = np.mean(md_after_md, axis=1)

    for md_idx in range(nb_md):
        pos_costs = [
            md_after_code_cost_pos[md_idx, :i+1].sum() + md_after_code_cost_neg[md_idx, i+1:].sum() # - md_between_code[md_idx, i] * 2
            for i in range(nb_bins)
        ]
        md_bins[np.argmin(pos_costs)].append(md_idx)

    for bin_idx in range(nb_bins):
        bin = md_bins[bin_idx]
        if len(bin) > 1:
            items_with_order = [(md_after_md[b, bin].mean(), b) for b in bin]
            # items_with_order = [(md_after_md[b], b) for b in bin]
            items_with_order = list(sorted(items_with_order))
            md_bins[bin_idx] = [b[1] for b in items_with_order]

    res_order = []
    for md_bin_idx, bin in enumerate(md_bins):
        for md_idx in bin:
            res_order.append(keys_md[md_idx])

        if md_bin_idx < nb_code:
            res_order.append(keys_code[md_bin_idx])

    return res_order



def restore_order_v2(keys_code, keys_md, md_after_code, md_after_md, md_between_code):
    nb_code = len(keys_code)
    nb_md = len(keys_md)
    soft_pred = 0.0001

    md_after_code_soft = np.clip(md_after_code * (1 - 2 * soft_pred) + soft_pred, soft_pred, 1.0 - soft_pred).astype(np.float64)
    md_after_code_cost_pos = -1 * np.log(md_after_code_soft)
    md_after_code_cost_neg = -1 * np.log(1 - md_after_code_soft)

    nb_bins = nb_code + 1
    md_bins = [[] for _ in range(nb_bins)]  # places to put md from before the first code to after the last code
    md_after_md = np.mean(md_after_md, axis=1)

    for md_idx in range(nb_md):
        # pos_costs = [
        #     md_after_code_cost_pos[md_idx, :i].sum() + md_after_code_cost_neg[md_idx, i:].sum()
        #     for i in range(nb_bins)
        # ]
        # md_bins[np.argmin(pos_costs)].append(md_idx)
        md_bins[np.argmax(md_between_code[md_idx, :])].append(md_idx)

    for bin_idx in range(nb_bins):
        bin = md_bins[bin_idx]
        if len(bin) > 1:
            items_with_order = [(md_after_md[i], b) for i, b in enumerate(bin)]
            items_with_order = list(sorted(items_with_order))
            md_bins[bin_idx] = [b[1] for b in items_with_order]

    res_order = []
    for md_bin_idx, bin in enumerate(md_bins):
        for md_idx in bin:
            res_order.append(keys_md[md_idx])

        if md_bin_idx < nb_code:
            res_order.append(keys_code[md_bin_idx])

    return res_order


def check_restore_order():
    import code_dataset

    ds = code_dataset.CodeDataset(
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
        low_case_code=False,
        return_order_only=True
    )

    for i, sample in enumerate(ds):
        orig_keys = sample['keys_all_sorted']

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
        print('Restored order matched:', orig_keys == restored_keys)
        print(sample['rank_code'])
        print(sample['rank_md'])
        assert(orig_keys == restored_keys)

        fix, ax = plt.subplots(1, 3, squeeze=False)

        for j, output_name in enumerate(['md_after_code', 'md_between_code', 'md_after_md']):
            ax[0, j].set_title(output_name.replace('_', ' '))
            ax[0, j].imshow(sample[output_name], vmin=0, vmax=1)

        plt.show()
        print(sample['item'])
        if i > 64:
            break


if __name__ == '__main__':
    check_restore_order()

