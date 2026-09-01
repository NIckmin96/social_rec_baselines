"""
TrustMF preprocessing.

Filtering / id-remap / the 8:1:1 split come from the shared `common_preprocess`
stage (identical across every benchmark baseline). This script only writes
TrustMF's model-specific inputs:
  - rating_{train,valid,test}_{seed}.csv   (same filenames main_mf.py expects)
  - trustnetwork.csv                       (directed edge list, consumed as a DataFrame)
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_preprocess.common import load_common  # noqa: E402


def add_negs(rating_split, rating_matrix):
    """Optional TRAIN-only negative augmentation. Never applied to valid/test."""
    user_lst, neg_lst = [], []
    all_items = np.arange(rating_matrix.shape[1])
    for user, group in tqdm(rating_split.groupby('user_id')):
        items = group['product_id']
        nnz = rating_matrix[user].indices
        zero_indices = np.setdiff1d(all_items, nnz)
        n = len(items)
        n_samples = max(10 - n, n)
        neg_samples = list(np.random.choice(zero_indices, size=n_samples, replace=False))
        user_lst.extend([user] * n_samples)
        neg_lst.extend(neg_samples)
    neg_df = pd.DataFrame({'user_id': user_lst, 'product_id': neg_lst, 'rating': [0] * len(neg_lst)})
    return pd.concat([rating_split, neg_df], axis=0)


def _prep(df):
    return (df[['user_id', 'item_id', 'rating']]
            .rename(columns={'item_id': 'product_id'})
            .reset_index(drop=True))


def process_data(data: str, seed: int, neg_train: bool):
    data_dir = os.path.join('data', data)
    dataset = data
    inter, social, splits, stats = load_common(dataset, seed)
    num_users, num_items = stats['n_users'], stats['n_items']

    rating_train = _prep(splits['train'])
    rating_valid = _prep(splits['valid'])
    rating_test = _prep(splits['test'])

    trust = social[['user_id_1', 'user_id_2']].reset_index(drop=True)

    print(f"[TrustMF/{dataset}] users {num_users} / items {num_items} / edges {len(trust)} "
          f"| train {len(rating_train)} valid {len(rating_valid)} test {len(rating_test)}")

    if neg_train:
        rating_all = pd.concat([rating_train, rating_valid, rating_test])
        rating_matrix = sparse.csr_matrix(
            ([1] * rating_all.shape[0], (rating_all['user_id'], rating_all['product_id'])),
            shape=(num_users + 1, num_items + 1),
        )
        rating_train = add_negs(rating_train, rating_matrix)

    train_name = f'rating_train_{seed}_neg.csv' if neg_train else f'rating_train_{seed}.csv'
    rating_train.to_csv(os.path.join(data_dir, train_name), index=False)
    rating_valid.to_csv(os.path.join(data_dir, f'rating_valid_{seed}.csv'), index=False)
    rating_test.to_csv(os.path.join(data_dir, f'rating_test_{seed}.csv'), index=False)
    trust.to_csv(os.path.join(data_dir, 'trustnetwork.csv'), index=False)

    return rating_test, rating_valid, rating_train, trust


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--neg_train', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    process_data(args.dataset, args.seed, args.neg_train)
