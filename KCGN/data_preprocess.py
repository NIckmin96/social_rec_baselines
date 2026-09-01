"""
KCGN raw-ingestion stage.

Filtering / id-remap now come from the shared `common_preprocess` stage, so KCGN
runs on the SAME user/item universe and the SAME social graph as every other
benchmark baseline. Ids are shifted to 0-based here to match KCGN's internal
convention; everything downstream (dataProcess.py: per-user temporal split, the
>=10 k-core, trust symmetrization, rating-class multi-graph, DGI sub-graphs) is
unchanged -- those remain KCGN's documented exceptions.

Produces the same artifacts the old data_process.py did:
  ratings.pkl / times.pkl / category.pkl / trust.pkl  (+ rating.csv, trustnetwork.csv)
"""
import os
import sys
import pickle
import argparse

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_preprocess.common import load_common  # noqa: E402


def process_kcgn(dataset, seed):
    data_dir = os.path.join('dataset', dataset)
    os.makedirs(data_dir, exist_ok=True)

    inter, social, splits, stats = load_common(dataset, seed)

    rating_df = inter.rename(columns={'item_id': 'product_id'}).copy()
    trust_df = social[['user_id_1', 'user_id_2']].copy()

    # common ids are 1..N -> KCGN wants 0-based
    rating_df['user_id'] -= 1
    rating_df['product_id'] -= 1
    trust_df['user_id_1'] -= 1
    trust_df['user_id_2'] -= 1

    # category: make 0-based dense (KCGN builds an item x category matrix from this)
    rating_df['category_id'] = rating_df['category_id'] - rating_df['category_id'].min()

    n_users = int(stats['n_users'])
    n_items = int(stats['n_items'])
    n_cat = int(rating_df['category_id'].max()) + 1

    u = rating_df['user_id'].to_numpy()
    p = rating_df['product_id'].to_numpy()
    r = rating_df['rating'].to_numpy()
    t = rating_df['timestamp'].to_numpy()

    rating_mat = csr_matrix((r, (u, p)), shape=(n_users, n_items))
    time_mat = csr_matrix((t, (u, p)), shape=(n_users, n_items))
    category_mat = csr_matrix(([1] * len(rating_df), (p, rating_df['category_id'].to_numpy())),
                              shape=(n_items, n_cat))

    # trust: shared directed edges -> symmetrized (KCGN design)
    edges = set(zip(trust_df['user_id_1'], trust_df['user_id_2']))
    edges |= {(v, u_) for u_, v in edges}
    rows = [a for a, _ in edges]
    cols = [b for _, b in edges]
    trust_mat = csr_matrix(([1] * len(edges), (rows, cols)), shape=(n_users, n_users))

    print("###### KCGN Data Source Processed ######")
    print(f"Num of users : {rating_mat.shape[0]}")
    print(f"Num of items : {rating_mat.shape[1]}")
    print(f"Num of categories : {category_mat.shape[1]}")
    print(f"Num of trust (symmetrized) : {trust_mat.nnz}")
    print("########################################\n")

    with open(os.path.join(data_dir, 'ratings.pkl'), 'wb') as f:
        pickle.dump(rating_mat, f)
    with open(os.path.join(data_dir, 'times.pkl'), 'wb') as f:
        pickle.dump(time_mat, f)
    with open(os.path.join(data_dir, 'category.pkl'), 'wb') as f:
        pickle.dump(category_mat, f)
    with open(os.path.join(data_dir, 'trust.pkl'), 'wb') as f:
        pickle.dump(trust_mat, f)

    rating_df.to_csv(os.path.join(data_dir, 'rating.csv'), index=False)
    trust_df.to_csv(os.path.join(data_dir, 'trustnetwork.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    process_kcgn(args.dataset, args.seed)
