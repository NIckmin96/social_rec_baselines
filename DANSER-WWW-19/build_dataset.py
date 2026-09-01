import os
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--neg_train', action='store_true',
                        help='Enable negative sampling for train.')
    parser.add_argument('--no_neg_test', action='store_true',
                        help='Disable negative sampling for test.')
    args = parser.parse_args()
    return args


def add_negs(rating_split, rating_matrix, rng):
    user_lst, neg_lst = [], []
    all_items = np.arange(rating_matrix.shape[1])
    for user, group in rating_split.groupby('user_id'):
        items = group['product_id']
        nnz = rating_matrix[user].indices
        zero_indices = np.setdiff1d(all_items, nnz)
        if zero_indices.size == 0:
            continue
        n = len(items)
        n_samples = max(10 - n, n)
        n_samples = min(n_samples, zero_indices.size)
        neg_samples = list(rng.choice(zero_indices, size=n_samples, replace=False))
        user_lst.extend([user] * n_samples)
        neg_lst.extend(neg_samples)

    if not neg_lst:
        return rating_split

    rating_lst = [0] * len(neg_lst)
    neg_df = pd.DataFrame(
        {'user_id': user_lst, 'product_id': neg_lst, 'rating': rating_lst}
    )
    rating_split = pd.concat([rating_split, neg_df], axis=0, ignore_index=True)
    return rating_split


def load_and_encode(data_dir):
    rating = pd.read_csv(os.path.join(data_dir, 'rating_org.csv'))
    trust = pd.read_csv(os.path.join(data_dir, 'trustnetwork_org.csv'))

    rating = rating.dropna(how='any')
    rating = rating.drop_duplicates(['user_id', 'product_id'], keep='first')
    rating = rating[rating['rating'].between(1, 5, inclusive='both')]
    trust = trust.dropna(how='any')
    trust = trust.drop_duplicates(keep='first')

    total_users = (
        set(trust.user_id_1.unique())
        .union(set(trust.user_id_2.unique()))
        .intersection(set(rating.user_id.unique()))
    )
    rating = rating[rating.user_id.isin(total_users)]
    trust = trust[trust.user_id_1.isin(total_users) & trust.user_id_2.isin(total_users)]

    total_users = (
        set(trust.user_id_1.unique())
        .union(set(trust.user_id_2.unique()))
        .intersection(set(rating.user_id.unique()))
    )
    rating = rating[rating.user_id.isin(total_users)]
    trust = trust[trust.user_id_1.isin(total_users) & trust.user_id_2.isin(total_users)]

    total_users = set(trust.user_id_1.unique()).union(set(trust.user_id_2.unique())).union(set(rating.user_id.unique()))
    total_items = rating.product_id.unique()
    item_encode = dict(zip(total_items, range(len(total_items))))

    rating_users = set(rating.user_id.unique())
    trust_users = total_users - rating_users
    if trust_users:
        r_user_encode = dict(zip(rating_users, range(len(rating_users))))
        t_user_encode = dict(
            zip(trust_users, range(len(rating_users) + 1, len(total_users) + 1))
        )
        user_encode = r_user_encode | t_user_encode
    else:
        user_encode = dict(zip(total_users, range(len(total_users))))

    rating['user_id'] = rating['user_id'].map(user_encode).astype(np.int32)
    trust['user_id_1'] = trust['user_id_1'].map(user_encode).astype(np.int32)
    trust['user_id_2'] = trust['user_id_2'].map(user_encode).astype(np.int32)

    rating['product_id'] = rating['product_id'].map(item_encode).astype(np.int32)
    trust['value'] = 1.0

    rating = rating[['user_id', 'product_id', 'rating']]
    trust = trust[['user_id_1', 'user_id_2']]

    return rating, trust


def create_dataset(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = os.path.join('data', args.dataset)
    rating, trust = load_and_encode(data_dir)

    user_count = int(rating['user_id'].max())
    item_count = int(rating['product_id'].max())

    rating_matrix = sparse.csr_matrix(
        ([1] * rating.shape[0], (rating['user_id'], rating['product_id'])),
        shape=(user_count + 1, item_count + 1),
    )

    shuffled = rating.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test_size = int(len(shuffled) * 0.2)
    rating_test = shuffled[:test_size].copy()
    rating_train = shuffled[test_size:].copy()

    rng = np.random.default_rng(args.seed)
    if args.neg_train:
        rating_train = add_negs(rating_train, rating_matrix, rng)
    if not args.no_neg_test:
        rating_test = add_negs(rating_test, rating_matrix, rng)

    train_set = list(
        rating_train[['user_id', 'product_id', 'rating']].itertuples(index=False, name=None)
    )
    test_set = list(
        rating_test[['user_id', 'product_id', 'rating']].itertuples(index=False, name=None)
    )
    print("Length of Train set :", len(train_set))
    print("Length of Test set :", len(test_set))

    with open(f'data/{args.dataset}/dataset_{args.seed}.pkl', 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

    train_df = pd.DataFrame(train_set, columns=['uid', 'iid', 'label'])
    train_pos_df = train_df[train_df['label'] > 0]

    u_read_list = []
    i_read_list = []
    u_friend_list = []
    uf_read_list = []
    i_friend_list = []
    if_read_list = []
    i_link_list = []

    train_pos_df = train_pos_df.sort_values(axis=0, ascending=True, by='uid')
    for u in range(user_count + 1):
        hist = train_pos_df[train_pos_df['uid'] == u]
        u_read = hist['iid'].unique().tolist()
        if u_read == []:
            u_read_list.append([0])
        else:
            u_read_list.append(u_read)

    train_pos_df = train_pos_df.sort_values(axis=0, ascending=True, by='iid')
    for i in range(item_count + 1):
        hist = train_pos_df[train_pos_df['iid'] == i]
        i_read = hist['uid'].unique().tolist()
        if i_read == []:
            i_read_list.append([0])
        else:
            i_read_list.append(i_read)

    trust_list = []
    for _, row in trust.iterrows():
        uid = int(row['user_id_1'])
        fid = int(row['user_id_2'])
        if uid > user_count or fid > user_count:
            continue
        trust_list.append([uid, fid])

    trust_df = pd.DataFrame(trust_list, columns=['uid', 'fid'])
    trust_df = trust_df.sort_values(axis=0, ascending=True, by='uid')

    for u in range(user_count + 1):
        hist = trust_df[trust_df['uid'] == u]
        u_friend = hist['fid'].unique().tolist()
        if u_friend == []:
            u_friend_list.append([0])
            uf_read_list.append([[0]])
        else:
            u_friend_list.append(u_friend)
            uf_read_f = []
            for f in u_friend:
                uf_read_f.append(u_read_list[f])
            uf_read_list.append(uf_read_f)

    for i in range(item_count + 1):
        if len(i_read_list[i]) <= 30:
            i_friend_list.append([0])
            if_read_list.append([[0]])
            i_link_list.append([0])
            continue
        i_friend = []
        for j in range(item_count + 1):
            if len(i_read_list[j]) <= 30:
                sim_ij = 0
            else:
                sim_ij = 0
                for s in i_read_list[i]:
                    sim_ij += np.sum(i_read_list[j] == s)
            i_friend.append([j, sim_ij])
        i_friend_cd = sorted(i_friend, key=lambda d: d[1], reverse=True)
        i_friend_i = []
        i_link_i = []
        for k in range(20):
            if i_friend_cd[k][1] > 5:
                i_friend_i.append(i_friend_cd[k][0])
                i_link_i.append(i_friend_cd[k][1])
        if i_friend_i == []:
            i_friend_list.append([0])
            if_read_list.append([[0]])
            i_link_list.append([0])
        else:
            i_friend_list.append(i_friend_i)
            i_link_list.append(i_link_i)
            if_read_f = []
            for f in i_friend_i:
                if_read_f.append(i_read_list[f])
            if_read_list.append(if_read_f)

    with open(f'data/{args.dataset}/list_{args.seed}.pkl', 'wb') as f:
        pickle.dump(u_friend_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(u_read_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(uf_read_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(i_friend_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(i_read_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(if_read_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(i_link_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((user_count, item_count), f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = get_args()
    create_dataset(args)
