# -*- coding: utf-8 -*-
"""
GDSRec preprocessing.

Filtering / id-remap / the 8:1:1 split come from the shared `common_preprocess`
stage (identical across every benchmark baseline). Everything after the split --
the per-user / per-item average & history lists, the trust-similarity lists -- is
GDSRec's original logic, unchanged.

Outputs (under datasets/<dataset>/):
    dataset_<sigma>.pkl   train_set / valid_set / test_set  (lists of (uid, iid, label))
    list_<sigma>.pkl      the aggregate lists + (user_count, item_count, rate_count)
"""
import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_preprocess.common import load_common  # noqa: E402

workdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def _tuples(df):
    return list(zip(df["user_id"].astype(int), df["item_id"].astype(int), df["rating"].astype(int)))


def process_gdsrec(dataset, seed, sigma="0"):
    inter, social, splits, stats = load_common(dataset, seed)
    user_count = int(stats["n_users"])
    item_count = int(stats["n_items"])
    rate_count = int(inter["rating"].max())

    train_set = _tuples(splits["train"])
    valid_set = _tuples(splits["valid"])
    test_set = _tuples(splits["test"])
    print("Train samples: {}, Valid samples: {}, Test samples: {}".format(
        len(train_set), len(valid_set), len(test_set)))

    out_dir = os.path.join(workdir, dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"dataset_{sigma}.pkl"), "wb") as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

    # ---- GDSRec's original aggregate-list construction (unchanged) ----
    train_df = pd.DataFrame(train_set, columns=["uid", "iid", "label"])
    train_df = train_df.sort_values(axis=0, ascending=True, by="uid")
    all_avg = train_df["label"].mean()

    i_avg_list = []
    for i in tqdm(range(item_count + 1)):
        hist = train_df[train_df["iid"] == i]
        i_ratings = hist["label"].tolist()
        i_avg_list.append(all_avg if i_ratings == [] else hist["label"].mean())

    u_avg_list = []
    for u in tqdm(range(user_count + 1)):
        hist = train_df[train_df["uid"] == u]
        u_ratings = hist["label"].tolist()
        u_avg_list.append(all_avg if u_ratings == [] else hist["label"].mean())

    u_items_divlist = []
    for u in tqdm(range(user_count + 1)):
        hist = train_df[train_df["uid"] == u]
        u_items = hist["iid"].tolist()
        u_ratings = hist["label"].tolist()
        if u_items == []:
            u_items_divlist.append([(0, 0)])
        else:
            u_items_divlist.append([(iid, round(abs(rating - i_avg_list[iid])))
                                    for iid, rating in zip(u_items, u_ratings)])

    u_items_list = []
    for u in tqdm(range(user_count + 1)):
        hist = train_df[train_df["uid"] == u]
        u_items = hist["iid"].tolist()
        u_ratings = hist["label"].tolist()
        if u_items == []:
            u_items_list.append([(0, 0)])
        else:
            u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])

    train_df = train_df.sort_values(axis=0, ascending=True, by="iid")

    i_users_divlist = []
    for i in tqdm(range(item_count + 1)):
        hist = train_df[train_df["iid"] == i]
        i_users = hist["uid"].tolist()
        i_ratings = hist["label"].tolist()
        if i_users == []:
            i_users_divlist.append([(0, 0)])
        else:
            i_users_divlist.append([(uid, round(abs(rating - u_avg_list[uid])))
                                    for uid, rating in zip(i_users, i_ratings)])

    i_users_list = []
    for i in tqdm(range(item_count + 1)):
        hist = train_df[train_df["iid"] == i]
        i_users = hist["uid"].tolist()
        i_ratings = hist["label"].tolist()
        if i_users == []:
            i_users_list.append([(0, 0)])
        else:
            i_users_list.append([(uid, rating) for uid, rating in zip(i_users, i_ratings)])

    # trust similarity lists (shared directed edge set)
    trust_df = social.rename(columns={"user_id_1": "uid", "user_id_2": "fid"})
    trust_df = trust_df[(trust_df["uid"] <= user_count) & (trust_df["fid"] <= user_count)]
    trust_df = trust_df.sort_values(axis=0, ascending=True, by="uid")

    u_users_similar = []
    u_users_items_list = []
    u_users_items_divlist = []
    for u in tqdm(range(user_count + 1)):
        u_u_similar = []
        u_info = dict(u_items_list[u])
        hist = trust_df[trust_df["uid"] == u]
        u_users = hist["fid"].unique().tolist()
        if u_users == []:
            u_users_similar.append([(0, 0)])
            u_users_items_list.append([[(0, 0)]])
            u_users_items_divlist.append([[0, 0]])
        else:
            for user in u_users:
                user_info = dict(u_items_list[user])
                inter_list = list(set(user_info.keys()).intersection(set(u_info.keys())))
                inter_count = len(inter_list)
                for item in inter_list:
                    if abs(u_info[item] - user_info[item]) > int(sigma):
                        inter_count -= 1
                u_u_similar.append((user, inter_count + 1))
            if u_u_similar == []:
                u_users_similar.append([(0, 0)])
                u_users_items_list.append([[(0, 0)]])
                u_users_items_divlist.append([[(0, 0)]])
            else:
                u_users_similar.append(u_u_similar)
                uu_items, uu_items_div = [], []
                for (uid, _cnt) in u_u_similar:
                    uu_items.append(u_items_list[uid])
                    uu_items_div.append(u_items_divlist[uid])
                u_users_items_list.append(uu_items)
                u_users_items_divlist.append(uu_items_div)

    with open(os.path.join(out_dir, f"list_{sigma}.pkl"), "wb") as f:
        pickle.dump(u_items_divlist, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(u_items_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(u_avg_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(u_users_similar, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(u_users_items_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(u_users_items_divlist, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(i_avg_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(i_users_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(i_users_divlist, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((user_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)
    print(f"[GDSRec/{dataset}] users {user_count} / items {item_count} / rate {rate_count} -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ciao_timestamp", help="common dataset name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", default="0", help="social strength definition")
    args = parser.parse_args()
    process_gdsrec(args.dataset, args.seed, str(args.sigma))
