"""
Common preprocessing for the social-recommendation benchmark.

This is the SINGLE source of truth for:
  - raw filtering (dedup / rating range / NaN drop; NO k-core)
  - the user/item universe (rating.user  INTERSECT  social users)
  - id remapping to a contiguous 1..N range (saved to *_map.json)
  - the interaction-level 8:1:1 random train/valid/test split (seeded, written to files)

Every baseline consumes the artifacts produced here and then applies its own
model-specific structure building (sparse R, trust adj, random walks, ...) on top.
The per-baseline scripts must NOT re-filter, re-remap or re-split.

The filter + remap + split logic is byte-compatible with the current SoFT pipeline
(`SoFT_source/data_utils.py::reset_and_filter_data` + `shuffle_and_split_dataset`),
so results already obtained on the SoFT splits remain valid.

Usage:
    python build_common.py --dataset ciao_timestamp --seeds 42,43,44
    python build_common.py --dataset epinions      --seeds 42

Outputs (under processed/<dataset>/):
    interactions.csv        user_id,item_id,category_id,rating,timestamp   (1..N ids)
    social.csv              user_id_1,user_id_2                            (directed, full edge set)
    user_map.json           {original_id: reindexed_id}   (reindexed in 1..N)
    item_map.json           {original_id: reindexed_id}
    stats.json              counts + per-seed split sizes + a content hash
    splits/rating_train_seed{S}.csv   user_id,item_id,category_id,rating,timestamp
    splits/rating_valid_seed{S}.csv
    splits/rating_test_seed{S}.csv
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")
PROCESSED_DIR = os.path.join(HERE, "processed")

# columns we try to carry from the raw rating file (whatever is present)
_RATING_COLS = ["user_id", "product_id", "category_id", "rating", "timestamp"]


def _find_raw(dataset, name):
    """raw/<dataset>/<name> if present, else fall back to SoFT_source/dataset/<dataset>/<name>."""
    p = os.path.join(RAW_DIR, dataset, name)
    if os.path.isfile(p):
        return p
    fallback = os.path.join(HERE, "..", "..", "SoFT_source", "dataset", dataset, name)
    fallback = os.path.abspath(fallback)
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        f"raw file '{name}' for dataset '{dataset}' not found "
        f"(looked in {p} and {fallback})"
    )


def load_raw(dataset):
    rating = pd.read_csv(_find_raw(dataset, "rating_org.csv"))
    trust = pd.read_csv(_find_raw(dataset, "trustnetwork_org.csv"))

    keep = [c for c in _RATING_COLS if c in rating.columns]
    if "user_id" not in keep or "product_id" not in keep or "rating" not in keep:
        raise ValueError(f"rating_org.csv for {dataset} missing required columns; has {list(rating.columns)}")
    rating = rating[keep].copy()
    trust = trust[["user_id_1", "user_id_2"]].copy()
    return rating, trust


def filter_and_remap(rating, trust):
    """Exact port of SoFT_source/data_utils.py::reset_and_filter_data (+ the dedup/range
    filters that mat_to_csv applies just before it). NO k-core."""
    # --- basic filters (mat_to_csv) ---
    rating = rating.dropna(subset=["user_id", "product_id", "rating"])
    rating = rating.drop_duplicates(["user_id", "product_id"], keep="first")
    rating = rating[rating.rating.between(1, 5, inclusive="both")]

    trust = trust.dropna(how="any")
    trust = trust.drop_duplicates(keep="first")

    # --- user universe: (u1 U u2) INTERSECT rating-users, applied twice to converge ---
    for _ in range(2):
        social_users = set(trust.user_id_1.unique()).union(set(trust.user_id_2.unique()))
        total_users = social_users.intersection(set(rating.user_id.unique()))
        rating = rating[rating.user_id.isin(total_users)]
        trust = trust[trust.user_id_1.isin(total_users) & trust.user_id_2.isin(total_users)]

    # --- id maps: users sorted by original id -> 1..N ; items in first-appearance order -> 1..M ---
    all_users = sorted(
        set(trust.user_id_1.unique())
        .union(set(trust.user_id_2.unique()))
        .union(set(rating.user_id.unique()))
    )
    user_map = {int(u): i + 1 for i, u in enumerate(all_users)}
    item_map = {int(it): i + 1 for i, it in enumerate(rating["product_id"].unique())}

    rating = rating.copy()
    rating["user_id"] = rating["user_id"].map(user_map).astype(np.int64)
    rating["product_id"] = rating["product_id"].map(item_map).astype(np.int64)
    trust = trust.copy()
    trust["user_id_1"] = trust["user_id_1"].map(user_map).astype(np.int64)
    trust["user_id_2"] = trust["user_id_2"].map(user_map).astype(np.int64)

    rating = rating.rename(columns={"product_id": "item_id"})
    # normalise optional columns so downstream schema is stable
    if "category_id" not in rating.columns:
        rating["category_id"] = -1
    if "timestamp" not in rating.columns:
        rating["timestamp"] = -1
    rating = rating[["user_id", "item_id", "category_id", "rating", "timestamp"]]
    trust = trust.rename(columns={"user_id_1": "user_id_1", "user_id_2": "user_id_2"})
    return rating, trust, user_map, item_map


def split_interactions(interactions, seed, held_ratio=0.2):
    """Port of SoFT_source/data_utils.py::shuffle_and_split_dataset.
    held_ratio is the valid+test total; it is halved -> 8:1:1 at the default 0.2."""
    df = interactions.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    df = shuffle(df, random_state=seed)
    num_hold = int(len(df) * held_ratio)
    num_test = num_hold // 2
    test = df.iloc[:num_test]
    valid = df.iloc[num_test:num_hold]
    train = df.iloc[num_hold:]
    assert len(train) + len(valid) + len(test) == len(df)
    return train, valid, test


def _hash_df(df):
    return hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def build(dataset, seeds, held_ratio=0.2):
    out_dir = os.path.join(PROCESSED_DIR, dataset)
    split_dir = os.path.join(out_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    rating_raw, trust_raw = load_raw(dataset)
    print(f"[{dataset}] raw: {len(rating_raw)} ratings / {len(trust_raw)} trust edges")

    interactions, social, user_map, item_map = filter_and_remap(rating_raw, trust_raw)
    n_users = max(social.user_id_1.max(), social.user_id_2.max(), interactions.user_id.max())
    n_items = int(interactions.item_id.max())
    print(f"[{dataset}] filtered: {len(interactions)} ratings / {len(social)} edges "
          f"/ {n_users} users / {n_items} items")

    interactions.to_csv(os.path.join(out_dir, "interactions.csv"), index=False)
    social.to_csv(os.path.join(out_dir, "social.csv"), index=False)
    with open(os.path.join(out_dir, "user_map.json"), "w") as f:
        json.dump({str(k): v for k, v in user_map.items()}, f)
    with open(os.path.join(out_dir, "item_map.json"), "w") as f:
        json.dump({str(k): v for k, v in item_map.items()}, f)

    stats = {
        "dataset": dataset,
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(len(interactions)),
        "n_social_edges": int(len(social)),
        "held_ratio": held_ratio,
        "interactions_hash": _hash_df(interactions),
        "social_hash": _hash_df(social),
        "splits": {},
    }

    for seed in seeds:
        train, valid, test = split_interactions(interactions, seed, held_ratio)
        for name, part in [("train", train), ("valid", valid), ("test", test)]:
            part.to_csv(os.path.join(split_dir, f"rating_{name}_seed{seed}.csv"), index=False)
        stats["splits"][str(seed)] = {
            "train": len(train), "valid": len(valid), "test": len(test),
            "train_hash": _hash_df(train), "valid_hash": _hash_df(valid), "test_hash": _hash_df(test),
        }
        print(f"[{dataset}] seed {seed}: train {len(train)} / valid {len(valid)} / test {len(test)}")

    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[{dataset}] wrote -> {out_dir}")
    return stats


def get_args():
    p = argparse.ArgumentParser(description="Build the common benchmark preprocessing artifacts")
    p.add_argument("--dataset", required=True, help="ciao_timestamp / epinions / ciao / ...")
    p.add_argument("--seeds", default="42", help="comma-separated, e.g. 42,43,44")
    p.add_argument("--held_ratio", type=float, default=0.2,
                   help="valid+test total ratio (halved -> 8:1:1 at 0.2)")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    build(args.dataset, seeds, args.held_ratio)
