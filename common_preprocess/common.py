"""
Loader for the common benchmark preprocessing artifacts.

Every baseline's preprocess script should import from here instead of reading
rating_org.csv / trustnetwork_org.csv and doing its own filter/remap/split.

    from common_preprocess.common import load_common
    inter, social, splits, stats = load_common("ciao_timestamp", seed=42)

    # inter   : DataFrame[user_id, item_id, category_id, rating, timestamp]  (1..N ids)
    # social  : DataFrame[user_id_1, user_id_2]  (directed, full edge set)
    # splits  : {"train": df, "valid": df, "test": df}  (same schema as inter)
    # stats   : dict from stats.json (n_users, n_items, hashes, split sizes, ...)

Build the artifacts first with:
    python common_preprocess/build_common.py --dataset <ds> --seeds 42
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(HERE, "processed")


def processed_dir(dataset):
    return os.path.join(PROCESSED_DIR, dataset)


def _require(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found -- run "
            f"`python common_preprocess/build_common.py --dataset <ds> --seeds <...>` first"
        )
    return path


def load_stats(dataset):
    with open(_require(os.path.join(processed_dir(dataset), "stats.json"))) as f:
        return json.load(f)


def load_interactions(dataset):
    return pd.read_csv(_require(os.path.join(processed_dir(dataset), "interactions.csv")))


def load_social(dataset):
    """Directed, full edge set. Baselines that need an undirected/symmetric or
    row-normalised trust matrix must do that transform themselves."""
    return pd.read_csv(_require(os.path.join(processed_dir(dataset), "social.csv")))


def load_maps(dataset):
    d = processed_dir(dataset)
    with open(_require(os.path.join(d, "user_map.json"))) as f:
        user_map = {int(k): int(v) for k, v in json.load(f).items()}
    with open(_require(os.path.join(d, "item_map.json"))) as f:
        item_map = {int(k): int(v) for k, v in json.load(f).items()}
    return user_map, item_map


def load_splits(dataset, seed):
    d = os.path.join(processed_dir(dataset), "splits")
    out = {}
    for name in ("train", "valid", "test"):
        out[name] = pd.read_csv(_require(os.path.join(d, f"rating_{name}_seed{seed}.csv")))
    return out


def load_common(dataset, seed):
    return load_interactions(dataset), load_social(dataset), load_splits(dataset, seed), load_stats(dataset)
