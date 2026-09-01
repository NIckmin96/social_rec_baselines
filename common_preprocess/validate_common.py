"""
Sanity checks for the common preprocessing artifacts.

  1. split files are pairwise disjoint on (user_id, item_id)
  2. train u valid u test == interactions  (no rows lost / duplicated)
  3. no rating == 0 rows anywhere (no negative padding leaked in)
  4. every id is in 1..N / 1..M
  5. (optional) split matches SoFT_source/dataset/<ds>/rating_{split}_seed_{seed}.csv
     row-for-row on (user_id, item_id, rating)

Usage:
    python validate_common.py --dataset ciao_timestamp --seed 42
"""
import argparse
import os

import pandas as pd

from common import load_common, processed_dir

HERE = os.path.dirname(os.path.abspath(__file__))


def _key(df):
    return set(map(tuple, df[["user_id", "item_id"]].to_numpy().tolist()))


def validate(dataset, seed):
    inter, social, splits, stats = load_common(dataset, seed)
    tr, va, te = splits["train"], splits["valid"], splits["test"]
    ok = True

    ktr, kva, kte = _key(tr), _key(va), _key(te)
    for a, b, na, nb in [(ktr, kva, "train", "valid"), (ktr, kte, "train", "test"), (kva, kte, "valid", "test")]:
        inter_n = len(a & b)
        print(f"  {na}∩{nb} overlap: {inter_n}")
        ok &= inter_n == 0

    total = len(tr) + len(va) + len(te)
    print(f"  train+valid+test = {total} vs interactions = {len(inter)}")
    ok &= total == len(inter)
    ok &= (ktr | kva | kte) == _key(inter)

    for name, df in [("interactions", inter), ("train", tr), ("valid", va), ("test", te)]:
        n_zero = int((df["rating"] == 0).sum())
        if n_zero:
            print(f"  !! {name} has {n_zero} rating==0 rows")
        ok &= n_zero == 0

    umax = max(social.user_id_1.max(), social.user_id_2.max(), inter.user_id.max())
    print(f"  id ranges: user 1..{umax} (stats {stats['n_users']}), item 1..{inter.item_id.max()} (stats {stats['n_items']})")
    ok &= inter.user_id.min() >= 1 and inter.item_id.min() >= 1
    ok &= umax == stats["n_users"] and int(inter.item_id.max()) == stats["n_items"]

    # optional cross-check against the SoFT splits
    soft_dir = os.path.abspath(os.path.join(HERE, "..", "..", "SoFT_source", "dataset", dataset))
    matched = None
    if os.path.isdir(soft_dir):
        matched = True
        for split, df in [("train", tr), ("valid", va), ("test", te)]:
            p = os.path.join(soft_dir, f"rating_{split}_seed_{seed}.csv")
            if not os.path.isfile(p):
                matched = None
                break
            s = pd.read_csv(p)[["user_id", "product_id", "rating"]].reset_index(drop=True)
            c = df[["user_id", "item_id", "rating"]].rename(columns={"item_id": "product_id"}).reset_index(drop=True)
            row_exact = s.equals(c)
            print(f"  vs SoFT {split}: row-exact={row_exact}")
            matched &= row_exact
        if matched is not None:
            ok &= matched

    print(f"\n  {dataset} seed {seed}: {'OK' if ok else 'FAILED'}"
          + ("" if matched is None else f" | SoFT-split match: {matched}"))
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    raise SystemExit(0 if validate(args.dataset, args.seed) else 1)
