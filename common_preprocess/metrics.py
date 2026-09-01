"""
Shared explicit-feedback metrics + prediction dump for the benchmark.

Every baseline computes RMSE / MAE through THESE functions and writes its test
predictions with `dump_predictions(...)`, so a single scorer (`score_file` /
`score_all`) produces the leaderboard -- no per-baseline metric drift.

    from common_preprocess.metrics import rmse, mae, dump_predictions

    # ... after the test loop, with 1-D arrays / lists / torch tensors ...
    r = rmse(y_true, y_pred); m = mae(y_true, y_pred)
    dump_predictions("socialmf", "ciao_timestamp", 42, user_id, item_id, y_true, y_pred)

Ranking metrics (NDCG/precision/recall) are intentionally NOT here -- this
benchmark is explicit-feedback only. Re-add them later if needed.
"""
import os
import csv

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


def _to_1d_float(x):
    """Accept python list / numpy array / torch tensor (any device) -> 1-D float64 np array."""
    if hasattr(x, "detach"):          # torch tensor
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x


def _aligned(y_true, y_pred):
    t = _to_1d_float(y_true)
    p = _to_1d_float(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"y_true {t.shape} and y_pred {p.shape} length mismatch")
    if t.size == 0:
        raise ValueError("empty prediction set")
    return t, p


def rmse(y_true, y_pred):
    t, p = _aligned(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(y_true, y_pred):
    t, p = _aligned(y_true, y_pred)
    return float(np.mean(np.abs(p - t)))


def evaluate(y_true, y_pred):
    t, p = _aligned(y_true, y_pred)
    return {"rmse": rmse(t, p), "mae": mae(t, p), "n": int(t.size)}


def pred_path(baseline, dataset, seed, results_root=None):
    root = results_root or RESULTS_DIR
    return os.path.join(root, baseline, f"pred_{dataset}_seed{seed}.csv")


def dump_predictions(baseline, dataset, seed, user_id, item_id, y_true, y_pred,
                     results_root=None):
    """Write test predictions in the one canonical format:
    columns  user_id,item_id,y_true,y_pred  (ids are the common 1..N ids).
    Returns (path, metrics_dict)."""
    u = _to_1d_float(user_id).astype(np.int64)
    it = _to_1d_float(item_id).astype(np.int64)
    t, p = _aligned(y_true, y_pred)
    if not (len(u) == len(it) == len(t)):
        raise ValueError("user_id / item_id / y_true length mismatch")

    path = pred_path(baseline, dataset, seed, results_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id", "y_true", "y_pred"])
        for a, b, c, d in zip(u, it, t, p):
            w.writerow([int(a), int(b), f"{c:.6f}", f"{d:.6f}"])
    m = evaluate(t, p)
    print(f"[metrics] {baseline}/{dataset} seed{seed}: "
          f"RMSE {m['rmse']:.4f} | MAE {m['mae']:.4f} | n={m['n']} -> {path}")
    return path, m


def score_file(path):
    import pandas as pd
    df = pd.read_csv(path)
    return evaluate(df["y_true"].values, df["y_pred"].values)


def score_all(dataset=None, results_root=None):
    """Scan results/<baseline>/pred_*.csv and return a list of rows for the leaderboard."""
    import pandas as pd
    root = results_root or RESULTS_DIR
    rows = []
    if not os.path.isdir(root):
        return rows
    for baseline in sorted(os.listdir(root)):
        bdir = os.path.join(root, baseline)
        if not os.path.isdir(bdir):
            continue
        for fn in sorted(os.listdir(bdir)):
            if not (fn.startswith("pred_") and fn.endswith(".csv")):
                continue
            body = fn[len("pred_"):-len(".csv")]
            ds, _, seed = body.rpartition("_seed")
            if dataset and ds != dataset:
                continue
            m = score_file(os.path.join(bdir, fn))
            rows.append({"baseline": baseline, "dataset": ds, "seed": int(seed), **m})
    return rows


if __name__ == "__main__":
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser(description="Score benchmark prediction dumps")
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    rows = score_all(args.dataset)
    if not rows:
        print("no prediction files under", RESULTS_DIR)
    else:
        df = pd.DataFrame(rows).sort_values(["dataset", "rmse"])
        print(df.to_string(index=False))
