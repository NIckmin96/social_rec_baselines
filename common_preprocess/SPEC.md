# Common preprocessing spec — social-recommendation benchmark

Single source of truth for everything that must be identical across baselines so
that RMSE / MAE numbers are comparable. Per-baseline scripts consume the artifacts
below and only add **model-specific structure** (sparse R, trust adj, random walks,
sub-graphs, …) on top. They must not re-filter, re-remap, or re-split.

`build_common.py` is byte-compatible with the current SoFT pipeline
(`SoFT_source/data_utils.py::reset_and_filter_data` + `shuffle_and_split_dataset`) —
verified: all 6 ciao_timestamp / epinions split files match SoFT's existing
`rating_{split}_seed_42.csv` row-for-row on `(user_id, item_id, rating)`. Results
already collected on the SoFT splits stay valid.

## Frozen decisions

| item | value | rationale |
|---|---|---|
| filters | `dropna` on (user,item,rating); `drop_duplicates(user,item, keep=first)`; `rating ∈ [1,5]` | common denominator of SoFT / SocialMF / TrustMF / TrustSVD |
| k-core | **none** | user directive |
| user universe | `(social.u1 ∪ social.u2) ∩ rating.user`, applied twice to converge | SoFT / TrustSVD_torch / KCGN already use this rule |
| social graph | **full directed edge set** among that user set; no symmetrization, no normalization | objective comparison; each baseline symmetrizes/normalizes in its own adapter |
| id remap | users sorted by original id → `1..N`; items in first-appearance order → `1..M`; saved to `user_map.json` / `item_map.json` | contiguous ids, recoverable originals for error analysis |
| split | interaction-level **random**, `sklearn.utils.shuffle(random_state=seed)`, then `test = [:h/2]`, `valid = [h/2:h]`, `train = [h:]` with `h = int(len*0.2)` → **8:1:1** | current SoFT split; written to files, never recomputed by a baseline |
| seeds | whatever is passed to `--seeds` (report mean ± std over them) | single-seed numbers are noisy |
| timestamp / category | kept as columns in `interactions.csv` and in every split file; the split itself ignores them | KCGN needs them as features |
| eval set | held-out **positive ratings only** — no negative padding in valid/test | explicit RMSE / MAE definition |
| metric | one shared `metrics.py` (RMSE, MAE); predictions dumped as `results/<baseline>/pred_<ds>_seed<S>.csv` with columns `user_id,item_id,y_true,y_pred` in canonical ids | one scorer, one leaderboard |

## Artifacts — `processed/<dataset>/`

| file | schema | notes |
|---|---|---|
| `interactions.csv` | `user_id,item_id,category_id,rating,timestamp` | 1..N ids; `category_id` / `timestamp` = `-1` when absent in the raw file |
| `social.csv` | `user_id_1,user_id_2` | directed, full edge set within the user universe |
| `user_map.json` / `item_map.json` | `{original_id: reindexed_id}` | reindexed ids are 1..N |
| `stats.json` | counts + per-seed split sizes + content hashes | baselines take `n_users` / `n_items` from here, not from `nunique()` |
| `splits/rating_{train,valid,test}_seed{S}.csv` | same schema as `interactions.csv` | the only split source |

## Per-baseline handling (after the common stage)

All 6 baselines' preprocess scripts are unified to `data_preprocess.py` and consume
the common artifacts via `from common_preprocess.common import load_common`; the old
per-baseline `data_process.py` / `preprocess.py` are deleted (no archive).

| baseline | preprocess script | model-specific part kept | leaderboard |
|---|---|---|---|
| **SoFT** | `SoFT_source/data_utils.py` (`mat_to_csv` + `shuffle_and_split_dataset`) | random walks, sequence tensorization, degree buckets, `context_rating = train` leakage guard. Regen verified **byte-identical** to prior splits | ✅ |
| **SocialMF** | `SocialMF/data_preprocess.py` | trust matrix row-normalization; `add_negs` train-only | ✅ |
| **TrustMF** | `TrustMF/data_preprocess.py`, run via `main_mf.py` (WMF) | `truster`/`trustee` adjacency from `social.csv` | ✅ |
| **TrustSVD** | `TrustSVD_torch/data_preprocess.py` (was `data_process.py`) | raw directed trust matrix. `add_negs(valid/test)` **removed**; filenames `_seed{S}`; `main.py::test()` NDCG block commented | ✅ |
| **KCGN** | `KCGN/data_preprocess.py` (was `data_process.py`) — `process_kcgn` reads `interactions.csv` (0-based shift) not the `.mat` | **documented exceptions:** `dataProcess.py` keeps its per-user *temporal* split + `≥10` k-core; keeps `trust + trust.T`, rating-class multi-graph, DGI sub-graphs. Run via `main_explicit.py` | ✅ * |
| **GDSRec** | `GDSRec/data_preprocess.py` (was `preprocess.py`) — split from common (was its own 60/20/20) | all per-user/item avg & history lists, trust-similarity lists unchanged | ✅ |

\* KCGN's RMSE / MAE are on a different (temporal, k-core-filtered) eval subset —
asterisk on the leaderboard.

**Not in the benchmark:** DGRec-pytorch, RecDiff (session / implicit-ranking, no
rating-regression head).

## Metrics — `common_preprocess/metrics.py` (shared)

Every baseline computes RMSE / MAE through `rmse()` / `mae()` here and writes its
test predictions with `dump_predictions("<baseline>", <dataset>, <seed>, u, i, y_true, y_pred)`
→ `common_preprocess/results/<baseline>/pred_<dataset>_seed<seed>.csv`
(cols `user_id,item_id,y_true,y_pred`, canonical 1..N ids). One scorer builds the board:

```bash
python common_preprocess/metrics.py --dataset ciao_timestamp   # scores every results/*/pred_*.csv
```

Wired into: SoFT `main.py::eval()`, SocialMF `main.py::test()`, TrustMF `main_mf.py::test()`,
TrustSVD_torch `main.py::test()`, KCGN `main_explicit.py::validModel(dump=...)`,
GDSRec `main.py`/`test.py::validate(dump=...)` (pass `--dataset <ds> --seed <s>`).
Valid-time early-stop keeps each baseline's own inline calc (identical formula).

## Usage

```bash
python common_preprocess/build_common.py --dataset ciao_timestamp --seeds 42
python common_preprocess/build_common.py --dataset epinions      --seeds 42
python common_preprocess/validate_common.py --dataset ciao_timestamp --seed 42   # checks vs SoFT splits
```
