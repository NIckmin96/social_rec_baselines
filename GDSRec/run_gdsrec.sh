#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
for DS in ciao_timestamp epinions; do
  echo "=========== GDSRec $DS  $(date) ==========="
  python main.py --dataset_path datasets/$DS/ --data $DS --dataset $DS \
      --seed 42 --sigma 0 --epoch 100 --batch_size 128 --embed_dim 256 --lr 1e-4 \
      > logs/gdsrec_${DS}.log 2>&1
  echo "--- $DS done (exit $?) ---"
  grep -aE "Test: MAE|\[metrics\]" logs/gdsrec_${DS}.log | tail -4
done
echo "=========== GDSRec ciao+epinions DONE $(date) ==========="
