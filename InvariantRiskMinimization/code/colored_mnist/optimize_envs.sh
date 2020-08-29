#!/bin/bash

echo "EIIL + IRM"
python -u main_optenv.py \
  --hidden_dim=390 \
  --l2_regularizer_weight=0.00110794568 \
  --lr=0.0004898536566546834 \
  --penalty_anneal_iters=190 \
  --penalty_weight=191257.18613115903 \
  --steps=501 \
  --n_restarts=10 \
  --eiil \
