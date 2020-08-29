#!/bin/bash
# CMNIST Experiment.

# Hyperparameters
N_RESTARTS=10
HIDDEN_DIM=390
L2_REGULARIZER_WEIGHT=0.00110794568
LR=0.0004898536566546834
LABEL_NOISE=${1-0.05}
PENALTY_ANNEAL_ITERS=190
PENALTY_WEIGHT=191257.18613115903
STEPS=501
ROOT=${2-/scratch/gobi1/creager/opt_env/cmnist}
# RNUM=$(printf "%05d" $(($RANDOM$RANDOM$RANDOM % 100000)))
TAG=$(date +'%Y-%m-%d')--$LABEL_NOISE
ROOT=$ROOT/label_noise_sweep/$TAG

# ERM
python -u -m opt_env.irm_cmnist \
  --results_dir $ROOT/erm \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --label_noise $LABEL_NOISE \
  --penalty_anneal_iters 0 \
  --penalty_weight 0.0 \
  --steps $STEPS

# IRM
python -u -m opt_env.irm_cmnist \
  --results_dir $ROOT/irm \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --label_noise $LABEL_NOISE \
  --penalty_anneal_iters $PENALTY_ANNEAL_ITERS \
  --penalty_weight $PENALTY_WEIGHT \
  --steps $STEPS
  
# EIIL
python -u -m opt_env.irm_cmnist \
  --results_dir $ROOT/eiil \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --label_noise $LABEL_NOISE \
  --penalty_anneal_iters $PENALTY_ANNEAL_ITERS \
  --penalty_weight $PENALTY_WEIGHT \
  --steps $STEPS \
  --eiil
  
# EIIL with color-based reference classifier
python -u -m opt_env.irm_cmnist \
  --results_dir $ROOT/eiil_cb \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --label_noise $LABEL_NOISE \
  --penalty_anneal_iters $PENALTY_ANNEAL_ITERS \
  --penalty_weight $PENALTY_WEIGHT \
  --steps $STEPS \
  --eiil \
  --color_based
  
# Evaluate color-based classifier on its own
python -u -m opt_env.irm_cmnist \
  --results_dir $ROOT/cb \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --label_noise $LABEL_NOISE \
  --penalty_anneal_iters $PENALTY_ANNEAL_ITERS \
  --penalty_weight $PENALTY_WEIGHT \
  --steps $STEPS \
  --color_based_eval \

# Grayscale baseline
python -u -m opt_env.irm_cmnist \
  --results_dir $ROOT/gray \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --label_noise $LABEL_NOISE \
  --penalty_anneal_iters $PENALTY_ANNEAL_ITERS \
  --penalty_weight 0.0 \
  --steps $STEPS \
  --grayscale_model

# Build latex tables
# accuracy
python -u -m opt_env.cmnist_results.acc_table \
  --erm_results_dir $ROOT/erm \
  --irm_results_dir $ROOT/irm \
  --eiil_results_dir $ROOT/eiil \
  --eiil_cb_results_dir $ROOT/eiil_cb \
  --cb_results_dir $ROOT/cb \
  --gray_results_dir $ROOT/gray \
  --results_dir $ROOT/acc_table
  