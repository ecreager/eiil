#!/bin/bash
echo RUNNING ALPHA SWEEP EXPER
RESULTS_DIR=/scratch/gobi1/creager/opt_env/run_sems_alpha_sweep/$RANDOM$RANDOM
echo $RESULTS_DIR
SETUP_HETERO=2
N_REPS=5
mkdir -p $RESULTS_DIR
echo results found here
echo $RESULTS_DIR
echo RUNNING ALL METHODS FOR $N_REPS RESTARTS IN HETEROSKEDASTIC SETTING
python -u main.py --verbose 1 --methods "EIIL,ERM,ICP,IRM" --setup_hetero $SETUP_HETERO --results_dir $RESULTS_DIR --n_reps $N_REPS
for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  echo RUNNING EIIL WITH $alpha SPURIOUS REFERENCE CLASSIFIER FOR $N_REPS RESTARTS IN HETEROSKEDASTIC SETTING
  python -u main.py --verbose 1 --methods "EIIL" --setup_hetero $SETUP_HETERO --results_dir $RESULTS_DIR --eiil_ref_alpha $alpha --n_reps $N_REPS
done
