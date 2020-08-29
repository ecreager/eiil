# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pickle
import sys

from sem import ChainEquationModel
from models import *

import argparse
import torch
import numpy


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"


def errors(w, w_hat):
    w = w.view(-1)
    w_hat = w_hat.view(-1)

    i_causal = (w != 0).nonzero().view(-1)
    i_noncausal = (w == 0).nonzero().view(-1)

    if len(i_causal):
        error_causal = (w[i_causal] - w_hat[i_causal]).pow(2).mean()
        error_causal = error_causal.item()
    else:
        error_causal = 0

    if len(i_noncausal):
        error_noncausal = (w[i_noncausal] - w_hat[i_noncausal]).pow(2).mean()
        error_noncausal = error_noncausal.item()
    else:
        error_noncausal = 0

    return error_causal, error_noncausal


def run_experiment(args):
    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)

    if args["setup_sem"] == "chain":
        setup_str = "chain_hidden={}_hetero={}_scramble={}".format(
            args["setup_hidden"],
            args["setup_hetero"],
            args["setup_scramble"])
    elif args["setup_sem"] == "icp":
        setup_str = "sem_icp"
    else:
        raise NotImplementedError

    args['results_dir'] = os.path.join(args['results_dir'], setup_str)
    if args['eiil_ref_alpha'] >= 0 and args['eiil_ref_alpha'] <= 1: 
        args['results_dir'] = '{results_dir}_alpha_{eiil_ref_alpha:.1f}'.format(**args)
 
    if not os.path.exists(args['results_dir']):
        os.makedirs(args['results_dir'])
    pickle.dump(args, open(os.path.join(args['results_dir'], 'flags.p'), 'wb'))
    for f in sys.stdout, open(os.path.join(args['results_dir'], 'flags.txt'), 'w'):
        print('Flags:', file=f)
        for k,v in sorted(args.items()):
            print("\t{}: {}".format(k, v), file=f)
    print('results will be found here:')
    print(args['results_dir'])
    
    all_methods = {
        "ERM": EmpiricalRiskMinimizer,
        "ICP": InvariantCausalPrediction,
        "IRM": InvariantRiskMinimization,
        "EIIL": LearnedEnvInvariantRiskMinimization
    }

    if args["methods"] == "all":
        methods = all_methods
    else:
        methods = {m: all_methods[m] for m in args["methods"].split(',')}

    all_sems = []
    all_solutions = []
    all_environments = []
    from collections import defaultdict
    all_err_causal = defaultdict(list)
    all_err_noncausal = defaultdict(list)

    for rep_i in range(args["n_reps"]):
        if args["setup_sem"] == "chain":
            sem = ChainEquationModel(args["dim"],
                                     hidden=args["setup_hidden"],
                                     scramble=args["setup_scramble"],
                                     hetero=args["setup_hetero"])
            environments = [sem(args["n_samples"], .2),
                            sem(args["n_samples"], 2.),
                            sem(args["n_samples"], 5.)]
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)

    for sem, environments in zip(all_sems, all_environments):
        soln = sem.solution()
        solutions = [
            "{} {:<5} {} {:.5f} {:.5f}".format(setup_str,
                                             "SEM",
                                             pretty(sem.solution()), 0, 0)
        ]
        

        for method_name, method_constructor in methods.items():
            method = method_constructor(environments, args)
            msolution = method.solution()
            err_causal, err_noncausal = errors(sem.solution(), msolution)
            all_err_causal[method_name].append(err_causal)
            all_err_noncausal[method_name].append(err_noncausal)

            solutions.append("{} {:<5} {} {:.5f} {:.5f}".format(setup_str,
                                                             method_name,
                                                             pretty(msolution),
                                                             err_causal,
                                                             err_noncausal))

        all_solutions += solutions

    # save results
    results = dict()
    results.update(setup_str=setup_str)
    results.update(all_sems=all_sems)
    results.update(all_solutions=all_solutions)
    results.update(all_environments=all_environments)
    results.update(all_environments=all_environments)
    results.update(all_err_causal=all_err_causal)
    results.update(all_err_noncausal=all_err_noncausal)
    with open(os.path.join(args['results_dir'], 'results.p'), 'wb') as f:
        pickle.dump(results, f)
    
    return all_solutions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invariant regression')
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_reps', type=int, default=1)
    parser.add_argument('--skip_reps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)  # Negative is random
    parser.add_argument('--print_vectors', type=int, default=1)
    parser.add_argument('--n_iterations', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--methods', type=str, default="EIIL,IRM,ERM")
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--setup_sem', type=str, default="chain")
    parser.add_argument('--setup_hidden', type=int, default=0)
    parser.add_argument('--setup_hetero', type=int, default=1)
    parser.add_argument('--setup_scramble', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default="/tmp/experiment_synthetic")
    parser.add_argument('--eiil_ref_alpha', type=float, default=-1,
                       help=('Value between zero and one to hard code the reference '
                             'classifier propensity to use the spurious feature. Set '
                             'to value outside zero one interval to disable.'))
    args = dict(vars(parser.parse_args()))

    all_solutions = run_experiment(args)
    print("\n".join(all_solutions))
    print("\n".join(all_solutions), file=open(
      os.path.join(args['results_dir'], 'all_solutions.txt'), 'w')
         )
