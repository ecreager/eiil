"""Build results table for CMNIST experiment."""
import argparse
import os
import pickle

import numpy as np
import pandas as pd


def load_results(dirname):
  return pickle.load(open(os.path.join(dirname, 'metrics.p'), 'rb'))

def main(flags):
  # load results from disk
  results = dict(
    erm=load_results(flags.erm_results_dir),
    irm=load_results(flags.irm_results_dir),
    eiil=load_results(flags.eiil_results_dir),
    eiil_cb=load_results(flags.eiil_cb_results_dir),
    cb=load_results(flags.cb_results_dir),
    gray=load_results(flags.gray_results_dir),
  )
  num_methods = len(results)
  def mean_plus_minus_std(x):
    """Format list as its mean plus minus one std dev."""
    return r"""%.1f $\pm$ %.1f""" % (100. * np.mean(x), 100. * np.std(x))
  results = pd.DataFrame.from_dict(results)
  results = results.T
  results_tex = results.to_latex(
    formatters=[mean_plus_minus_std, ] * 2,
    escape=False
  )
  print(results_tex)
  if not os.path.exists(flags.results_dir):
    os.makedirs(flags.results_dir)
  print(results_tex, file=open(os.path.join(flags.results_dir, 'results.tex'), 'w'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Build results table for CMNIST experiment.')
  parser.add_argument('--erm_results_dir', type=str, default='/scratch/gobi1/creager/opt_env/cmnist/erm')
  parser.add_argument('--irm_results_dir', type=str, default='/scratch/gobi1/creager/opt_env/cmnist/irm')
  parser.add_argument('--eiil_results_dir', type=str, default='/scratch/gobi1/creager/opt_env/cmnist/eiil')
  parser.add_argument('--eiil_cb_results_dir', type=str, default='/scratch/gobi1/creager/opt_env/cmnist/eiil_cb')
  parser.add_argument('--cb_results_dir', type=str, default='/scratch/gobi1/creager/opt_env/cmnist/cb')
  parser.add_argument('--gray_results_dir', type=str, default='/scratch/gobi1/creager/opt_env/cmnist/gray')
  parser.add_argument('--results_dir', type=str, default='/scratch/gobi1/creager/opt_env/cmnist_results/acc_table',
                     help='where tex tables should be saved')
  flags = parser.parse_args()
  main(flags)