import argparse
import os
import pdb
import pickle
import sys

import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from tqdm import tqdm

from opt_env.utils.env_utils import get_envs
from opt_env.utils.opt_utils import split_data_opt
from opt_env.utils.opt_utils import train_irm_batch
from opt_env.utils.model_utils import load_mlp
from opt_env.utils.model_utils import ColorBasedClassifier

def main(flags):
  
  if not os.path.exists(flags.results_dir):
    os.makedirs(flags.results_dir)
  
  # save this file and command for reproducibility
  if flags.results_dir != '.':
    with open(__file__, 'r') as f:
      this_file = f.readlines()
      with open(os.path.join(flags.results_dir, 'irm_cmnist.py'), 'w') as f: 
        f.write(''.join(this_file))
    cmd = 'python ' + ' '.join(sys.argv)
    with open(os.path.join(flags.results_dir, 'command.sh'), 'w') as f:
      f.write(cmd)
  # save params for later
  if not os.path.exists(flags.results_dir):
    os.makedirs(flags.results_dir)
  pickle.dump(flags, open(os.path.join(flags.results_dir, 'flags.p'), 'wb'))
  for f in sys.stdout, open(os.path.join(flags.results_dir, 'flags.txt'), 'w'):
    print('Flags:', file=f)
    for k,v in sorted(vars(flags).items()):
      print("\t{}: {}".format(k, v), file=f)
      
  print('results will be found here:')
  print(flags.results_dir)

  final_train_accs = []
  final_test_accs = []
  for restart in range(flags.n_restarts):
    print("Restart", restart)

    rng_state = np.random.get_state()
    np.random.set_state(rng_state)

    # Build environments
    envs = get_envs(flags=flags)

    # Define and instantiate the model
    if flags.color_based:  # use color-based reference classifier without trainable params
      mlp_pre = ColorBasedClassifier()
    else:
      mlp_pre = load_mlp(results_dir=None, flags=flags).cuda()  # reference classifier
    mlp = load_mlp(results_dir=None, flags=flags).cuda()  # invariant representation learner
    mlp_pre.train()
    mlp.train()

    # Define loss function helpers

    def nll(logits, y, reduction='mean'):
      return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction=reduction)

    def mean_accuracy(logits, y):
      preds = (logits > 0.).float()
      return ((preds - y).abs() < 1e-2).float().mean()

    def penalty(logits, y):
      scale = torch.tensor(1.).cuda().requires_grad_()
      loss = nll(logits * scale, y)
      grad = autograd.grad(loss, [scale], create_graph=True)[0]
      return torch.sum(grad**2)

    # Train loop

    def pretty_print(*values):
      col_width = 13
      def format_val(v):
        if not isinstance(v, str):
          v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
      str_values = [format_val(v) for v in values]
      print("   ".join(str_values))


    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

    if flags.eiil:
      if flags.color_based:
        print('Color-based refernece classifier was specified, to skipping pre-training.') 
      else:
        optimizer_pre = optim.Adam(mlp_pre.parameters(), lr=flags.lr)
        for step in range(flags.steps):
          for env in envs:
            logits = mlp_pre(env['images'])
            env['nll'] = nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])

          train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
          train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
          train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

          weight_norm = torch.tensor(0.).cuda()
          for w in mlp_pre.parameters():
            weight_norm += w.norm().pow(2)

          loss = train_nll.clone()
          loss += flags.l2_regularizer_weight * weight_norm

          optimizer_pre.zero_grad()
          loss.backward()
          optimizer_pre.step()

          test_acc = envs[2]['acc']
          if step % 100 == 0:
            pretty_print(
              np.int32(step),
              train_nll.detach().cpu().numpy(),
              train_acc.detach().cpu().numpy(),
              train_penalty.detach().cpu().numpy(),
              test_acc.detach().cpu().numpy()
            )
      torch.save(mlp_pre.state_dict(), 
                 os.path.join(flags.results_dir, 'mlp_pre.%d.p' % restart))
      envs = split_data_opt(envs, mlp_pre)
    mlp, final_train_acc, final_test_acc = train_irm_batch(mlp, envs, flags)
    final_train_accs.append(final_train_acc)
    final_test_accs.append(final_test_acc)
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    print('done with restart %d' % restart)
    torch.save(mlp.state_dict(), 
               os.path.join(flags.results_dir, 'mlp.%s.p' % restart))

  print('done with all restarts')
  final_train_accs = [t.item() for t in final_train_accs]
  final_test_accs = [t.item() for t in final_test_accs]
  metrics = {'Train accs': final_train_accs,
             'Test accs': final_test_accs}
  with open(os.path.join(flags.results_dir, 'metrics.p'), 'wb') as f:
    pickle.dump(metrics, f)

  print('results are here:')
  print(flags.results_dir)

  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='IRM Colored MNIST')
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--n_restarts', type=int, default=1)
  parser.add_argument('--penalty_anneal_iters', type=int, default=100)
  parser.add_argument('--penalty_weight', type=float, default=10000.0)
  parser.add_argument('--steps', type=int, default=5001)
  parser.add_argument('--grayscale_model', action='store_true')
  parser.add_argument('--eiil', action='store_true')
  parser.add_argument('--results_dir', type=str, default='/tmp/opt_env/irm_cmnist', 
                      help='Directory where results should be saved.')
  parser.add_argument('--label_noise', type=float, default=0.25)
  parser.add_argument('--train_env_1__color_noise', type=float, default=0.2)
  parser.add_argument('--train_env_2__color_noise', type=float, default=0.1)
  parser.add_argument('--test_env__color_noise', type=float, default=0.9)
  parser.add_argument('--color_based', action='store_true')  # use color-based reference classifier without trainable params
  parser.add_argument('--color_based_eval', action='store_true')  # skip IRM phase and evaluate color-based classifier
  flags = parser.parse_args()
  torch.cuda.set_device(0)
  main(flags)
