import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import pdb
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=5001)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--eiil', action='store_true')
flags = parser.parse_args()
torch.cuda.set_device(0)

print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

def split_data_opt(envs, model, n_steps=10000, n_samples=-1):
  """Learn soft environment assignment."""
  images = torch.cat((envs[0]['images'][:n_samples],envs[1]['images'][:n_samples]),0)
  labels = torch.cat((envs[0]['labels'][:n_samples],envs[1]['labels'][:n_samples]),0)
  print('size of pooled envs: '+str(len(images)))

  scale = torch.tensor(1.).cuda().requires_grad_()
  logits = model(images)
  loss = nll(logits * scale, labels, reduction='none')

  env_w = torch.randn(len(logits)).cuda().requires_grad_()
  optimizer = optim.Adam([env_w], lr=0.001)

  print('learning soft environment assignments')
  for i in tqdm(range(n_steps)):
    # penalty for env a
    lossa = (loss.squeeze() * env_w.sigmoid()).mean()
    grada = autograd.grad(lossa, [scale], create_graph=True)[0]
    penaltya = torch.sum(grada**2)
    # penalty for env b
    lossb = (loss.squeeze() * (1-env_w.sigmoid())).mean()
    gradb = autograd.grad(lossb, [scale], create_graph=True)[0]
    penaltyb = torch.sum(gradb**2)
    # negate
    npenalty = - torch.stack([penaltya, penaltyb]).mean()

    optimizer.zero_grad()
    npenalty.backward(retain_graph=True)
    optimizer.step()

  # split envs based on env_w threshold
  new_envs = []
  idx0 = (env_w.sigmoid()>.5)
  idx1 = (env_w.sigmoid()<=.5)
  # train envs
  for idx in (idx0, idx1):
    new_envs.append(dict(images=images[idx], labels=labels[idx]))
  # test env
  new_envs.append(dict(images=envs[-1]['images'],
                       labels=envs[-1]['labels']))
  print('size of env0: '+str(len(new_envs[0]['images'])))
  print('size of env1: '+str(len(new_envs[1]['images'])))
  print('size of env2: '+str(len(new_envs[2]['images'])))
  return new_envs

final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
  print("Restart", restart)

  # Load MNIST, make train/val splits, and shuffle train set examples

  mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
  mnist_train = (mnist.data[:50000], mnist.targets[:50000])
  mnist_val = (mnist.data[50000:], mnist.targets[50000:])

  rng_state = np.random.get_state()
  np.random.shuffle(mnist_train[0].numpy())
  np.random.set_state(rng_state)
  np.random.shuffle(mnist_train[1].numpy())

  # Build environments

  def make_environment(images, labels, e):
    def torch_bernoulli(p, size):
      return (torch.rand(size) < p).float()
    def torch_xor(a, b):
      return (a-b).abs() # Assumes both inputs are either 0 or 1
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
      'images': (images.float() / 255.).cuda(),
      'labels': labels[:, None].cuda()
    }
  envs = [
    make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
    make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
    make_environment(mnist_val[0], mnist_val[1], 0.9)
  ]

  # Define and instantiate the model

  class MLP(nn.Module):
    def __init__(self):
      super(MLP, self).__init__()
      if flags.grayscale_model:
        lin1 = nn.Linear(14 * 14, flags.hidden_dim)
      else:
        lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
      lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
      lin3 = nn.Linear(flags.hidden_dim, 1)
      for lin in [lin1, lin2, lin3]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
      self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
    def forward(self, input):
      if flags.grayscale_model:
        out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
      else:
        out = input.view(input.shape[0], 2 * 14 * 14)
      out = self._main(out)
      return out

  mlp_pre = MLP().cuda()
  mlp = MLP().cuda()

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

  optimizer_pre = optim.Adam(mlp_pre.parameters(), lr=flags.lr)
  optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

  pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

  if flags.eiil:
    # Pre-train reference model
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
      # NOTE: IRM penalties not used in pre-training
      #penalty_weight = (flags.penalty_weight
      #  if step >= flags.penalty_anneal_iters else 1.0)
      #loss += penalty_weight * train_penalty
      #if penalty_weight > 1.0:
      #  # Rescale the entire loss to keep gradients in a reasonable range
      #  loss /= penalty_weight

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
        
    envs = split_data_opt(envs, mlp_pre)

  for step in range(flags.steps):
    for env in envs:
      logits = mlp(env['images'])
      env['nll'] = nll(logits, env['labels'])
      env['acc'] = mean_accuracy(logits, env['labels'])
      env['penalty'] = penalty(logits, env['labels'])

    train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
    train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
    train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

    weight_norm = torch.tensor(0.).cuda()
    for w in mlp.parameters():
      weight_norm += w.norm().pow(2)

    loss = train_nll.clone()
    loss += flags.l2_regularizer_weight * weight_norm
    penalty_weight = (flags.penalty_weight
        if step >= flags.penalty_anneal_iters else 1.0)
    loss += penalty_weight * train_penalty
    if penalty_weight > 1.0:
      # Rescale the entire loss to keep gradients in a reasonable range
      loss /= penalty_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    test_acc = envs[2]['acc']
    if step % 100 == 0:
      pretty_print(
        np.int32(step),
        train_nll.detach().cpu().numpy(),
        train_acc.detach().cpu().numpy(),
        train_penalty.detach().cpu().numpy(),
        test_acc.detach().cpu().numpy()
      )

  final_train_accs.append(train_acc.detach().cpu().numpy())
  final_test_accs.append(test_acc.detach().cpu().numpy())
  print('Final train acc (mean/std across restarts so far):')
  print(np.mean(final_train_accs), np.std(final_train_accs))
  print('Final test acc (mean/std across restarts so far):')
  print(np.mean(final_test_accs), np.std(final_test_accs))
