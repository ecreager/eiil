"""Model utils."""
import os
import pickle

import torch
from torch import nn


class ColorBasedClassifier(nn.Module):
  LOG_CONFIDENCE = .5
  def forward(self, input):
    # estimate color based on arg max over the two channels
    input = input.sum((-1, -2))  # add pixels to get per-channel sums
    if len(input) == 0:  # hack to handle size zero batches
      hard_prediction = torch.zeros(0, 1).to(input.device)
    else:
      hard_prediction = torch.argmax(input, -1, keepdim=True)  # in {0, 1}
      hard_prediction = hard_prediction.float()
    hard_prediction = hard_prediction * 2 - 1.  # in {-1, 1}
    return hard_prediction * self.LOG_CONFIDENCE


def load_mlp(results_dir=None, flags=None, basename='params.p'):
  if flags is None:
    assert results_dir is not None, "flags and results_dir cannot both be None."
    flags = pickle.load(open(os.path.join(results_dir, 'flags.p'), 'rb'))

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

  mlp = MLP()
  if torch.cuda.is_available():
    mlp = mlp.cuda()

  if results_dir is not None:
    mlp.load_state_dict(torch.load(os.path.join(results_dir, basename)))
    print('Model params loaded from %s.' % results_dir)
  else:
    print('Model built with randomly initialized parameters.')
  mlp.eval()

  return mlp


