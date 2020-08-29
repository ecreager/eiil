# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np


class ChainEquationModel(object):
    def __init__(self, dim, scramble=False, hetero=True, hidden=False):
        self.hetero = hetero
        self.hidden = hidden
        self.dim = dim // 2
        ones = True

        if ones:
            self.wxy = torch.eye(self.dim)
            self.wyz = torch.eye(self.dim)
        else:
            self.wxy = torch.randn(self.dim, self.dim) / dim
            self.wyz = torch.randn(self.dim, self.dim) / dim

        if scramble:
            self.scramble, _ = torch.qr(torch.randn(dim, dim))
        else:
            self.scramble = torch.eye(dim)

        if hidden:
            self.whx = torch.randn(self.dim, self.dim) / dim
            self.why = torch.randn(self.dim, self.dim) / dim
            self.whz = torch.randn(self.dim, self.dim) / dim
        else:
            self.whx = torch.eye(self.dim, self.dim)
            self.why = torch.zeros(self.dim, self.dim)
            self.whz = torch.zeros(self.dim, self.dim)

    def solution(self):
        w = torch.cat((self.wxy.sum(1), torch.zeros(self.dim))).view(-1, 1)
        return self.scramble.t() @ w

    def __call__(self, n, env):
        h = torch.randn(n, self.dim) * env

        if self.hetero == 2:
            x = torch.randn(n, self.dim) * 5.
            y = x @ self.wxy + torch.randn(n, self.dim) * env
            z = y @ self.wyz + torch.randn(n, self.dim)
        elif self.hetero == 1:
            x = h @ self.whx + torch.randn(n, self.dim) * env
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim) * env
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim)
        else:
            x = h @ self.whx + torch.randn(n, self.dim) * env
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim)
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim) * env

        variances = dict(
          h=h.var().item(),
          x=x.var().item(),
          y=y.var().item(),
          z=z.var().item(),
          e=(torch.randn(n, self.dim) * env).var().item()  # any env dependent noise we might add
        )
        from pprint import pprint
        print('in setting %d data in env %d have following variances' % (self.hetero, env))
        pprint(variances)
        return torch.cat((x, z), 1) @ self.scramble, y.sum(1, keepdim=True)
