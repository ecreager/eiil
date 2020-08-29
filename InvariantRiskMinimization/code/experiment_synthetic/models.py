# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import math

from sklearn.linear_model import LinearRegression
from itertools import chain, combinations
from scipy.stats import f as fdist
from scipy.stats import ttest_ind

from torch.autograd import grad

import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"



class InvariantRiskMinimization(object):
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            reg = 1. - reg  # change of variables for consistency with old codebase
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("IRM (reg={:.6f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        self.phi = best_phi

    def train(self, environments, args, reg=0):
        print('learning representation with', self, 'and reg', reg)
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss(x_e @ self.phi @ self.w, y_e)
                penalty += grad(error_e, self.w,
                                create_graph=True)[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
#             (reg * error + (1 - reg) * penalty).backward()  # dumb; zero reg means regularize 100%
            ((1 - reg) * error + reg * penalty).backward()  # good
            opt.step()

            if args["verbose"] and iteration % 1000 == 0:
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration,
                                                                      reg,
                                                                      error,
                                                                      penalty,
                                                                      w_str))

    def solution(self):
        return self.phi @ self.w

      
class LearnedEnvInvariantRiskMinimization(InvariantRiskMinimization):
    def __init__(self, environments, args, pretrain=False):
        best_reg = 0
        best_err = 1e6

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        if args['eiil_ref_alpha'] >= 0 and args['eiil_ref_alpha'] <= 1: 
            print('Using hard-coded reference classifier with alpha={:.2f}'.format(
              args['eiil_ref_alpha']
            ))
            alpha = args['eiil_ref_alpha']
            w_causal = (1. - alpha) * np.ones((1, 5))
            w_noncausal = alpha * np.ones((1, 5))  # spurious contribution to prediction
            w_ref = np.hstack((w_causal, w_noncausal))
            w_ref = torch.tensor(w_ref, dtype=torch.float32)
        else:
            print('Using ERM soln as reference classifier.')
            w_ref = EmpiricalRiskMinimizer(environments, args).solution()

        self.phi = torch.nn.Parameter(torch.diag(w_ref.squeeze()))
        dim_x = environments[0][0].size(1)
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True
        err = (x_val @ self.solution() - y_val).pow(2).mean().item()

        if args["verbose"]:
            print("EIIL's reference classifier has {:.3f} validation error.".format(
                err))
            print("EIIL's reference classifier has the following solution:\n.",
                  pretty(self.solution()))

        self.phi = self.phi.clone()

        environments = self.split(environments, args)
        if args["verbose"]:
            print("EIIL+ERM ref clf still has the following solution after AED (sanity check):\n.", pretty(self.solution()))
        best_reg = 0
        best_err = 1e6

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            reg = 1. - reg  # change of variables for consistency with old codebase
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("EIIL+IRM (reg={:.6f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        self.phi = best_phi


    def split(self, environments, args, n_samples=-1):
          """Learn soft environment assignment."""
          envs = environments
          test_env = envs[-1]
          x = torch.cat((envs[0][0][:n_samples],envs[1][0][:n_samples]),0)
          y = torch.cat((envs[0][1][:n_samples],envs[1][1][:n_samples]),0)
          print('size of pooled envs: '+str(len(x)))
     
          loss = torch.nn.MSELoss(reduction='none')
          error = loss(x @ self.phi @ self.w, y)

          env_w = torch.randn(len(error)).requires_grad_()
          optimizer = torch.optim.Adam([env_w], lr=0.001)

          print('learning soft environment assignments')
          with tqdm(total=args['n_iterations'],
                    position=1,
                    bar_format='{desc}',
                    desc='negative penalty: ',
                   ) as desc:
              for i in tqdm(range(args["n_iterations"])):
                # penalty for env a
                error_a = (error.squeeze() * env_w.sigmoid()).mean()
                penalty_a = grad(error_a, self.w, create_graph=True)[0].pow(2).mean()
                # penalty for env b
                error_b = (error.squeeze() * (1-env_w.sigmoid())).mean()
                penalty_b = grad(error_b, self.w, create_graph=True)[0].pow(2).mean()
                # negate
                npenalty = - torch.stack([penalty_a, penalty_b]).mean()
                desc.set_description('negative penalty: '+ str(npenalty))

                optimizer.zero_grad()
                npenalty.backward(retain_graph=True)
                optimizer.step()

          envs = []
          idx0 = (env_w.sigmoid()>.5)
          idx1 = (env_w.sigmoid()<=.5)
          envs.append((x[idx0],y[idx0]))
          print('size of env 0: '+str(len(x[idx0])))
          envs.append((x[idx1],y[idx1]))
          print('size of env 1: '+str(len(x[idx1])))
          print('weights: '+str(env_w.sigmoid()))
          envs.append(test_env)
          return envs

 
class InvariantCausalPrediction(object):
    def __init__(self, environments, args):
        self.coefficients = None
        self.alpha = args["alpha"]

        x_all = []
        y_all = []
        e_all = []

        for e, (x, y) in enumerate(environments):
            x_all.append(x.numpy())
            y_all.append(y.numpy())
            e_all.append(np.full(x.shape[0], e))

        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        e_all = np.hstack(e_all)

        dim = x_all.shape[1]

        accepted_subsets = []
        for subset in self.powerset(range(dim)):
            if len(subset) == 0:
                continue

            x_s = x_all[:, subset]
            reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

            p_values = []
            for e in range(len(environments)):
                e_in = np.where(e_all == e)[0]
                e_out = np.where(e_all != e)[0]

                res_in = (y_all[e_in] - reg.predict(x_s[e_in, :])).ravel()
                res_out = (y_all[e_out] - reg.predict(x_s[e_out, :])).ravel()

                p_values.append(self.mean_var_test(res_in, res_out))

            # TODO: Jonas uses "min(p_values) * len(environments) - 1"
            p_value = min(p_values) * len(environments)

            if p_value > self.alpha:
                accepted_subsets.append(set(subset))
                if args["verbose"]:
                    print("Accepted subset:", subset)

        if len(accepted_subsets):
            accepted_features = list(set.intersection(*accepted_subsets))
            if args["verbose"]:
                print("Intersection:", accepted_features)
            self.coefficients = np.zeros(dim)

            if len(accepted_features):
                x_s = x_all[:, list(accepted_features)]
                reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)
                self.coefficients[list(accepted_features)] = reg.coef_

            self.coefficients = torch.Tensor(self.coefficients)
        else:
            self.coefficients = torch.zeros(dim)

    def mean_var_test(self, x, y):
        pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                    x.shape[0] - 1,
                                    y.shape[0] - 1)

        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)

    def powerset(self, s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def solution(self):
        return self.coefficients


class EmpiricalRiskMinimizer(object):
    def __init__(self, environments, args):
        x_all = torch.cat([x for (x, y) in environments]).numpy()
        y_all = torch.cat([y for (x, y) in environments]).numpy()

        w = LinearRegression(fit_intercept=False).fit(x_all, y_all).coef_
        self.w = torch.Tensor(w)
        if args['verbose']:
          print('Done training ERM.')
          err = np.mean((x_all.dot(self.solution().T) - y_all) ** 2.).item()
          print("ERM has {:.3f} train error.".format(err))
          print("ERM has the following solution:\n ", pretty(self.solution()))

    def solution(self):
        return self.w
