# Environment Inference for Invariant Learning
This code accompanies the paper [Environment Inference for Invariant Learning](https://arxiv.org/abs/2010.07249), which appears at ICML 2021.

Thanks to my wonderful co-authors [Jörn-Henrik Jacobsen](https://github.com/jhjacobsen/) and [Richard Zemel](https://www.cs.toronto.edu/~zemel/inquiry/home.php).

The InvariantRiskMinimization subdirectory is modified from https://github.com/facebookresearch/InvariantRiskMinimization, and has its own license.

## Reproducing paper results

### Sythetic data
To produce results
```
cd InvariantRiskMinimization/code/experiment_synthetic/
./run_sems.sh
```
To analyze results
```
noteooks/sem_results.ipynb
```

### Color MNIST
To produce results
```
./exps/cmnist_label_noise_sweep.sh
```
To analyze results
```
notebooks/plot_cmnist_label_noise_sweep.ipynb
```
As an alternative, `InvariantRiskMinimization/code/colored_mnist/optimize_envs.sh` also runs EIIL+IRM on CMNIST with 25% label noise (the default from the IRM paper).

## Citing this work
If you find this code to your research useful please consider citing our workshop paper using the following bibtex entry
```
@inproceedings{creager21environment,
  title={Environment Inference for Invariant Learning},
  author={Creager, Elliot and Jacobsen, J{\"o}rn-Henrik and Zemel, Richard},
  booktitle={International Conference on Machine Learning},
  year={2021},
}

```
