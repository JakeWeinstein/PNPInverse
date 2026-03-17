# Gradient Benchmark Report

## Accuracy: Relative Error vs Fine-FD Reference (h=1e-7)

| Model | Method | Mean Rel Error | Max Rel Error |
|-------|--------|----------------|---------------|
| nn_ensemble | autograd | 2.49e+00 | 7.05e+00 |
| nn_ensemble | fd_h=1e-03 | 2.49e+00 | 7.06e+00 |
| nn_ensemble | fd_h=1e-04 | 2.49e+00 | 7.06e+00 |
| nn_ensemble | fd_h=1e-05 | 2.49e+00 | 7.21e+00 |
| nn_ensemble | fd_h=1e-06 | 2.18e+00 | 6.06e+00 |
| gp | autograd_gp | 1.37e+12 | 2.81e+13 |
| gp | fd_h=1e-03 | 1.73e+12 | 3.06e+13 |
| gp | fd_h=1e-04 | 1.53e+12 | 2.02e+13 |
| gp | fd_h=1e-05 | 3.09e+13 | 4.70e+14 |
| gp | fd_h=1e-06 | 2.75e+14 | 4.78e+15 |
| pce | analytic | 9.17e-09 | 5.32e-08 |
| pce | fd_h=1e-03 | 1.26e-05 | 1.01e-04 |
| pce | fd_h=1e-04 | 1.22e-07 | 1.02e-06 |
| pce | fd_h=1e-05 | 8.61e-09 | 5.22e-08 |
| pce | fd_h=1e-06 | 9.27e-09 | 4.98e-08 |
| pod_rbf | fd_h=1e-03 | 1.51e-03 | 3.55e-02 |
| pod_rbf | fd_h=1e-04 | 3.91e-05 | 4.10e-04 |
| pod_rbf | fd_h=1e-05 | 2.51e-05 | 4.17e-04 |
| pod_rbf | fd_h=1e-06 | 2.12e-05 | 3.21e-04 |
| rbf_baseline | fd_h=1e-03 | 5.77e-04 | 9.44e-03 |
| rbf_baseline | fd_h=1e-04 | 1.10e-04 | 1.56e-03 |
| rbf_baseline | fd_h=1e-05 | 1.05e-04 | 1.53e-03 |
| rbf_baseline | fd_h=1e-06 | 1.02e-04 | 1.50e-03 |

## Speed: Wall-Clock Gradient Evaluation Time

| Model | Method | ms/eval | Batch Size |
|-------|--------|---------|------------|
| nn_ensemble | autograd | 3.87 | 1 |
| nn_ensemble | fd | 8.29 | 1 |
| gp | autograd_gp | 2843.40 | 1 |
| gp | fd | 440.75 | 1 |
| pce | analytic | 258.24 | 1 |
| pce | fd | 312.06 | 1 |
| pod_rbf | fd | 3.83 | 1 |
| rbf_baseline | fd | 0.66 | 1 |

## Ranking by Autograd/Analytic Speed (single-point)

1. **nn_ensemble** (autograd): 3.87 ms/eval
2. **pce** (analytic): 258.24 ms/eval
3. **gp** (autograd_gp): 2843.40 ms/eval

## Ranking by Autograd/Analytic Accuracy

1. **pce** (analytic): mean rel error = 9.17e-09
2. **nn_ensemble** (autograd): mean rel error = 2.49e+00
3. **gp** (autograd_gp): mean rel error = 1.37e+12
