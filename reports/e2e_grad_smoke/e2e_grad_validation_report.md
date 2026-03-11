# E2E Gradient Validation Report

## Setup

- provider_uri: `/home/chainsmoker/.qlib/qlib_data/cn_data`
- market: `all`
- label_expr: `Ref($close, -2) / Ref($close, -1) - 1`
- step_len: `80`
- batch_size: `8`
- seed: `0`
- full extractor factors: `157`
- subset factors: `19`

## Batch Summary

### full_e2e

- pass_rate: `1.0000`
- train: batches=`1`, pass_rate=`1.0000`, mean_loss=`3.980148e-01`, mean_grad_mean_abs=`8.209504e-04`
- test: batches=`1`, pass_rate=`1.0000`, mean_loss=`3.010627e-01`, mean_grad_mean_abs=`1.648649e-03`

### subset_e2e

- pass_rate: `1.0000`
- train: batches=`1`, pass_rate=`1.0000`, mean_loss=`2.325818e-01`, mean_grad_mean_abs=`3.233461e-04`
- test: batches=`1`, pass_rate=`1.0000`, mean_loss=`2.509540e-01`, mean_grad_mean_abs=`1.170004e-03`

## Suspicious Columns

### full_e2e

- no suspicious columns detected under the current threshold

### subset_e2e

- no suspicious columns detected under the current threshold

## Factor Probe Summary

- VSUMD5 (greater_relu_volume): pass_rate=`0.0000`, mean_grad_mean_abs=`1.202789e-10`, mean_expected_grad_mean_abs=`6.013943e-10`, suspicious_expected_columns=`none`
- VSUMN5 (greater_relu_volume): pass_rate=`0.0000`, mean_grad_mean_abs=`5.517517e-11`, mean_expected_grad_mean_abs=`2.758758e-10`, suspicious_expected_columns=`none`
- VSUMP5 (greater_relu_volume): pass_rate=`0.0000`, mean_grad_mean_abs=`4.816934e-11`, mean_expected_grad_mean_abs=`2.408467e-10`, suspicious_expected_columns=`none`
