# E2E Gradient Validation Report

## Setup

- provider_uri: `/home/chainsmoker/.qlib/qlib_data/cn_data`
- market: `all`
- label_expr: `Ref($close, -2) / Ref($close, -1) - 1`
- step_len: `80`
- batch_size: `64`
- seed: `0`
- full extractor factors: `157`
- subset factors: `19`

## Batch Summary

### full_e2e

- pass_rate: `1.0000`
- train: batches=`20`, pass_rate=`1.0000`, mean_loss=`4.107092e-01`, mean_grad_mean_abs=`1.859743e+04`
- test: batches=`10`, pass_rate=`1.0000`, mean_loss=`3.760647e-01`, mean_grad_mean_abs=`1.898736e+04`

### subset_e2e

- pass_rate: `1.0000`
- train: batches=`20`, pass_rate=`1.0000`, mean_loss=`2.572222e-01`, mean_grad_mean_abs=`1.898303e+04`
- test: batches=`10`, pass_rate=`1.0000`, mean_loss=`2.368064e-01`, mean_grad_mean_abs=`2.037327e+04`

## Suspicious Columns

### full_e2e

- no suspicious columns detected under the current threshold

### subset_e2e

- volume: low_ratio=`0.9667`, mean_relative_scaled_score=`3.049858e-02`

## Factor Probe Summary

- VSUMD5 (greater_relu_volume): pass_rate=`0.0000`, mean_grad_mean_abs=`6.309932e-12`, mean_expected_grad_mean_abs=`3.154966e-11`, suspicious_expected_columns=`none`
- VSUMN5 (greater_relu_volume): pass_rate=`0.0000`, mean_grad_mean_abs=`3.221421e-12`, mean_expected_grad_mean_abs=`1.610711e-11`, suspicious_expected_columns=`none`
- VSUMP5 (greater_relu_volume): pass_rate=`0.0000`, mean_grad_mean_abs=`2.417879e-12`, mean_expected_grad_mean_abs=`1.208939e-11`, suspicious_expected_columns=`none`
