# 部分股票白盒攻击回测实验报告

## 攻击设置

- constraint_mode: physical_stat
- tau_ret: 0.005
- tau_body: 0.005
- tau_range: 0.01
- tau_vol: 0.05
- lambda_ret: 0.8
- lambda_candle: 0.4
- lambda_vol: 0.3
- selected_count: 52391
- selected_available_count: 50940
- selected_missing_count: 1451
- attackable_count: 50940
- selected_ratio: 0.050081
- attackable_ratio: 0.048694

## 四组主指标

### reference_clean

- annualized_excess_return_with_cost: 0.3558367573201772
- max_drawdown_with_cost: -0.06930818385370086
- rank_ic_mean: 0.0753396735420719

### partial_clean

- annualized_excess_return_with_cost: 0.35206136242731234
- max_drawdown_with_cost: -0.0721160117463493
- rank_ic_mean: 0.0753482507707909

### partial_fgsm

- annualized_excess_return_with_cost: 0.3691078055604902
- max_drawdown_with_cost: -0.08543400068583828
- rank_ic_mean: 0.06395905442405639

### partial_pgd

- annualized_excess_return_with_cost: 0.2753953029271193
- max_drawdown_with_cost: -0.07628703712278398
- rank_ic_mean: 0.06425050794576259

## 主比较差值

|                                     |   annualized_excess_return_with_cost |   annualized_excess_return_without_cost |      ic_mean |         icir |   information_ratio_with_cost |   information_ratio_without_cost |   max_drawdown_with_cost |   max_drawdown_without_cost |   rank_ic_mean |   rank_icir |
|:------------------------------------|-------------------------------------:|----------------------------------------:|-------------:|-------------:|------------------------------:|---------------------------------:|-------------------------:|----------------------------:|---------------:|------------:|
| reference_clean                     |                           0.355837   |                              0.404298   |  0.0315578   |  0.269899    |                     2.65346   |                         3.01444  |              -0.0693082  |                 -0.0586682  |    0.0753397   |   0.581247  |
| partial_clean                       |                           0.352061   |                              0.400526   |  0.0316375   |  0.270311    |                     2.59057   |                         2.94673  |              -0.072116   |                 -0.0634922  |    0.0753483   |   0.581179  |
| partial_fgsm                        |                           0.369108   |                              0.417596   |  0.0162528   |  0.148895    |                     2.76438   |                         3.12736  |              -0.085434   |                 -0.0780787  |    0.0639591   |   0.503819  |
| partial_pgd                         |                           0.275395   |                              0.323348   |  0.0162327   |  0.149105    |                     2.19145   |                         2.57302  |              -0.076287   |                 -0.068972   |    0.0642505   |   0.505483  |
| partial_clean_minus_reference_clean |                          -0.00377539 |                             -0.00377158 |  7.97246e-05 |  0.000411952 |                    -0.0628889 |                        -0.067711 |              -0.00280783 |                 -0.00482405 |    8.57723e-06 |  -6.726e-05 |
| partial_fgsm_minus_partial_clean    |                           0.0170464  |                              0.0170693  | -0.0153847   | -0.121415    |                     0.17381   |                         0.180635 |              -0.013318   |                 -0.0145865  |   -0.0113892   |  -0.0773604 |
| partial_pgd_minus_partial_clean     |                          -0.0766661  |                             -0.077179   | -0.0154049   | -0.121205    |                    -0.399123  |                        -0.373711 |              -0.00417103 |                 -0.00547979 |   -0.0110977   |  -0.0756962 |
