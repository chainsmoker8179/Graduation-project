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
- selected_available_count: 50862
- selected_missing_count: 1529
- attackable_count: 50862
- selected_ratio: 0.050081
- attackable_ratio: 0.048620

## 四组主指标

### reference_clean

- annualized_excess_return_with_cost: 0.35583675732017767
- max_drawdown_with_cost: -0.06930818385370136
- rank_ic_mean: 0.0753396735420719

### partial_clean

- annualized_excess_return_with_cost: 0.359028283238322
- max_drawdown_with_cost: -0.06808604020614661
- rank_ic_mean: 0.07531454071520083

### partial_fgsm

- annualized_excess_return_with_cost: 0.4190270356697687
- max_drawdown_with_cost: -0.0746192447862673
- rank_ic_mean: 0.06388896524180387

### partial_pgd

- annualized_excess_return_with_cost: 0.31311437851825996
- max_drawdown_with_cost: -0.09264255147638467
- rank_ic_mean: 0.064102070851694

## 主比较差值

|                                     |   annualized_excess_return_with_cost |   annualized_excess_return_without_cost |      ic_mean |         icir |   information_ratio_with_cost |   information_ratio_without_cost |   max_drawdown_with_cost |   max_drawdown_without_cost |   rank_ic_mean |    rank_icir |
|:------------------------------------|-------------------------------------:|----------------------------------------:|-------------:|-------------:|------------------------------:|---------------------------------:|-------------------------:|----------------------------:|---------------:|-------------:|
| reference_clean                     |                           0.355837   |                              0.404298   |  0.0315578   |  0.269899    |                     2.65346   |                        3.01444   |              -0.0693082  |                 -0.0586682  |    0.0753397   |  0.581247    |
| partial_clean                       |                           0.359028   |                              0.407427   |  0.0315806   |  0.269882    |                     2.68584   |                        3.04754   |              -0.068086   |                 -0.0605719  |    0.0753145   |  0.580663    |
| partial_fgsm                        |                           0.419027   |                              0.467221   |  0.0162196   |  0.148839    |                     3.25216   |                        3.62598   |              -0.0746192  |                 -0.0717827  |    0.063889    |  0.502551    |
| partial_pgd                         |                           0.313114   |                              0.361319   |  0.0159602   |  0.146678    |                     2.47002   |                        2.85043   |              -0.0926426  |                 -0.0853324  |    0.0641021   |  0.503436    |
| partial_clean_minus_reference_clean |                           0.00319153 |                              0.00312925 |  2.27884e-05 | -1.67026e-05 |                     0.0323758 |                        0.0330985 |               0.00122214 |                 -0.00190377 |   -2.51328e-05 | -0.000583499 |
| partial_fgsm_minus_partial_clean    |                           0.0599988  |                              0.0597938  | -0.015361    | -0.121043    |                     0.566323  |                        0.578447  |              -0.0065332  |                 -0.0112107  |   -0.0114256   | -0.0781115   |
| partial_pgd_minus_partial_clean     |                          -0.0459139  |                             -0.0461083  | -0.0156204   | -0.123204    |                    -0.215817  |                       -0.19711   |              -0.0245565  |                 -0.0247604  |   -0.0112125   | -0.0772273   |
