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
- selected_available_count: 50887
- selected_missing_count: 1504
- attackable_count: 50887
- selected_ratio: 0.050081
- attackable_ratio: 0.048644

## 四组主指标

### reference_clean

- annualized_excess_return_with_cost: 0.3558367573201775
- max_drawdown_with_cost: -0.0693081838537013
- rank_ic_mean: 0.0753396735420719

### partial_clean

- annualized_excess_return_with_cost: 0.37494553912894335
- max_drawdown_with_cost: -0.06112518417933693
- rank_ic_mean: 0.07534356046054519

### partial_fgsm

- annualized_excess_return_with_cost: 0.35530934189803526
- max_drawdown_with_cost: -0.06878076637752084
- rank_ic_mean: 0.063976985990896

### partial_pgd

- annualized_excess_return_with_cost: 0.21931309892945852
- max_drawdown_with_cost: -0.0981698724803185
- rank_ic_mean: 0.06422904866777757

## 主比较差值

|                                     |   annualized_excess_return_with_cost |   annualized_excess_return_without_cost |      ic_mean |         icir |   information_ratio_with_cost |   information_ratio_without_cost |   max_drawdown_with_cost |   max_drawdown_without_cost |   rank_ic_mean |    rank_icir |
|:------------------------------------|-------------------------------------:|----------------------------------------:|-------------:|-------------:|------------------------------:|---------------------------------:|-------------------------:|----------------------------:|---------------:|-------------:|
| reference_clean                     |                            0.355837  |                               0.404298  |  0.0315578   |  0.269899    |                     2.65346   |                        3.01444   |              -0.0693082  |                 -0.0586682  |    0.0753397   |  0.581247    |
| partial_clean                       |                            0.374946  |                               0.423378  |  0.0315513   |  0.269714    |                     2.77153   |                        3.12911   |              -0.0611252  |                 -0.053004   |    0.0753436   |  0.581268    |
| partial_fgsm                        |                            0.355309  |                               0.403583  |  0.0154799   |  0.142924    |                     2.81495   |                        3.19704   |              -0.0687808  |                 -0.0614303  |    0.063977    |  0.505422    |
| partial_pgd                         |                            0.219313  |                               0.267571  |  0.0152427   |  0.140908    |                     1.73108   |                        2.11191   |              -0.0981699  |                 -0.0908121  |    0.064229    |  0.506531    |
| partial_clean_minus_reference_clean |                            0.0191088 |                               0.0190796 | -6.51746e-06 | -0.000184505 |                     0.118074  |                        0.114668  |               0.008183   |                  0.00566415 |    3.88692e-06 |  2.18811e-05 |
| partial_fgsm_minus_partial_clean    |                           -0.0196362 |                              -0.0197942 | -0.0160714   | -0.12679     |                     0.0434159 |                        0.0679339 |              -0.00765558 |                 -0.00842632 |   -0.0113666   | -0.0758464   |
| partial_pgd_minus_partial_clean     |                           -0.155632  |                              -0.155806  | -0.0163086   | -0.128806    |                    -1.04046   |                       -1.0172    |              -0.0370447  |                 -0.0378081  |   -0.0111145   | -0.0747371   |
