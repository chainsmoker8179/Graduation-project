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
- selected_available_count: 50811
- selected_missing_count: 1580
- attackable_count: 50811
- selected_ratio: 0.050081
- attackable_ratio: 0.048571

## 四组主指标

### reference_clean

- annualized_excess_return_with_cost: 0.3558367573201775
- max_drawdown_with_cost: -0.06930818385370124
- rank_ic_mean: 0.0753396735420719

### partial_clean

- annualized_excess_return_with_cost: 0.3739082470694016
- max_drawdown_with_cost: -0.0554280683255573
- rank_ic_mean: 0.07535117622179344

### partial_fgsm

- annualized_excess_return_with_cost: 0.3193130640304159
- max_drawdown_with_cost: -0.09735667724121583
- rank_ic_mean: 0.06372508166665432

### partial_pgd

- annualized_excess_return_with_cost: 0.2250271813171564
- max_drawdown_with_cost: -0.12363185214188327
- rank_ic_mean: 0.06398626174132

## 主比较差值

|                                     |   annualized_excess_return_with_cost |   annualized_excess_return_without_cost |     ic_mean |         icir |   information_ratio_with_cost |   information_ratio_without_cost |   max_drawdown_with_cost |   max_drawdown_without_cost |   rank_ic_mean |    rank_icir |
|:------------------------------------|-------------------------------------:|----------------------------------------:|------------:|-------------:|------------------------------:|---------------------------------:|-------------------------:|----------------------------:|---------------:|-------------:|
| reference_clean                     |                            0.355837  |                               0.404298  |  0.0315578  |  0.269899    |                      2.65346  |                         3.01444  |               -0.0693082 |                  -0.0586682 |    0.0753397   |  0.581247    |
| partial_clean                       |                            0.373908  |                               0.422391  |  0.0315395  |  0.269563    |                      2.7917   |                         3.15348  |               -0.0554281 |                  -0.0507829 |    0.0753512   |  0.581187    |
| partial_fgsm                        |                            0.319313  |                               0.367609  |  0.0157807  |  0.145115    |                      2.36876  |                         2.72685  |               -0.0973567 |                  -0.0899359 |    0.0637251   |  0.500828    |
| partial_pgd                         |                            0.225027  |                               0.273105  |  0.0157956  |  0.145641    |                      1.70961  |                         2.07465  |               -0.123632  |                  -0.116363  |    0.0639863   |  0.502033    |
| partial_clean_minus_reference_clean |                            0.0180715 |                               0.0180932 | -1.8266e-05 | -0.000335904 |                      0.138236 |                         0.139043 |                0.0138801 |                   0.0078853 |    1.15027e-05 | -5.95375e-05 |
| partial_fgsm_minus_partial_clean    |                           -0.0545952 |                              -0.0547824 | -0.0157589  | -0.124448    |                     -0.422941 |                        -0.42663  |               -0.0419286 |                  -0.0391531 |   -0.0116261   | -0.0803593   |
| partial_pgd_minus_partial_clean     |                           -0.148881  |                              -0.149286  | -0.0157439  | -0.123921    |                     -1.08208  |                        -1.07883  |               -0.0682038 |                  -0.0655806 |   -0.0113649   | -0.0791535   |
