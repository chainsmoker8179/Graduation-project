# 部分股票白盒攻击回测实验报告

## 攻击设置

- selected_count: 52422
- selected_available_count: 50874
- selected_missing_count: 1548
- attackable_count: 50874
- selected_ratio: 0.050092
- attackable_ratio: 0.048613

## 四组主指标

### reference_clean

- annualized_excess_return_with_cost: 0.37858373034478554
- max_drawdown_with_cost: -0.09231016208032033
- rank_ic_mean: 0.071964566111338

### partial_clean

- annualized_excess_return_with_cost: 0.38335518305113325
- max_drawdown_with_cost: -0.0932320665816948
- rank_ic_mean: 0.07200721786247624

### partial_fgsm

- annualized_excess_return_with_cost: 0.10333266619731009
- max_drawdown_with_cost: -0.14324567286098197
- rank_ic_mean: 0.05901900090695281

### partial_pgd

- annualized_excess_return_with_cost: -0.014962706800120785
- max_drawdown_with_cost: -0.1538094271052532
- rank_ic_mean: 0.057629478311671055

## 主比较差值

|                                     |   annualized_excess_return_with_cost |   annualized_excess_return_without_cost |      ic_mean |        icir |   information_ratio_with_cost |   information_ratio_without_cost |   max_drawdown_with_cost |   max_drawdown_without_cost |   rank_ic_mean |    rank_icir |
|:------------------------------------|-------------------------------------:|----------------------------------------:|-------------:|------------:|------------------------------:|---------------------------------:|-------------------------:|----------------------------:|---------------:|-------------:|
| reference_clean                     |                           0.378584   |                               0.426208  |  0.0332701   |  0.296025   |                     2.77726   |                        3.12615   |             -0.0923102   |                -0.084229    |    0.0719646   |  0.555608    |
| partial_clean                       |                           0.383355   |                               0.430812  |  0.0331755   |  0.294231   |                     2.86665   |                        3.22119   |             -0.0932321   |                -0.0849969   |    0.0720072   |  0.555024    |
| partial_fgsm                        |                           0.103333   |                               0.151388  |  0.0153688   |  0.150241   |                     0.89119   |                        1.30583   |             -0.143246    |                -0.135845    |    0.059019    |  0.463055    |
| partial_pgd                         |                          -0.0149627  |                               0.0328901 |  0.00755682  |  0.0831533  |                    -0.129149  |                        0.28392   |             -0.153809    |                -0.139441    |    0.0576295   |  0.45379     |
| partial_clean_minus_reference_clean |                           0.00477145 |                               0.0046039 | -9.45206e-05 | -0.00179416 |                     0.0893919 |                        0.0950422 |             -0.000921905 |                -0.000767863 |    4.26518e-05 | -0.000584528 |
| partial_fgsm_minus_partial_clean    |                          -0.280023   |                              -0.279425  | -0.0178067   | -0.14399    |                    -1.97546   |                       -1.91536   |             -0.0500136   |                -0.0508476   |   -0.0129882   | -0.0919685   |
| partial_pgd_minus_partial_clean     |                          -0.398318   |                              -0.397922  | -0.0256187   | -0.211078   |                    -2.9958    |                       -2.93727   |             -0.0605774   |                -0.0544445   |   -0.0143777   | -0.101234    |
