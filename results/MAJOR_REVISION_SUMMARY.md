# Major-revision analysis summary

All revised prospective models exclude `delta_days` from the predictor matrix.
Positive paired deltas indicate improvement after adding physical tests.
Uncertainty for paired deltas is a 2000-replicate athlete-cluster bootstrap percentile interval.

## Incremental regression value of physical tests

```text
                  analysis              target            model     without_physical_tests with_physical_tests  n_rows  n_athletes  bootstrap_replicates  delta_r2  delta_r2_ci_low  delta_r2_ci_high  delta_mae  delta_mae_ci_low  delta_mae_ci_high  delta_rmse  delta_rmse_ci_low  delta_rmse_ci_high
   future_next_observation         next_points            Ridge context_current_implements          g1_current    1034         289                  2000   -0.0002          -0.0079            0.0071    -0.3545           -1.1603             0.4994     -0.0210            -0.8067              0.7266
   future_next_observation         next_points     RandomForest context_current_implements          g1_current    1034         289                  2000   -0.0069          -0.0159            0.0018    -1.4888           -2.4026            -0.5682     -0.6957            -1.6157              0.1807
   future_next_observation         next_points GradientBoosting context_current_implements          g1_current    1034         289                  2000    0.0009          -0.0309            0.0345    -1.5634           -3.3551             0.1183      0.0864            -3.0358              3.2196
        future_horizon_max   future_max_points            Ridge context_current_implements          g1_current    1041         289                  2000    0.0101          -0.0061            0.0265     0.0548           -1.4890             1.6489      0.9689            -0.5774              2.5662
        future_horizon_max   future_max_points     RandomForest context_current_implements          g1_current    1041         289                  2000    0.0061          -0.0122            0.0253    -0.1375           -1.6085             1.4077      0.5891            -1.1709              2.3245
        future_horizon_max   future_max_points GradientBoosting context_current_implements          g1_current    1041         289                  2000    0.0397           0.0124            0.0681     2.0682            0.1085             4.0351      3.7903             1.1553              6.4489
baseline_continuous_change future_delta_points            Ridge context_current_implements          g1_current     289         289                  2000   -0.0048          -0.0367            0.0272    -0.4854           -1.8223             0.8628     -0.2299            -1.7852              1.2645
baseline_continuous_change future_delta_points     RandomForest context_current_implements          g1_current     289         289                  2000   -0.0093          -0.0565            0.0395    -1.0598           -3.2359             0.9098     -0.4445            -2.7642              1.7648
baseline_continuous_change future_delta_points GradientBoosting context_current_implements          g1_current     289         289                  2000    0.0582          -0.0327            0.1595     1.4580           -2.3356             5.1004      2.7139            -1.5197              7.2632
```

## Incremental classification value of physical tests

```text
     outcome              model without_physical_tests with_physical_tests  n_rows  n_positive  bootstrap_replicates  delta_roc_auc  delta_roc_auc_ci_low  delta_roc_auc_ci_high  delta_pr_auc  delta_pr_auc_ci_low  delta_pr_auc_ci_high  delta_brier  delta_brier_ci_low  delta_brier_ci_high
improve_ge25 LogisticRegression        context_current          g1_current     289         137                  2000        -0.0031               -0.0137                 0.0066        0.0040              -0.0126                0.0221      -0.0014             -0.0054               0.0023
improve_ge25       RandomForest        context_current          g1_current     289         137                  2000        -0.0029               -0.0293                 0.0238        0.0171              -0.0211                0.0548      -0.0010             -0.0105               0.0088
improve_ge25   GradientBoosting        context_current          g1_current     289         137                  2000        -0.0014               -0.0391                 0.0369       -0.0102              -0.0617                0.0460      -0.0017             -0.0224               0.0202
improve_ge50 LogisticRegression        context_current          g1_current     289         113                  2000         0.0009               -0.0136                 0.0149       -0.0025              -0.0234                0.0271       0.0003             -0.0053               0.0059
improve_ge50       RandomForest        context_current          g1_current     289         113                  2000        -0.0107               -0.0325                 0.0097       -0.0006              -0.0425                0.0370      -0.0013             -0.0093               0.0062
improve_ge50   GradientBoosting        context_current          g1_current     289         113                  2000         0.0236               -0.0142                 0.0602        0.0574               0.0034                0.1100       0.0161             -0.0047               0.0361
improve_ge75 LogisticRegression        context_current          g1_current     289          92                  2000         0.0084               -0.0115                 0.0274        0.0129              -0.0216                0.0553       0.0029             -0.0041               0.0098
improve_ge75       RandomForest        context_current          g1_current     289          92                  2000        -0.0063               -0.0298                 0.0174        0.0119              -0.0352                0.0619       0.0041             -0.0049               0.0133
improve_ge75   GradientBoosting        context_current          g1_current     289          92                  2000         0.0001               -0.0505                 0.0505        0.0142              -0.0435                0.0816       0.0047             -0.0183               0.0288
future_ge900 LogisticRegression        context_current          g1_current     289          79                  2000        -0.0058               -0.0152                 0.0027       -0.0015              -0.0205                0.0170      -0.0013             -0.0055               0.0028
future_ge900       RandomForest        context_current          g1_current     289          79                  2000         0.0063               -0.0123                 0.0243        0.0517              -0.0041                0.0999       0.0049             -0.0031               0.0125
future_ge900   GradientBoosting        context_current          g1_current     289          79                  2000         0.0031               -0.0207                 0.0279        0.0272              -0.0395                0.0859      -0.0022             -0.0165               0.0126
future_ge950 LogisticRegression        context_current          g1_current     289          41                  2000        -0.0007               -0.0238                 0.0247        0.0163              -0.0492                0.0942       0.0073             -0.0038               0.0186
future_ge950       RandomForest        context_current          g1_current     289          41                  2000         0.0028               -0.0334                 0.0368        0.0480              -0.0422                0.1405       0.0208              0.0095               0.0319
future_ge950   GradientBoosting        context_current          g1_current     289          41                  2000         0.0133               -0.0363                 0.0680       -0.0130              -0.0873                0.0526       0.0022             -0.0127               0.0162
```

## Strict chronological holdout

```text
          task           outcome              model feature_set     r2     mae     rmse  roc_auc  pr_auc  balanced_accuracy     f1  brier cutoff_date training_index_limit  n_train  n_test  train_athletes  test_athletes  overlapping_athletes  train_positive  test_positive
    regression future_max_points              Ridge  g1_current 0.3199 79.1249 100.9471      NaN     NaN                NaN    NaN    NaN  2012-10-27           2010-10-28      168      82             168             82                     0             NaN            NaN
    regression future_max_points       RandomForest  g1_current 0.1285 89.7440 114.2757      NaN     NaN                NaN    NaN    NaN  2012-10-27           2010-10-28      168      82             168             82                     0             NaN            NaN
    regression future_max_points   GradientBoosting  g1_current 0.2951 75.2213 102.7750      NaN     NaN                NaN    NaN    NaN  2012-10-27           2010-10-28      168      82             168             82                     0             NaN            NaN
classification      future_ge900 LogisticRegression  g1_current    NaN     NaN      NaN   0.8919  0.8732             0.8068 0.7761 0.1505  2012-10-27           2010-10-28      168      82             168             82                     0            30.0           36.0
classification      future_ge900       RandomForest  g1_current    NaN     NaN      NaN   0.8599  0.7785             0.6310 0.4490 0.2004  2012-10-27           2010-10-28      168      82             168             82                     0            30.0           36.0
classification      future_ge900   GradientBoosting  g1_current    NaN     NaN      NaN   0.9049  0.8640             0.7313 0.6552 0.1639  2012-10-27           2010-10-28      168      82             168             82                     0            30.0           36.0
classification      future_ge950 LogisticRegression  g1_current    NaN     NaN      NaN   0.8366  0.7724             0.7308 0.6383 0.1786  2012-10-27           2010-10-28      168      82             168             82                     0             7.0           28.0
classification      future_ge950       RandomForest  g1_current    NaN     NaN      NaN   0.7784  0.6380             0.5179 0.0690 0.2469  2012-10-27           2010-10-28      168      82             168             82                     0             7.0           28.0
classification      future_ge950   GradientBoosting  g1_current    NaN     NaN      NaN   0.7801  0.7641             0.7407 0.6512 0.1727  2012-10-27           2010-10-28      168      82             168             82                     0             7.0           28.0
```

## Follow-up exposure

```text
 minimum_future_observations  n_athletes  percent  future_ge900_positive  future_ge950_positive
                           1         289   100.00                     79                     41
                           2         180    62.28                     67                     36
                           3         118    40.83                     56                     33
                           4          71    24.57                     42                     27
```

## Future-maximum regression by minimum follow-up exposure

Gradient boosting with current score, context, implements, and physical tests:

```text
 minimum_future_observations  n_instances  n_unique_athletes                analysis            target            model feature_set  n_rows  n_athletes  r2_fold_mean  r2_fold_ci_low  r2_fold_ci_high  r2_oof  mae_fold_mean  mae_fold_ci_low  mae_fold_ci_high  mae_oof  rmse_fold_mean  rmse_fold_ci_low  rmse_fold_ci_high  rmse_oof
                           1         1041                289 future_horizon_max_min1 future_max_points GradientBoosting  g1_current    1041         289        0.6160          0.4839           0.7482  0.6290        51.3097          37.8982           64.7211  51.3059         71.8846           56.7800            86.9893   72.6971
                           2          737                185 future_horizon_max_min2 future_max_points GradientBoosting  g1_current     737         185        0.5003          0.2949           0.7057  0.5595        54.9447          43.2827           66.6067  54.9348         74.5783           61.9921            87.1646   75.1159
                           3          524                128 future_horizon_max_min3 future_max_points GradientBoosting  g1_current     524         128        0.5261          0.2628           0.7895  0.5606        55.2896          41.0919           69.4873  55.2539         70.7051           52.2189            89.1913   71.9032
```

Paired gradient-boosting increment after adding physical tests:

```text
 minimum_future_observations  n_instances  n_unique_athletes                analysis            target            model     without_physical_tests with_physical_tests  n_rows  n_athletes  bootstrap_replicates  delta_r2  delta_r2_ci_low  delta_r2_ci_high  delta_mae  delta_mae_ci_low  delta_mae_ci_high  delta_rmse  delta_rmse_ci_low  delta_rmse_ci_high
                           1         1041                289 future_horizon_max_min1 future_max_points GradientBoosting context_current_implements          g1_current    1041         289                  2000    0.0397           0.0124            0.0681     2.0682            0.1085             4.0351      3.7903             1.1553              6.4489
                           2          737                185 future_horizon_max_min2 future_max_points GradientBoosting context_current_implements          g1_current     737         185                  2000   -0.0148          -0.0616            0.0241    -0.5828           -3.0537             2.0806     -1.2769            -4.8453              2.1054
                           3          524                128 future_horizon_max_min3 future_max_points GradientBoosting context_current_implements          g1_current     524         128                  2000   -0.0072          -0.0459            0.0271    -1.7214           -4.5362             1.0169     -0.5919            -3.5417              2.2701
```

## Calendar-period sensitivity

The full-cohort out-of-fold predictions were summarized by index-observation period; these are not independent temporal validations.

```text
   period           task           outcome            model                feature_set  n_instances  n_athletes  n_positive     r2     mae    rmse  roc_auc  pr_auc  balanced_accuracy     f1  brier
1997-2005     regression future_max_points GradientBoosting context_current_implements          290         127         NaN 0.4895 52.5725 70.2132      NaN     NaN                NaN    NaN    NaN
1997-2005     regression future_max_points GradientBoosting                 g1_current          290         127         NaN 0.5633 48.6113 64.9376      NaN     NaN                NaN    NaN    NaN
1997-2005 classification      future_ge900     RandomForest            context_current          127         127        23.0    NaN     NaN     NaN   0.8482  0.5089             0.8073 0.5938 0.1446
1997-2005 classification      future_ge900     RandomForest                 g1_current          127         127        23.0    NaN     NaN     NaN   0.8495  0.5262             0.7926 0.6071 0.1358
2006-2012     regression future_max_points GradientBoosting context_current_implements          221          93         NaN 0.3713 61.0964 83.0877      NaN     NaN                NaN    NaN    NaN
2006-2012     regression future_max_points GradientBoosting                 g1_current          221          93         NaN 0.4715 57.6996 76.1831      NaN     NaN                NaN    NaN    NaN
2006-2012 classification      future_ge900     RandomForest            context_current           87          87        24.0    NaN     NaN     NaN   0.8327  0.6108             0.7817 0.6557 0.1634
2006-2012 classification      future_ge900     RandomForest                 g1_current           87          87        24.0    NaN     NaN     NaN   0.8327  0.6259             0.7718 0.6545 0.1550
2013-2020     regression future_max_points GradientBoosting context_current_implements          530          94         NaN 0.5801 50.5926 76.9075      NaN     NaN                NaN    NaN    NaN
2013-2020     regression future_max_points GradientBoosting                 g1_current          530          94         NaN 0.5987 50.1144 75.1853      NaN     NaN                NaN    NaN    NaN
2013-2020 classification      future_ge900     RandomForest            context_current           75          75        32.0    NaN     NaN     NaN   0.9208  0.8569             0.8870 0.8710 0.1163
2013-2020 classification      future_ge900     RandomForest                 g1_current           75          75        32.0    NaN     NaN     NaN   0.9092  0.8931             0.8830 0.8667 0.1219
```

Paired within-period increment after adding the four physical tests:

```text
   period           task           outcome            model     without_physical_tests with_physical_tests  n_instances  n_athletes  n_positive  bootstrap_replicates  delta_r2  delta_r2_ci_low  delta_r2_ci_high  delta_mae  delta_mae_ci_low  delta_mae_ci_high  delta_rmse  delta_rmse_ci_low  delta_rmse_ci_high  delta_roc_auc  delta_roc_auc_ci_low  delta_roc_auc_ci_high  delta_pr_auc  delta_pr_auc_ci_low  delta_pr_auc_ci_high  delta_brier  delta_brier_ci_low  delta_brier_ci_high
1997-2005     regression future_max_points GradientBoosting context_current_implements          g1_current          290         127         NaN                  2000    0.0738           0.0123            0.1508     3.9613            0.7213             7.5887      5.2756             0.8987             10.6380            NaN                   NaN                    NaN           NaN                  NaN                   NaN          NaN                 NaN                  NaN
1997-2005 classification      future_ge900     RandomForest            context_current          g1_current          127         127        23.0                  2000       NaN              NaN               NaN        NaN               NaN                NaN         NaN                NaN                 NaN         0.0013               -0.0314                 0.0310        0.0173              -0.0472                0.1201       0.0088             -0.0022               0.0200
2006-2012     regression future_max_points GradientBoosting context_current_implements          g1_current          221          93         NaN                  2000    0.1001           0.0406            0.1716     3.3968           -0.1152             7.0163      6.9046             2.8065             11.2541            NaN                   NaN                    NaN           NaN                  NaN                   NaN          NaN                 NaN                  NaN
2006-2012 classification      future_ge900     RandomForest            context_current          g1_current           87          87        24.0                  2000       NaN              NaN               NaN        NaN               NaN                NaN         NaN                NaN                 NaN         0.0000               -0.0371                 0.0399        0.0151              -0.1008                0.0931       0.0084             -0.0064               0.0252
2013-2020     regression future_max_points GradientBoosting context_current_implements          g1_current          530          94         NaN                  2000    0.0186          -0.0210            0.0636     0.4783           -2.3904             3.4318      1.7222            -1.9405              5.6429            NaN                   NaN                    NaN           NaN                  NaN                   NaN          NaN                 NaN                  NaN
2013-2020 classification      future_ge900     RandomForest            context_current          g1_current           75          75        32.0                  2000       NaN              NaN               NaN        NaN               NaN                NaN         NaN                NaN                 NaN        -0.0116               -0.0440                 0.0151        0.0361              -0.0391                0.0978      -0.0056             -0.0198               0.0071
```

## Future >=950 validation-fold event counts

```text
 fold  n  n_positive  n_negative
    1 58           9          49
    2 58           8          50
    3 58           8          50
    4 58           8          50
    5 57           8          49
```

Cross-validation intervals are descriptive t intervals across the five fixed folds; they are not inferential confidence intervals.