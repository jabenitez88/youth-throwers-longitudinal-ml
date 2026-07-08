# Results summary

## Dataset

- Records in DatosGrupo1: 1564
- Unique athletes: 502
- Period: 1997-07-02 to 2020-01-23
- Sex counts: {'Masculino': 884, 'Femenino': 680}
- Event counts: {'MARTILLO': 419, 'DISCO': 397, 'PESO': 383, 'JABALINA': 365}
- DatosGrupo2 equals DatosGrupo3: True

## Best future classification models

```text
     outcome feature_set              model  n_rows  positive_rate  auc_mean  auc_ci_low  auc_ci_high  auc_sd  balanced_accuracy_mean  balanced_accuracy_ci_low  balanced_accuracy_ci_high  f1_mean  f1_ci_low  f1_ci_high
future_ge900  g1_current       RandomForest     289          0.273     0.866       0.812        0.920   0.055                   0.813                     0.767                      0.859    0.714      0.650       0.778
future_ge950  g1_current LogisticRegression     289          0.142     0.862       0.815        0.908   0.047                   0.771                     0.738                      0.804    0.495      0.460       0.530
improve_ge50  g1_current LogisticRegression     289          0.391     0.775       0.735        0.815   0.041                   0.702                     0.659                      0.745    0.651      0.603       0.699
```

## Best future next-observation regression models

```text
               analysis                feature_set        model  n_rows  n_athletes      target  r2_mean  r2_ci_low  r2_ci_high  r2_sd  mae_mean  mae_ci_low  mae_ci_high  rmse_mean  rmse_ci_low  rmse_ci_high
future_next_observation context_current_implements RandomForest    1034         289 next_points    0.635      0.515       0.756  0.123    40.500      35.708       45.292     68.365       58.425        78.304
future_next_observation            context_current RandomForest    1034         289 next_points    0.633      0.514       0.753  0.122    40.158      35.411       44.906     68.570       58.937        78.203
future_next_observation                 g1_current RandomForest    1034         289 next_points    0.633      0.508       0.757  0.127    41.336      36.664       46.008     68.532       58.397        78.668
future_next_observation         broader_g1_current RandomForest    1034         289 next_points    0.633      0.508       0.758  0.127    41.379      36.606       46.153     68.551       58.445        78.658
future_next_observation            context_current        Ridge    1034         289 next_points    0.629      0.462       0.797  0.171    43.168      37.167       49.169     68.094       54.786        81.401
```

## Chronological holdout for high performance (future_ge900)

```text
     outcome              model feature_set cutoff_date  n_train  n_test  test_positive_rate   auc  balanced_accuracy    f1
future_ge900 LogisticRegression  g1_current  2012-10-27      207      82               0.439 0.891              0.796 0.765
future_ge900       RandomForest  g1_current  2012-10-27      207      82               0.439 0.864              0.700 0.593
future_ge900   GradientBoosting  g1_current  2012-10-27      207      82               0.439 0.871              0.768 0.727
```

## Permutation importance for future_ge900

```text
       feature  importance_mean  importance_sd
        PUNTOS           0.1052         0.0345
          Sexo           0.0407         0.0104
          Edad           0.0234         0.0110
     ARTEFACTO           0.0125         0.0058
        Prueba           0.0098         0.0086
     Categoria           0.0068         0.0020
   Longitud PJ           0.0047         0.0036
          BOLA           0.0025         0.0024
     Triple PJ           0.0024         0.0028
Salto Vertical           0.0020         0.0030
        Dorsal           0.0004         0.0038
```

## Threshold summary

```text
       threshold  future_max_percentile_rank  current_score_percentile_rank  n_future_positive  future_positive_rate
             900                      73.010                         85.813                 79                 0.273
             950                      85.813                         93.426                 41                 0.142
improvement >=50                         NaN                            NaN                113                 0.391
```
