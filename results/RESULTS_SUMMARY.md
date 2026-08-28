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
future_ge900  g1_current       RandomForest     289          0.273     0.866       0.790        0.942   0.055                   0.813                     0.747                      0.878    0.714      0.624       0.805
future_ge950  g1_current LogisticRegression     289          0.142     0.862       0.796        0.927   0.047                   0.771                     0.725                      0.817    0.495      0.446       0.545
improve_ge50  g1_current LogisticRegression     289          0.391     0.775       0.719        0.832   0.041                   0.702                     0.641                      0.763    0.651      0.583       0.719
```

## Best future next-observation regression models

```text
               analysis                feature_set        model  n_rows  n_athletes      target  r2_mean  r2_ci_low  r2_ci_high  r2_sd  mae_mean  mae_ci_low  mae_ci_high  rmse_mean  rmse_ci_low  rmse_ci_high
future_next_observation context_current_implements RandomForest    1034         289 next_points    0.635      0.461       0.810  0.126    41.737      35.298       48.177     68.294       54.188        82.399
future_next_observation            context_current RandomForest    1034         289 next_points    0.635      0.457       0.813  0.128    41.153      33.840       48.466     68.250       53.898        82.601
future_next_observation            context_current        Ridge    1034         289 next_points    0.631      0.393       0.870  0.172    43.138      34.582       51.695     67.886       48.844        86.929
future_next_observation context_current_implements        Ridge    1034         289 next_points    0.629      0.390       0.869  0.172    43.357      34.576       52.137     68.083       48.999        87.168
future_next_observation         broader_g1_current RandomForest    1034         289 next_points    0.629      0.448       0.809  0.130    43.163      36.518       49.808     68.884       54.408        83.360
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
