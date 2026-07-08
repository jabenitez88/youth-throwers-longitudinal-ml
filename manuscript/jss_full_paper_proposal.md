# Athlete-aware longitudinal machine learning supports two-year performance forecasting in youth athletics throwers

**Target journal:** *Journal of Sports Sciences*  
**Article type:** Original Article  
**Prepared as:** Full paper proposal/draft  
**Approximate word count, main text excluding references/tables/figures:** 5,250  
**Style notes applied:** English manuscript; unstructured abstract <=200 words; 3-6 keywords; IMRaD structure; concise results; end-of-manuscript declarations; table and figure captions listed after references.

## Title page

**Title:** Athlete-aware longitudinal machine learning supports two-year performance forecasting in youth athletics throwers

**Short running title:** Longitudinal ML in youth throwers

**Authors:** [Insert author names]

**Affiliations:** [Insert affiliations]

**Corresponding author:** [Insert name, postal address, email]

**Acknowledgements:** The authors acknowledge the coaches, athletes, and data collection staff involved in the long-term monitoring of youth throwers. [Insert exact institutional/RFEA acknowledgement wording.]

**Funding:** [Insert funding statement or "This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors."]

**Disclosure statement:** The authors report no conflict of interest. [Amend if needed.]

**Data availability statement:** The data that support the findings of this study are not publicly available because they contain longitudinal athlete records and are subject to institutional and data-owner restrictions. Analysis code and derived, non-identifiable summary outputs can be made available upon reasonable request, subject to approval by the relevant data owners. [Confirm exact policy before submission.]

**Ethics approval:** [Insert ethics approval committee, approval code, consent/anonymisation statement.]

## Abstract

Machine learning is increasingly used for sport talent development, but repeated athlete observations can inflate performance estimates when validation ignores longitudinal structure. This study evaluated an athlete-aware workflow for two-year performance forecasting in youth athletics throwers. A dataset of 1,564 records from 502 athletes tested between 1997 and 2020 in discus, hammer, javelin, and shot put was analysed. Longitudinal cohorts were constructed from irregular repeated observations, yielding 1,034 transitions from 289 athletes within a 730-day horizon and an athlete-level baseline cohort of 289 athletes. Models were assessed using athlete-grouped cross-validation, chronological holdout validation, incremental feature-set benchmarking, and permutation importance. Physical tests alone showed limited next-score performance (R2 = 0.213), whereas current score plus context reached R2 = 0.633 and maximum performance within 730 days reached R2 = 0.681. Classification of future high performance achieved AUC = 0.866 for score >=900 and AUC = 0.862 for score >=950. Chronological holdout validation for score >=900 achieved AUC = 0.891. Current performance appeared to act as a summary marker of competitive development, while isolated physical tests added limited incremental value.

**Keywords:** athletics; talent development; machine learning; longitudinal modelling; prediction; validation

## 1. Introduction

Machine learning methods are increasingly used to support decision-making in sport, including athlete profiling, performance prediction, training monitoring, and talent development. However, the value of these models depends strongly on how the prediction task is formulated. In youth development settings, the relevant applied question is rarely whether a physical test is associated with performance measured on the same day. Coaches and sport scientists are more often interested in whether information collected at a camp or testing session can help identify athletes likely to improve or reach a higher level over subsequent months or years.

Athletics throwing events are a useful setting in which to study this problem. Performance in discus, hammer, javelin, and shot put depends on physical qualities such as strength and power, but also on age, maturation, event-specific technical development, implement characteristics, training history, and competition experience. This makes prediction attractive but difficult: a model that appears accurate in cross-sectional analysis may be less useful when required to forecast future development.

Repeated-measure datasets introduce an additional methodological challenge. If several observations from the same athlete are randomly split across training and test sets, the model may benefit from athlete-specific information during evaluation. This data leakage can produce optimistic results and may lead to inappropriate conclusions about real-world deployment. Longitudinal sport datasets therefore require validation strategies that respect athlete identity and time.

An additional interpretative issue concerns the role of current competitive performance. It is unsurprising that current score is associated with future score, but in a youth development environment current performance should not be viewed as a trivial single variable. Rather, it can be interpreted as a contextual summary that probably integrates maturation, technical skill, training history, event-specific coordination, competitive experience, and adaptation to the implement. A key practical question is therefore whether isolated physical tests add useful information beyond this competitive summary and its context.

The present study proposes a leakage-aware longitudinal machine learning workflow for youth throwers. The workflow converts irregular repeated testing records into future-prediction cohorts, evaluates models with athlete-grouped validation, adds a chronological holdout analysis, compares incremental feature sets, and estimates model-agnostic variable importance.

**TODO-INTRO-1:** Add a stronger sport-science paragraph on determinants of performance in youth throwing events. Include key literature on strength, explosive power, anthropometry, maturation, technical skill, and event-specific constraints.

**TODO-INTRO-2:** Add a paragraph positioning this paper against previous talent-identification and machine-learning work in throwers. The important contrast is: previous work mostly identifies cross-sectional talent indicators; this paper tests longitudinal prediction and leakage-aware validation.

The thresholds used in this analysis were operational rather than formal selection criteria. The future score thresholds of 900 and 950 represented upper-tail performance within the available cohort: 900 corresponded approximately to the 73rd percentile and 950 to the 86th percentile of future maximum score. The improvement threshold of 50 points was used to represent meaningful medium-term development. **TODO-INTRO-3:** Confirm with domain experts whether these operational thresholds have federation, ranking, or coaching meaning; if not, keep them explicitly framed as empirical high-performance cut-offs.

Therefore, the aims of this study were: (i) to evaluate whether future performance development in youth throwers can be predicted from repeated testing records under leakage-aware validation; (ii) to determine whether physical tests add incremental predictive value beyond current performance and contextual information; and (iii) to identify the variables most influential in future high-performance classification.

## 2. Methods

### 2.1 Study design

This was a retrospective longitudinal modelling study using repeated performance and physical testing records from youth athletics throwers. The analysis was designed to emulate a future decision-support problem: given an athlete's available information at a baseline testing session, can subsequent performance development within approximately two years be predicted?

The computational workflow was implemented as a reproducible Python pipeline (`src/run_longitudinal_ml.py`). The pipeline imports the raw spreadsheet files, standardises labels, constructs longitudinal cohorts, evaluates machine learning models, exports metrics, and generates figures. This allowed all reported tables and figures to be regenerated from the same code path.

### 2.2 Data source and participants

The primary dataset contained 1,564 records from 502 youth throwers tested between 2 July 1997 and 23 January 2020. Records represented four throwing events: hammer (n = 419), discus (n = 397), shot put (n = 383), and javelin (n = 365). The dataset included 884 male and 680 female records. Age categories were U14, U16, U18, U20, and U23, with most observations in U16 to U20.

Repeated observations were available for 297 athletes with at least two records. A separate strength subset contained 325 records from 91 athletes, but two files labelled as separate groups were identical. Therefore, the smaller strength subset was not treated as an independent dataset in the primary analysis.

### 2.3 Predictor variables

The main physical test variables were:

- backward overhead throw (`Dorsal`);
- vertical jump (`Salto Vertical`);
- standing long jump (`Longitud PJ`);
- standing triple jump (`Triple PJ`).

Contextual variables were age, sex, age category, throwing event, medicine ball weight, and implement weight. Current performance score (`PUNTOS`) was included in future-prediction feature sets because any deployable athlete-monitoring model would normally know the athlete's current competitive level at the time of prediction.

Additional variables with lower completeness, including 30 m sprint, flexibility, height, body mass, and arm span, were tested in broader feature sets as sensitivity analyses.

### 2.4 Outcome definitions

Two continuous future outcomes were defined:

1. `next_points`: performance score at the next valid observation within 730 days;
2. `future_max_points`: maximum observed performance score within 730 days.

Three athlete-level classification outcomes were defined using the first eligible baseline observation per athlete:

1. `improve_ge50`: improvement of at least 50 points within 730 days;
2. `future_ge900`: maximum future score of at least 900 points within 730 days;
3. `future_ge950`: maximum future score of at least 950 points within 730 days.

The 730-day horizon was selected to capture medium-term development while retaining a sufficient number of repeated observations. The baseline athlete-level cohort contained 289 athletes. The next-observation transition cohort contained 1,034 transitions from 289 athletes.

### 2.5 Leakage-aware cohort construction

Athlete identity was constructed from athlete name and date of birth. Records were sorted by athlete and date. For the next-observation cohort, each record was paired with the immediately subsequent observation from the same athlete if it occurred within 730 days. For the horizon-maximum cohort, each eligible record was paired with the maximum performance score observed within the subsequent 730 days. For the athlete-level baseline cohort, only the first eligible record per athlete was retained.

This procedure prevented future information from entering baseline predictors and avoided treating repeated observations from the same athlete as independent observations during model evaluation.

### 2.6 Machine learning models and preprocessing

Regression models were ridge regression, random forest regression, and gradient boosting regression. Classification models were logistic regression, random forest classification, and gradient boosting classification.

All models were embedded in preprocessing pipelines. Numeric predictors were imputed using the median and standardised. Categorical predictors were imputed using the most frequent category and one-hot encoded. Preprocessing was fitted only within the training folds to avoid leakage.

Feature sets were compared incrementally:

1. physical tests without current score;
2. current score and prediction horizon;
3. current score plus contextual variables;
4. current score plus context and implement variables;
5. current score plus context and physical tests;
6. current score plus context, physical tests, and broader available variables.

### 2.7 Model validation

Repeated-observation regression models were evaluated using five-fold grouped cross-validation by athlete. This ensured that records from the same athlete did not appear in both training and test folds.

Classification models on the one-row-per-athlete baseline cohort were evaluated using stratified cross-validation. In addition, the `future_ge900` classification task was tested with a chronological holdout. Baseline observations up to 27 October 2012 were used for training, and later observations were reserved for testing. This split was intended to approximate future deployment more closely than random partitioning.

### 2.8 Statistical analysis and performance metrics

Regression performance was quantified using coefficient of determination (R2), mean absolute error (MAE), and root mean squared error (RMSE). Classification performance was quantified using area under the receiver operating characteristic curve (AUC), balanced accuracy, and F1 score. Fold-level 95% confidence intervals were calculated from cross-validation fold estimates as descriptive indicators of metric stability.

Permutation importance was estimated for the `future_ge900` classification task using mean decrease in ROC AUC after shuffling each predictor. This model-agnostic analysis was used to evaluate the relative contribution of current score, contextual variables, and physical tests.

All analyses were conducted in Python using scikit-learn. The complete analysis script exported processed cohorts, summary tables, and manuscript-ready figures.

## 3. Results

### 3.1 Cohort construction and descriptive characteristics

The raw working dataset contained 1,564 records from 502 athletes. Among these, 297 athletes had at least two observations, 196 had at least three observations, and 100 had at least five observations. The next-observation cohort included 1,034 transitions from 289 athletes within a 730-day horizon. The athlete-level baseline cohort included 289 athletes with one baseline record per athlete.

In the athlete-level baseline cohort, 113 athletes (39.1%) improved by at least 50 points within 730 days, 79 athletes (27.3%) reached a future score of at least 900 points, and 41 athletes (14.2%) reached a future score of at least 950 points.

### 3.2 Same-observation performance prediction

When predicting current performance score from the same observation, physical tests alone showed moderate predictive value under athlete-grouped validation. Ridge regression using the four main physical tests achieved R2 = 0.350. Adding contextual variables improved performance. The strongest same-observation model used broader contextual and physical variables with random forest regression (R2 = 0.469).

These values were lower than estimates from random row-level splitting in preliminary analyses, indicating that grouped validation gives a more conservative and realistic estimate when repeated athlete records are present.

### 3.3 Future continuous performance prediction

The incremental feature-set comparison showed a clear separation between models based only on physical tests and models that included current competitive performance. Physical tests without current score achieved R2 = 0.213 (95% CI: 0.098 to 0.327), MAE = 80.35 (95% CI: 75.76 to 84.94), and RMSE = 101.65 (95% CI: 96.67 to 106.62) for next-observation prediction.

Adding current score produced a large improvement. A model using current score alone achieved R2 = 0.616 (95% CI: 0.503 to 0.730), MAE = 41.38 (95% CI: 36.96 to 45.80), and RMSE = 70.33 (95% CI: 61.53 to 79.13). Adding basic context increased performance modestly to R2 = 0.633 (95% CI: 0.514 to 0.753), MAE = 40.16 (95% CI: 35.41 to 44.91), and RMSE = 68.57 (95% CI: 58.94 to 78.20).

The best next-observation model was random forest using current score, context, and implement variables (R2 = 0.635, 95% CI: 0.515 to 0.756; MAE = 40.50, 95% CI: 35.71 to 45.29; RMSE = 68.36, 95% CI: 58.43 to 78.30). Adding physical tests resulted in similar performance (R2 = 0.633, 95% CI: 0.508 to 0.757; MAE = 41.34, 95% CI: 36.66 to 46.01; RMSE = 68.53, 95% CI: 58.40 to 78.67).

For maximum future score within 730 days, the best model was gradient boosting using current score, contextual variables, and physical tests (R2 = 0.681, 95% CI: 0.592 to 0.771; MAE = 44.94; RMSE = 65.23). Models using current score and contextual variables alone performed similarly, suggesting limited incremental gain from physical tests after accounting for current competitive level.

### 3.4 Future high-performance classification

Classification models produced strong discrimination for future high-performance outcomes. For `future_ge900`, random forest using current score, context, and physical tests achieved AUC = 0.866 (95% CI: 0.812 to 0.920), balanced accuracy = 0.813, and F1 = 0.714. For `future_ge950`, logistic regression with the same feature set achieved AUC = 0.862 (95% CI: 0.815 to 0.908), balanced accuracy = 0.771, and F1 = 0.495. For `improve_ge50`, logistic regression achieved AUC = 0.775 (95% CI: 0.735 to 0.815), balanced accuracy = 0.702, and F1 = 0.651.

The chronological holdout analysis for `future_ge900` supported the cross-validation results. Logistic regression achieved AUC = 0.891, balanced accuracy = 0.796, and F1 = 0.765 on the later holdout set. Random forest and gradient boosting achieved AUC = 0.864 and AUC = 0.871, respectively. The holdout test set had a higher positive rate than the full baseline cohort (0.439 vs 0.273), indicating a possible temporal or cohort-composition shift that should be considered when interpreting deployment stability.

### 3.5 Variable importance

Permutation importance for `future_ge900` showed that current performance score was the dominant predictor, with a mean AUC decrease of 0.1052 after permutation. Sex (0.0407), age (0.0234), implement weight (0.0125), event (0.0098), and age category (0.0068) followed.

The four main physical tests showed smaller importance values. Standing long jump had the highest physical-test importance (0.0047), followed by standing triple jump (0.0024), vertical jump (0.0020), and backward overhead throw (0.0004). Thus, physical tests contributed less to future high-performance classification than current score and contextual variables.

## 4. Discussion

This study developed a leakage-aware longitudinal machine learning workflow for youth athletics throwers and evaluated its ability to support two-year performance forecasting. The main findings were that future performance could be forecast with useful accuracy, particularly for high-performance classification, but that current performance and contextual variables were more influential than isolated physical tests.

The best classification models achieved AUC values of approximately 0.86 for future high-performance thresholds, and chronological holdout validation remained strong. This suggests that the modelling approach has potential as a monitoring, watch-list, and prioritisation tool in youth throwing development. However, the models should not be interpreted as deterministic selection systems. Rather, they can provide probabilistic decision support to coaches who already integrate technical, physical, maturational, and contextual information.

The dominance of current competitive score should be interpreted carefully. The result is not simply that "current performance predicts future performance"; instead, current score appears to operate as a compact summary of the athlete's present developmental state. In youth throwers, a competition score is likely to integrate maturation, event-specific technical skill, training history, coordination under the constraints of the event, competitive experience, and adaptation to the age-appropriate implement. From this perspective, the strong contribution of current score is practically informative: isolated physical tests did not replace contextualised competitive performance when the task was longitudinal forecasting rather than same-session profiling. **TODO-DISCUSSION-1:** Add sport-specific references supporting the interpretation of current score as an integrated marker of maturation, technique, training history, and event adaptation.

The feature-set comparison strengthens this interpretation. Physical tests without current score had substantially lower next-score performance (R2 = 0.213) than current score alone (R2 = 0.616). Adding context and implement characteristics produced modest gains, whereas adding physical tests to current score and context did not materially improve performance. This does not mean that physical tests are unimportant for training, monitoring, or understanding performance determinants. Rather, it suggests that, in this cohort and forecasting task, much of their predictive information may already be expressed through current competitive level, or may be too event-specific and technique-dependent to add stable incremental signal in a pooled model. **TODO-DISCUSSION-2:** Add sport-science literature explaining why strength and power tests may be useful for training prescription even when their incremental forecasting value is limited.

**TODO-DISCUSSION-3:** Add a paragraph comparing the results with previous studies on young throwers and talent identification. Be explicit about similarities and differences: previous models identify influential tests; this paper asks whether they improve future prediction under athlete-aware validation.

**TODO-DISCUSSION-4:** Add coaching implications. For example: use model outputs as a watch-list/risk-ranking layer; avoid selecting athletes solely from test scores; interpret probabilities separately by sex/event/category; repeat testing may be more valuable than single measurements.

Sex emerged as the second most influential predictor in the permutation analysis and should not be interpreted as an isolated individual biological determinant. In this dataset, sex is likely entangled with scoring distributions, implement characteristics, age categories, maturation timing, event-specific development pathways, and historical cohort composition. Before practical use, model calibration and error should therefore be examined by sex, event, and age category. If sample size permits, future work should compare pooled models with sex- and event-specific calibration layers or separate models. **TODO-DISCUSSION-5:** Add domain interpretation of sex/event differences and decide whether subgroup calibration should be presented as mandatory future validation or as an additional sensitivity analysis.

From a methodological perspective, the findings reinforce the importance of validation design. Random row-level splits are inappropriate when repeated athlete observations are present because the same athlete may contribute information to both training and test sets. The present workflow addressed this by constructing future-prediction cohorts and using athlete-grouped validation. This is relevant for many sport science datasets, where irregular repeated monitoring is common.

The chronological holdout analysis is also important. Although not a substitute for external validation, it provides a closer approximation to future deployment than cross-validation alone. The maintenance of strong performance in the holdout analysis suggests that the classification models were not merely exploiting random fold structure. However, the later holdout set had a higher positive rate for `future_ge900` than the global baseline cohort (0.439 vs 0.273). This may reflect temporal changes in the monitored population, selection of athletes entering the dataset, event composition, or performance standards. Future implementation should therefore monitor temporal stability, calibration drift, and subgroup-specific error over time.

### 4.1 Strengths and limitations

The main strength of this study is the longitudinal, leakage-aware formulation of a practical talent-development question. The analysis used a long-term dataset, repeated observations, grouped validation, chronological holdout testing, incremental feature-set comparison, and model-agnostic interpretability.

Several limitations should be acknowledged. First, the dataset was retrospective and was not collected specifically for machine learning deployment. Second, key determinants of throwing development, including technical quality, training load, maturity status, injury history, psychological variables, and coach assessments, were not available. Third, the high-performance thresholds were operational and should be refined with domain experts; empirically, 900 and 950 points corresponded approximately to the 73rd and 86th percentiles of future maximum score in the available cohort. Fourth, missing data were uneven across variables, and variables such as maximal strength were available only in a smaller subset, raising possible availability bias and preventing robust primary inference from maximal strength tests. Fifth, external validation on a more recent or independent cohort is required before real-world implementation.

### 4.2 Practical implications

The results suggest that models based on current performance and basic contextual variables can provide useful forecasts of medium-term development. Physical tests remain important for training prescription and athlete monitoring, but their standalone predictive contribution to future performance appears limited once current score and context are known. Coaches should therefore use model outputs as decision-support information for monitoring, watch-list generation, and risk ranking, not as automatic selection decisions. Physical test results should remain part of a broader athlete profile rather than isolated selection indicators.

## 5. Conclusion

A leakage-aware longitudinal machine learning workflow supported two-year performance forecasting in youth athletics throwers with useful accuracy. Future high-performance classification reached AUC values around 0.86-0.89, while future score regression reached R2 values up to 0.68. Current performance, sex, age, event, and implement context were more influential than isolated physical tests. These findings support the use of athlete-aware longitudinal validation in sport performance modelling and caution against overinterpreting same-session or row-random prediction results in repeated-measure athlete datasets.

## References

Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5-32.

Fernandez, E., Izquierdo, J. M., Zarauz, A., & Redondo, J. C. (2023). Prediction of sports talent in young throwers using machine learning. *International Journal of Medicine and Science of Physical Activity and Sport*, 23(93).

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

Zuvela, F., Skoric, J., & Matijasevic, P. (2025). Influence of different strength and power dimensions on success in throwing disciplines. *Human Movement*, 26(3), 79-88. https://doi.org/10.5114/hm/203540

## Tables

**Table 1.** Dataset and longitudinal cohort construction.

| Cohort element | Value |
|---|---:|
| Records in primary dataset | 1,564 |
| Unique athletes | 502 |
| Athletes with >=2 records | 297 |
| Valid next-observation transitions <=730 days | 1,034 |
| Athletes in transition cohort | 289 |
| Athlete-level baseline cohort | 289 |
| Time span | 1997-2020 |

**Table 2.** Incremental feature-set comparison for next-observation regression within 730 days. Values are grouped cross-validated estimates; confidence intervals are fold-level descriptive 95% intervals.

| Feature set | Model | R2 (95% CI) | MAE (95% CI) | RMSE (95% CI) |
|---|---|---:|---:|---:|
| Physical tests without current score | Ridge | 0.213 (0.098-0.327) | 80.35 (75.76-84.94) | 101.65 (96.67-106.62) |
| Current score only | Random forest | 0.616 (0.503-0.730) | 41.38 (36.96-45.80) | 70.33 (61.53-79.13) |
| Current score + context | Random forest | 0.633 (0.514-0.753) | 40.16 (35.41-44.91) | 68.57 (58.94-78.20) |
| Current score + context + implement | Random forest | 0.635 (0.515-0.756) | 40.50 (35.71-45.29) | 68.36 (58.43-78.30) |
| Current score + context + physical tests | Random forest | 0.633 (0.508-0.757) | 41.34 (36.66-46.01) | 68.53 (58.40-78.67) |

**Table 3.** Best grouped cross-validated regression model for maximum future score within 730 days.

| Outcome | Feature set and model | Sample | R2 (95% CI) | Error metrics |
|---|---|---|---:|---:|
| Maximum score <=730 days | Current + context + physical tests; gradient boosting | n=1,041; athletes=289 | 0.681 (0.592-0.771) | MAE=44.94; RMSE=65.23 |

**Table 4.** Best classification models for athlete-level future outcomes. All models used current score, context, and physical tests.

| Outcome | Model | Positive rate | AUC (95% CI) | Bal. acc. | F1 |
|---|---|---:|---:|---:|---:|
| Future score >=900 | Random forest | 0.273 | 0.866 (0.812-0.920) | 0.813 | 0.714 |
| Future score >=950 | Logistic regression | 0.142 | 0.862 (0.815-0.908) | 0.771 | 0.495 |
| Improvement >=50 | Logistic regression | 0.391 | 0.775 (0.735-0.815) | 0.702 | 0.651 |

**Table 5.** Chronological holdout validation for future score >=900 using current score, context, and physical tests. The cut-off date was 2012-10-27.

| Model | Train/test n | Test positive rate | AUC | Bal. acc. | F1 |
|---|---|---:|---:|---:|---:|
| Logistic regression | 207/82 | 0.439 | 0.891 | 0.796 | 0.765 |
| Random forest | 207/82 | 0.439 | 0.864 | 0.700 | 0.593 |
| Gradient boosting | 207/82 | 0.439 | 0.871 | 0.768 | 0.727 |

**Table 6.** Operational threshold summary in the athlete-level baseline cohort.

| Outcome | Threshold | Positive n | Positive rate | Percentile interpretation |
|---|---:|---:|---:|---|
| Improvement within 730 days | >=50 points | 113 | 0.391 | Operational medium-term development |
| Future maximum score | >=900 points | 79 | 0.273 | Approximately 73rd percentile |
| Future maximum score | >=950 points | 41 | 0.142 | Approximately 86th percentile |

**Table 7.** Permutation importance for future score >=900, expressed as mean decrease in ROC AUC.

| Variable | Mean decrease in AUC |
|---|---:|
| Current score | 0.1052 |
| Sex | 0.0407 |
| Age | 0.0234 |
| Implement weight | 0.0125 |
| Event | 0.0098 |
| Age category | 0.0068 |
| Standing long jump | 0.0047 |
| Medicine ball weight | 0.0025 |
| Standing triple jump | 0.0024 |
| Vertical jump | 0.0020 |
| Backward overhead throw | 0.0004 |

## Figure captions

**Figure 1.** Leakage-aware longitudinal cohort construction. The workflow starts from 1,564 cleaned records from 502 athletes, identifies athletes with repeated observations, constructs next-observation transitions within 730 days, and creates a one-record-per-athlete baseline cohort for classification.

**Figure 2.** Cross-validated AUC for future classification outcomes. Models using current score, context, and physical tests are compared with models using current score and context only.

**Figure 3.** Permutation importance for future high-performance classification (`future_ge900`). Importance is expressed as mean decrease in ROC AUC after predictor permutation.

**Figure 4.** Grouped cross-validated R2 for next-observation regression by feature set. The comparison highlights the contrast between physical tests without current score and current score-anchored models. All folds were grouped by athlete to avoid leakage between training and test sets.
