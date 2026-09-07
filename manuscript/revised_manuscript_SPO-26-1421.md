# Do physical tests add predictive value for medium-term competitive performance in youth throwers? A longitudinal machine-learning study

**Manuscript ID:** SPO-26-1421

**Article type:** Original Research Article

**Authors:** Juan Carlos Redondo-Castán1; Enrique Fernández-Martínez1; Alicia Merayo-Corcoba2,3; Gemma Villarroel-Fernández2,3; José Alberto Benítez-Andrades2,3

1 Department of Physical Education and Sports, University of León, León, Spain

2 ALBA Research Group, School of Industrial, Computer and Aerospace Engineering, University of León, Campus de Vegazana s/n, 24071 León, Spain

3 ALBA Research Group, Instituto de Investigación Biosanitaria de León (IBIOLEÓN), University of León, Avenida Facultad de Veterinaria 25, 24004 León, Spain

**Corresponding author:** Juan Carlos Redondo-Castán, Department of Physical Education and Sports, University of León, León, Spain. Email: jcredc@unileon.es

## Abstract

This study examined whether medium-term competitive performance in youth throwers could be predicted from longitudinal monitoring data and whether physical tests added predictive value beyond current performance and contextual information. A retrospective dataset comprising 1,564 records from 502 youth hammer, discus, shot put, and javelin throwers was analysed using a 730-day horizon. Outcomes included next performance, maximum future performance, improvement of at least 50 points, and attainment of at least 900 or 950 points. Ridge and logistic regression, random forest, and gradient boosting models were evaluated using athlete-grouped or stratified cross-validation and strict chronological holdout analysis. The subsequently observed follow-up interval was excluded from all predictor sets. Physical tests alone showed modest next-performance prediction (R2 = 0.213), whereas current score plus context reached R2 = 0.635. For maximum future performance, gradient boosting showed a small full-cohort increment after adding physical tests (Delta R2 = 0.040, bootstrap 95% interval 0.012 to 0.068), but this did not persist after requiring at least two or three future observations. Classification reached ROC-AUC = 0.866 for at least 900 points and 0.862 for at least 950 points. Strict chronological validation for at least 900 points reached ROC-AUC = 0.892. Current competitive performance was dominant, while physical-test incremental value was small and not robust across outcomes and follow-up restrictions.

**Keywords:** athlete monitoring; performance prediction; talent development; throwing events; machine learning; physical testing

## 1. Introduction

Identifying young athletes with the potential to achieve high levels of performance remains a major challenge in talent-development systems. Although early selection programmes often rely on current competitive results and standardised physical tests, adolescent performance emerges from the interaction of biological development, training history, technical learning, competitive experience, and access to developmental opportunities1,2. Current performance should therefore not be interpreted as a direct proxy for latent future potential. Athletes who appear highly promising at one stage may not maintain their advantage, whereas others may be underestimated despite substantial long-term development. This distinction is particularly important because selection decisions can influence subsequent access to coaching, competition, and specialised support. Thus, an important applied question is whether information collected during athlete monitoring can help forecast observed subsequent improvement or higher levels of performance, rather than merely describe current performance.

Athletics throwing events provide a relevant context in which to examine this problem. Performance in shot put, discus, hammer, and javelin is multidimensional, reflecting the interaction of physical capacities with event-specific technical and biomechanical demands. These relationships are further complicated in developing athletes by growth and biological maturation, which influence anthropometry and the expression of strength, speed, and power during adolescence3. Competitive conditions also change across development because implement specifications vary by age category and sex4. Consequently, associations between physical characteristics and performance at a single time point may not indicate which variables provide additional information about subsequent improvement or future performance.

Physical testing is commonly incorporated into talent-development and athlete-monitoring programmes alongside competitive performance measures. Field-based tests of lower- and upper-body explosive performance, including vertical and horizontal jumps and medicine-ball throws, can characterise athletes' physical capabilities, and strength- and power-related characteristics have been associated with throwing performance5. However, association with current performance and prediction of future development are different questions6. A test may distinguish higher- and lower-performing athletes because it reflects current biological development, training status, or accumulated experience, while providing little additional information about subsequent performance once an athlete's current competitive level and developmental context are known. The practical value of physical testing therefore depends not only on its association with current performance, but also on its incremental predictive contribution. Longitudinal evidence addressing this question in youth throwers remains limited.

Statistical and machine-learning approaches provide flexible frameworks for combining predictors and modelling potentially non-linear relationships. Previous work using the same underlying monitoring system showed that machine-learning models could identify physical and contextual variables associated with competitive performance in young throwers7. However, characterising variables associated with observed performance does not establish their prospective predictive value6. A longitudinal prediction task requires predictors to be defined at an index observation, outcomes to occur subsequently, and model evaluation to reflect the intended forecasting setting. The present study therefore focuses on whether information available during athlete monitoring can forecast subsequent competitive development and, critically, whether physical tests provide additional predictive information beyond current competitive performance and contextual characteristics.

The formulation of the prediction task and validation strategy is particularly important in longitudinal datasets containing repeated observations from the same athletes. If observations from the same athlete are distributed across training and test sets, the model may be evaluated partly on individuals whose other observations contributed to model development, producing optimistic estimates of generalisation performance8. Validation should therefore reflect both the dependency structure of the data and the intended prediction setting9. In the present study, repeated observations were grouped by athlete during cross-validation to prevent the same individual from contributing observations to both training and validation folds. A chronologically later cohort separated by the full outcome window was also used as a complementary temporal evaluation, although this does not constitute external validation because both periods originated from the same monitoring system.

Current competitive performance warrants particular consideration because it may act as an integrated marker of an athlete's developmental state, incorporating, to varying degrees, biological development, technical proficiency, training history, competitive experience, coordination, and adaptation to age-specific implements. The relevant applied question is therefore whether physical tests add useful predictive information beyond this competitive summary and readily available contextual characteristics such as age, sex, event, age category, and implement characteristics. Comparing incrementally constructed predictor sets, and then comparing identical algorithms with and without physical tests on identical validation folds, allows this question to be addressed directly.

To our knowledge, no previous longitudinal study has examined whether commonly used physical performance tests provide incremental predictive value beyond current competitive performance and developmental context when forecasting medium-term competitive development in youth throwers. Accordingly, the aims were: (i) to evaluate whether subsequent competitive performance could be forecast within a two-year horizon using longitudinal athlete-monitoring data and validation procedures accounting for repeated observations; (ii) to determine the incremental predictive value of physical performance tests beyond current competitive performance and developmental context; and (iii) to evaluate subsequent improvement and attainment of predefined performance thresholds, including model calibration and the relative predictive importance of available variables. We hypothesised that models incorporating current competitive performance and developmental context would predict subsequent performance more accurately than models based exclusively on physical tests, and that physical tests would provide a small additional contribution for some, but not necessarily all, prediction outcomes.

## 2. Methods

### 2.1. Study design

This retrospective longitudinal modelling study used repeated competitive-performance and physical-testing records from youth athletics throwers. The analysis emulated a prospective monitoring problem: given information available for an athlete at an index assessment, to what extent could observed competitive development over the following 730 days be forecast? All predictors were restricted to information available at the index assessment. Future information was used exclusively to define outcomes and follow-up descriptors.

The analytical strategy comprised two complementary tasks. First, repeated observations were used to predict continuous performance at the next eligible observation and the maximum performance attained within the prediction horizon. Second, one baseline observation per athlete was used to classify subsequent improvement and attainment of higher performance levels. The workflow comprised data harmonisation, temporally ordered cohort construction, incremental comparison of predictor sets, model development, athlete-aware validation, strict chronological evaluation, calibration, sensitivity analyses, and model interpretation.

### 2.2. Data source and participants

Data were obtained from the historical athlete-monitoring database of the Spanish Athletics Federation's National Technical Development Programme, in which physical testing and competitive performance were routinely collected during national technical development camps. Each row represented an athlete assessment at a specific testing session. The original database comprised 2,032 assessment records. The primary spreadsheet used for longitudinal modelling contained 1,564 records from 502 youth throwers collected between 2 July 1997 and 23 January 2020.

Observations were excluded only when the outcome required for a specific analysis was unavailable. There was no complete-case deletion across every candidate predictor. Eligibility required a valid athlete identifier, index date, current score, and at least one strictly later score within 730 days. The four primary physical tests were complete in the primary spreadsheet. Medicine-ball load was missing in 14 of 1,564 rows and was handled by fold-specific median imputation. Broader secondary variables had greater missingness and were used only in sensitivity feature sets.

The dataset included hammer throw (n = 419 records), discus throw (n = 397), shot put (n = 383), and javelin throw (n = 365); 884 records were from male and 680 from female athletes. Athletes belonged to U14, U16, U18, U20, and U23 categories. Athlete identity was constructed by combining name and date of birth. Names were standardised by trimming leading and trailing spaces, collapsing repeated spaces, and converting to uppercase. No exact duplicate rows were identified. Seven athlete-date combinations contained more than one non-identical row, reflecting multiple events or different test values; same-day rows were not treated as longitudinal transitions.

### 2.3. Measures and candidate predictors

The primary physical tests were the backward overhead medicine-ball throw (Dorsal), countermovement vertical jump (Salto Vertical), standing long jump (Longitud PJ), and standing triple jump (Triple PJ). Testing followed the standardised protocols used by the National Technical Development Programme and described by Fernandez et al.7. Athletes performed three maximal trials of the three jump tests and six maximal medicine-ball throws, separated by approximately 3 min of recovery, and the best trial was retained. Medicine-ball loads were selected by sex and age category. In the baseline cohort, observed loads were 3-4 kg for female U16-U20 athletes, 4-5 kg for male U16 athletes, 5-6 kg for male U18 athletes, and 6 kg for male U20 athletes; the four U14 athletes used 4 kg.

Contextual predictors were age, sex, age category, throwing event, medicine-ball weight, and competition-implement weight. Current competitive performance (`PUNTOS`) was defined as the athlete's best official competitive result achieved and available on or before the index assessment, expressed as points. Performances achieved after the index assessment were not included in predictors. Additional variables with lower completeness, including 30-m sprint, flexibility, height, body mass, and arm span, were used only in broader sensitivity sets.

All competitive marks were converted using the 2017 revised IAAF Scoring Tables10 and the same scoring edition was applied across the study period. Conversion was based on sex, event, and recorded mark; age-specific implement weight was retained as a separate contextual predictor. The points scale facilitated comparison across events, but identical points were not assumed to represent identical developmental meaning across sex, age category, or implement. This limitation was addressed through contextual predictors and subgroup and calendar-period sensitivity summaries.

### 2.4. Outcome definitions and prediction tasks

A 730-day prediction horizon was specified a priori as an operational medium-term window covering approximately two annual training and competition cycles while retaining enough repeated observations for grouped validation. It did not imply complete observation for every athlete.

`next_points` was the score recorded at the first strictly later eligible assessment within 730 days. `future_max_points` was the highest score recorded at any strictly later assessment within the same period. For athlete-level classification, the first eligible index assessment was retained and outcomes were derived from the maximum observed future score. `improve_ge50` indicated `future_max_points - PUNTOS >= 50`; `future_ge900` and `future_ge950` indicated a maximum observed future score of at least 900 or 950 points. The threshold outcomes represented future performance level rather than incident first attainment; athletes already at or above a threshold at baseline remained eligible.

The 50-point cut-off was an empirical operational threshold rather than a validated minimal meaningful change or federation selection criterion. Sensitivity analyses additionally considered improvements of at least 25 and 75 points and continuous score change. The 900- and 950-point thresholds were operational upper-performance cut-offs corresponding approximately to the 73rd and 86th percentiles of future maximum score in this cohort. Athletes who did not reach a threshold in their observed records were classified as observed non-cases, not as athletes known to have been under complete surveillance for all 730 days.

### 2.5. Longitudinal cohort construction

Records were chronologically ordered within athlete. For `next_points`, each eligible assessment was paired with the athlete's next strictly later observation when it occurred within 730 days, generating 1,034 transitions from 289 athletes. For `future_max_points`, each eligible assessment was paired with the maximum score among all strictly later records within 730 days, producing 1,041 prediction instances from 289 athletes. For classification, only the first eligible baseline assessment was retained, producing 289 independent athlete-level prediction instances.

The observed interval from the index assessment to a subsequent record (`delta_days`), the number of future observations, and the observed follow-up span were retained only as follow-up descriptors. They were excluded from all predictor matrices because they are unavailable at the index assessment. Repeated observations from the same athlete were kept together during validation.

### 2.6. Predictor sets and incremental comparisons

Corrected continuous-outcome predictor sets were: (1) four physical tests only; (2) context plus physical tests without current score; (3) current score only; (4) current score plus age, sex, age category, and event; (5) the preceding variables plus medicine-ball and competition-implement weights; (6) the preceding variables plus the four physical tests; and (7) a broader sensitivity set adding 30-m sprint, flexibility, height, body mass, and arm span.

The primary regression incremental comparison held the algorithm, observations, folds, and preprocessing constant and contrasted current score plus context and implement variables with the same set plus the four physical tests. For classification, the paired comparison contrasted current score plus all contextual and implement variables with the identical set plus the four physical tests. Thus, paired differences were attributable to the physical-test block rather than to a change of algorithm or validation partition.

### 2.7. Predictive modelling algorithms and preprocessing

Continuous outcomes were modelled using ridge regression, random forest regression, and gradient boosting regression. Binary outcomes were modelled using logistic regression, random forest classification, and gradient boosting classification. All models were implemented in scikit-learn pipelines so that preprocessing was fitted only in each training fold and then applied to its held-out validation fold.

Numerical variables were imputed using the training-fold median and standardised using `StandardScaler`. Categorical variables were imputed using the most frequent training-fold category and encoded using `OneHotEncoder(handle_unknown="ignore")`.

Hyperparameters were fixed in advance. Ridge regression used `alpha=10.0`. Random forest regression used 200 trees, `min_samples_leaf=5`, `random_state=42`, and `n_jobs=-1`. Gradient boosting regression used `random_state=42` with other parameters at scikit-learn defaults. Logistic regression used `max_iter=3000`, `class_weight="balanced"`, and `random_state=42`. Random forest classification used 200 trees, `min_samples_leaf=5`, `class_weight="balanced_subsample"`, `random_state=42`, and `n_jobs=-1`. Gradient boosting classification used `random_state=42` with other parameters at defaults. No `GridSearchCV`, `RandomizedSearchCV`, systematic hyperparameter selection, nested cross-validation, or post-comparison tuning was performed. The best-performing algorithm was a descriptive selection among three pre-specified candidates.

### 2.8. Model validation

Repeated-observation regression used five-fold `GroupKFold` with athlete identity as the grouping variable, preventing any athlete from appearing in both training and validation partitions. Athlete-level binary outcomes used five-fold `StratifiedKFold` with `shuffle=True` and `random_state=42`. Stratification was generated separately for each outcome. Within each outcome, fold membership was held constant across algorithms and predictor sets to permit paired comparisons.

Strict chronological validation used 27 October 2012, the 70th percentile of baseline index dates, as the start of the later holdout. To ensure that every training outcome window ended before the holdout baseline period, training index dates were restricted to 28 October 2010 or earlier. This created a 730-day gap and yielded 168 training and 82 holdout athletes with no athlete overlap. The same `g1_current` predictor set was used for `future_max_points`, `future_ge900`, and `future_ge950`: current score, age, sex, age category, event, medicine-ball weight, competition-implement weight, backward overhead throw, vertical jump, standing long jump, and standing triple jump.

### 2.9. Performance, uncertainty, calibration, and interpretation

Regression performance was assessed using R2, mean absolute error (MAE), and root mean squared error (RMSE). Classification performance was assessed using ROC-AUC, precision-recall AUC (PR-AUC), balanced accuracy, F1 score, Brier score, calibration intercept, and calibration slope. Calibration curves were generated from out-of-fold probabilities.

Fold-level results are reported as means with descriptive 95% t intervals calculated as mean +/- t(0.975, df=4) x SD/sqrt(5). These intervals describe between-fold variation and are not population confidence intervals. Incremental-value uncertainty was calculated from paired out-of-fold predictions using 2,000 percentile bootstrap replicates. Regression resampling was clustered by athlete. Positive Delta R2, Delta ROC-AUC, Delta PR-AUC, and reductions in MAE, RMSE, or Brier score indicated improvement after adding physical tests.

Because only 41 athletes reached 950 points, a sensitivity analysis used ten repetitions of five-fold stratified cross-validation. Follow-up sensitivity analyses restricted the cohort to athletes with at least two and at least three future observations. For this sensitivity table, R2, MAE, and RMSE were calculated from the pooled out-of-fold predictions so that the displayed R2 values used the same estimand as the paired Delta R2. Additional analyses evaluated 25-, 50-, and 75-point improvement thresholds and continuous change. For calendar-period sensitivity, full-cohort out-of-fold predictions were summarized by the year of the index observation for 1997-2005, 2006-2012, and 2013-2020. These were sensitivity summaries rather than period-specific model fits or independent temporal validations. Gradient boosting was used for paired `future_max_points` comparisons and random forest for paired `future_ge900` comparisons, in each case comparing the same algorithm without and with the four physical tests. Within-period incremental-value intervals used 2,000 paired percentile bootstrap replicates, clustered by athlete for regression. The available database did not contain a version history with which to verify that testing procedures remained completely unchanged across all 23 years.

Permutation importance was calculated for the random forest `future_ge900` model containing current score, contextual and implement variables, and physical tests. An independent three-fold stratified procedure was used. Importance was calculated exclusively on held-out validation data using eight permutations per original variable per fold and ROC-AUC as the scoring metric. Reported means were averaged across folds, and the standard deviation represented variation among the three fold means. Importance was interpreted as model- and correlation-dependent rather than causal.

### 2.10. Statistical software and reproducibility

Analyses used Python 3.12.13, NumPy 1.26.4, pandas 2.2.3, SciPy 1.13.1, scikit-learn 1.9.0, Matplotlib 3.9.2, and openpyxl 3.1.5. The complete version-controlled workflow, aggregate results, and figure-generation materials are publicly available through the version-independent Zenodo record at https://doi.org/10.5281/zenodo.21480517. Raw and row-level processed data are not public because they contain longitudinal information from youth athletes and are subject to data-protection and institutional restrictions.

## 3. Results

### 3.1. Participant characteristics and cohort construction

The primary dataset comprised 1,564 records from 502 athletes. Repeated observations were available for 297 athletes with at least two assessments, 196 with at least three, and 100 with at least five. The eligible baseline cohort contained 289 athletes (160 male and 129 female), with mean age 15.42 +/- 1.16 years (range 12.87-18.98). Baseline characteristics are reported in Table 1.

The 730-day horizon yielded 1,034 next-observation transitions and 1,041 future-maximum instances from 289 athletes. In the athlete-level cohort, 113 athletes (39.1%) improved by at least 50 points, 79 (27.3%) recorded a future score of at least 900 points, and 41 (14.2%) recorded a future score of at least 950 points. At baseline, 41 athletes (14.2%) were already at or above 900 points and 19 (6.6%) were already at or above 950 points. Among the 79 future 900-point cases, 35 were already at or above 900 at baseline and 44 were below 900 before subsequently attaining the threshold. Among the 41 future 950-point cases, the corresponding counts were 18 and 23. Athletes had a median of 2 future observations (IQR 1-3; range 1-13) and median observed follow-up span of 371 days (IQR 210-546; range 21-730). Athletes who reached 900 or 950 points had more observed records and longer follow-up than observed non-cases, indicating potential informative observation exposure.

### 3.2. Incremental prediction of continuous performance

Physical tests alone showed modest next-observation prediction (R2 = 0.213, descriptive 95% interval 0.051-0.375; MAE = 80.35 points; RMSE = 101.65 points). Context plus physical tests without current score reached R2 = 0.302. Current score alone reached R2 = 0.611, and adding age, sex, category, and event increased R2 to 0.635. Adding implement context produced R2 = 0.635. The full set including physical tests reached R2 = 0.628 when the best algorithm for that feature set was selected.

Paired within-algorithm comparisons showed no consistent incremental value of physical tests for next performance. Delta R2 ranged from -0.007 to 0.001 and every bootstrap interval included zero. For maximum future score, physical tests alone reached R2 = 0.250 and context plus physical tests without current score reached R2 = 0.351. Current score alone reached R2 = 0.559. The corrected best full model was gradient boosting using current score, context, implements, and physical tests (R2 = 0.616, descriptive 95% interval 0.484-0.748; MAE = 51.31; RMSE = 71.88).

To translate point-scale error into applied terms, exact entries from the sex- and event-specific 2017 revised IAAF Scoring Tables were examined at 800 points, a representative level within the baseline score distribution (median 772; IQR 710-841). In women's shot put, 13.46 m corresponds to 800 points; 760 and 840 points correspond to 12.81 and 14.10 m, and 755 and 845 points correspond to 12.73 and 14.18 m. Thus, around this level, an absolute difference of 40-45 points represents approximately 0.64-0.73 m. In men's discus throw, 45.76 m corresponds to 800 points; 760 and 840 points correspond to 43.57 and 47.95 m, and 755 and 845 points correspond to 43.30 and 48.23 m. The corresponding 40-45-point difference is approximately 2.19-2.47 m. Because the scoring tables are progressive and sex- and event-specific, these are local examples rather than fixed distance equivalents across performances.

For maximum future score, the paired gradient boosting comparison showed Delta R2 = 0.040 (bootstrap 95% interval 0.012-0.068) and RMSE reduction = 3.79 points (1.16-6.45). Ridge and random forest intervals included zero. Thus, physical tests showed a small algorithm-dependent increment for maximum performance but no uniform contribution across algorithms or continuous outcomes.

### 3.3. Improvement and performance-threshold classification

Classification was stronger for future performance level than for individual improvement. Random forest with the full predictor set reached ROC-AUC = 0.866 (descriptive 95% interval 0.790-0.942), PR-AUC = 0.717, balanced accuracy = 0.813, F1 = 0.714, and Brier score = 0.138 for at least 900 points. Logistic regression reached ROC-AUC = 0.862 (0.796-0.927), PR-AUC = 0.628, balanced accuracy = 0.771, F1 = 0.495, and Brier score = 0.147 for at least 950 points. Logistic regression reached ROC-AUC = 0.775 (0.719-0.832), PR-AUC = 0.709, balanced accuracy = 0.702, F1 = 0.651, and Brier score = 0.197 for improvement of at least 50 points.

For the same algorithms on identical folds, the corresponding ROC-AUC values without and with physical tests were 0.859 and 0.866 for at least 900 points, 0.853 and 0.862 for at least 950 points, and 0.774 and 0.775 for at least 50-point improvement. Paired ROC-AUC bootstrap intervals all included zero. Some PR-AUC or Brier differences favoured physical tests for individual algorithm-outcome combinations, but no consistent classification increment was evident.

The five principal validation folds for `future_ge950` contained 9, 8, 8, 8, and 8 positive cases (fold sizes 58, 58, 58, 58, and 57). The ten-times repeated five-fold analysis supported the overall discrimination estimate while showing the expected variability of a 41-event endpoint. For random forest with physical tests, mean ROC-AUC across repeats was 0.844, with the repeat-level 2.5th-97.5th percentile range 0.827-0.866. Calibration metrics and curves indicated that discrimination should not be interpreted as interchangeable with well-calibrated individual probabilities.

### 3.4. Follow-up and threshold sensitivity

Among athletes with at least two future observations (n = 180), random forest ROC-AUC was 0.854 for at least 900 points and 0.825 for at least 950 points. Among those with at least three future observations (n = 118), corresponding ROC-AUC values were 0.899 and 0.806. Predictive signal therefore remained after restricting observation exposure, although these analyses do not eliminate informative censoring and their cohort composition differed.

The continuous `future_max_points` analysis was also repeated after restricting each index instance to at least two or at least three future observations within its 730-day window. The restrictions retained 737 instances from 185 athletes and 524 instances from 128 athletes, respectively. From pooled out-of-fold predictions, gradient boosting with current score, context, implements, and physical tests achieved R2 = 0.560, MAE = 54.93, and RMSE = 75.12 in the at-least-two cohort and R2 = 0.561, MAE = 55.25, and RMSE = 71.90 in the at-least-three cohort.

In paired gradient-boosting comparisons, the physical-test increment did not persist. Delta R2 was -0.015 (bootstrap 95% interval -0.062 to 0.024) when at least two future observations were required and -0.007 (-0.046 to 0.027) when at least three were required. Corresponding RMSE reductions were -1.28 points (-4.85 to 2.11) and -0.59 points (-3.54 to 2.27), where negative values indicate worse performance after adding physical tests. Continuous prediction therefore retained moderate performance, but the small full-cohort physical-test increment was not robust to follow-up-exposure restrictions.

Sensitivity analyses using improvement thresholds of 25, 50, and 75 points and continuous score change did not support a robust general classification increment from physical tests. Calendar-period results are reported in Table 9. For `future_max_points`, the periods 1997-2005, 2006-2012, and 2013-2020 comprised 290 instances/127 athletes, 221/93, and 530/94, respectively. Gradient-boosting R2 without versus with physical tests was 0.489 versus 0.563, 0.371 versus 0.471, and 0.580 versus 0.599. Paired Delta R2 was 0.074 (bootstrap 95% interval 0.012-0.151), 0.100 (0.041-0.172), and 0.019 (-0.021 to 0.064), respectively.

For `future_ge900`, the three periods contained 127 athletes/23 positive cases, 87/24, and 75/32. Random-forest ROC-AUC without versus with physical tests was 0.848 versus 0.849, 0.833 versus 0.833, and 0.921 versus 0.909. Paired Delta ROC-AUC was 0.001 (-0.031 to 0.031), 0.000 (-0.037 to 0.040), and -0.012 (-0.044 to 0.015), respectively. Thus, the absence of a clear classification increment was consistent across periods, whereas the regression increment was larger in the first two periods and small and uncertain in 2013-2020.

### 3.5. Strict chronological holdout

The strict split contained 168 training athletes and 82 later holdout athletes. For `future_ge900`, 30/168 (17.9%) training athletes and 36/82 (43.9%) holdout athletes were positive, demonstrating substantial temporal or cohort-composition shift. Logistic regression achieved ROC-AUC = 0.892, PR-AUC = 0.873, balanced accuracy = 0.807, F1 = 0.776, and Brier score = 0.150. Gradient boosting achieved ROC-AUC = 0.905 and random forest 0.860.

For `future_ge950`, 7/168 training athletes and 28/82 holdout athletes were positive; logistic regression achieved ROC-AUC = 0.837. For `future_max_points`, holdout R2 was 0.320 for ridge regression, 0.129 for random forest, and 0.295 for gradient boosting. Continuous-outcome transport was therefore materially weaker than grouped cross-validation performance.

### 3.6. Permutation importance

Current competitive performance produced the largest mean decrease in ROC-AUC when permuted (0.1052), followed by sex (0.0407), age (0.0234), implement weight (0.0125), event (0.0098), and age category (0.0068). Physical tests had smaller values: standing long jump 0.0047, standing triple jump 0.0024, vertical jump 0.0020, and backward overhead medicine-ball throw 0.0004. These values describe the fitted model under correlated predictors and do not demonstrate causal irrelevance.

## 4. Discussion

This study examined whether medium-term competitive development in youth throwers could be forecast from longitudinal monitoring data and whether commonly used physical tests provided information beyond current competitive performance and developmental context. Three main findings emerged. First, current score and context contained substantially more predictive information than isolated physical tests. Second, paired comparisons showed that physical tests did not consistently improve next-performance regression or classification; a small increment was observed for maximum future score with gradient boosting, but not with ridge or random forest. Third, future attainment of 900 and 950 points was predicted more accurately than 50-point improvement, while strict chronological evaluation retained strong discrimination for threshold outcomes but showed weaker continuous-outcome transport and marked prevalence shift.

The dominant role of current competitive performance is consistent with multidimensional models of talent development, which conceptualise performance as the interaction of biological, physical, technical, psychological, and environmental factors rather than the consequence of any single characteristic11,12. Competitive score should not be interpreted as an isolated biological determinant or as latent future potential. Within the two-year horizon examined, it appears to act as a compact summary of the athlete's present developmental state, incorporating technical proficiency, training adaptations, coordination, competitive experience, and adaptation to the event and implement. This interpretation is consistent with athlete-monitoring approaches that advocate integration of multiple information sources13,14.

The limited incremental contribution of physical testing should not be interpreted as evidence that strength and power are unimportant for throwing performance. Strength-, power-, and speed-related characteristics are associated with throwing performance5,15,16, and the same tests showed meaningful contemporaneous associations in previous work7. Association and prediction nevertheless address different questions6. Once current score and contextual information were included, part of the information captured by jumping and medicine-ball throwing may already have been represented in the athlete's competitive result. Physical tests also retain applied value for monitoring training adaptation, identifying modifiable strengths and weaknesses, and guiding physical preparation even when their additional forecasting contribution is small.

The paired design materially changes the interpretation of incremental value. The original comparison for maximum future score changed both predictor set and best-performing algorithm, so the observed difference could not be attributed solely to physical tests. By holding algorithm, rows, folds, and preprocessing constant, the revised full-cohort analysis showed a gradient-boosting increment of 0.040 in R2, whereas intervals for ridge and random forest included zero. However, this gradient-boosting increment disappeared when at least two or three future observations were required. For classification, all paired ROC-AUC intervals included zero. The evidence therefore supports, at most, a small and sample-dependent contribution rather than a stable general increment from physical testing.

Future attainment of the 900- and 950-point thresholds was predicted more accurately than individual improvement. Threshold attainment depends strongly on baseline competitive status, whereas improvement requires forecasting change arising between assessments from biological, technical, behavioural, and environmental processes11,17. Training exposure, technical development, injury, coaching practice, psychological characteristics, and biological maturation were unavailable. Change outcomes are also more variable because they depend on both baseline and follow-up measurements18. Sensitivity analyses across 25-, 50-, and 75-point cut-offs and continuous change reinforced the distinction between forecasting a future performance state and forecasting individual developmental change.

Unequal follow-up exposure is an important limitation of the outcome definition. Athletes with more future observations had more opportunity to record a high maximum, and positive cases had longer observed follow-up than observed non-cases. Restricting analyses to athletes with at least two or three future observations preserved discrimination, but it cannot establish that those without a recorded threshold event were continuously observed true negatives. The results therefore concern attainment within the available monitoring records. Prospective studies should standardise follow-up schedules or use time-to-event and censoring-aware methods when complete event ascertainment is available.

The validation strategy is central to interpreting the findings. Repeated athlete records are not independent, and placing observations from the same athlete in training and validation data can produce optimistic estimates8,9. Athlete-grouped cross-validation provided a more realistic assessment of generalisation to unseen athletes, while identical fold assignment supported paired model comparisons19. The strict chronological design further ensured that training outcome windows ended before the later baseline period.

Temporal discrimination for performance thresholds remained high despite substantial prevalence shift. This suggests that ranking signal persisted across time, but calibration and operating thresholds cannot be assumed stable. The much lower holdout R2 for `future_max_points` indicates weaker transport of continuous predictions. Calendar-period OOF summaries showed no clear physical-test increment in `future_ge900` discrimination in any period, but the continuous-outcome increment was larger in 1997-2005 and 2006-2012 than in 2013-2020. These descriptive differences may reflect sampling variability as well as historical changes in cohort composition, monitoring practice, coaching systems, competition opportunities, or developmental pathways; they do not establish a calendar-period effect. Although the programme used standardised testing procedures, the available database did not document protocol versions sufficiently to verify complete invariance across 1997-2020. External validation in a newer and independent monitoring system remains necessary.

Sex was the second most important variable in permutation analysis, but it should not be interpreted as an isolated individual biological determinant. In this pooled dataset, sex is entangled with scoring distributions, implement characteristics, age categories, maturation timing, event-specific pathways, and historical composition. Similarly, the operational 900- and 950-point cut-offs cannot be assumed to have identical developmental meaning across every subgroup. Future applied work should examine calibration and error by sex, event, age category, and implement, and compare pooled models with subgroup-specific calibration or models where sample size permits.

From an applied perspective, the findings argue against using isolated physical-test scores as stand-alone selection indicators. Predictive models may support watch-list generation, monitoring, and prioritisation by combining longitudinal competitive and contextual information, but they should complement rather than replace expert judgement. Predicted probabilities require calibration assessment in the deployment population, and physical assessments should remain part of a broader athlete profile for training prescription and development monitoring.

Several limitations should be acknowledged. First, the retrospective database was not designed for prospective model deployment. Second, follow-up timing and frequency were irregular, and outcome non-attainment represented an observed non-case rather than guaranteed complete surveillance. Third, relevant determinants including biological maturation, training exposure, technical quality, injury history, coaching, and psychosocial factors were unavailable. Fourth, exhaustive hyperparameter optimisation and nested cross-validation were not performed; descriptive best-algorithm selection used the same cross-validation framework as performance estimation and may retain selection optimism19. Fifth, permutation importance is model- and correlation-dependent. Sixth, the operational thresholds and pooled scoring approach require subgroup and external validation. Finally, the chronological holdout was internal to the same national programme and cannot establish transportability to other systems.

In conclusion, current competitive performance and developmental context contained substantially more predictive information about medium-term competitive performance in youth throwers than isolated physical-test scores. Physical tests showed no consistent incremental contribution to next-performance or classification models, although gradient boosting showed a small paired increment for maximum future score. Athlete-aware validation, paired feature-block comparisons, calibration, and strict temporal evaluation provided a more defensible estimate of model performance. These models may provide model-based decision support for longitudinal athlete monitoring, but external validation, standardised follow-up, and subgroup calibration are required before applied selection use.

## Statements and Declarations

### Ethical considerations

The study was conducted in accordance with the principles of the Declaration of Helsinki and was approved by the Research Ethics Committee of the University of León.

### Consent to participate

Written informed consent was obtained from all participants. For participants younger than 18 years, written informed consent was obtained from their parents or legal guardians before participation.

### Consent for publication

Not applicable.

### Declaration of conflicting interest

The authors declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article.

### Funding statement

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

### Data availability

Raw and row-level processed data are not publicly available because they contain longitudinal information that could compromise athlete privacy. De-identified data may be available from the corresponding author upon reasonable request, subject to applicable ethical, institutional, and data-protection requirements. Aggregate results and analysis code are publicly available through Zenodo.

### Author contributions

JCR conceptualised the study, supervised the research, and contributed to data interpretation and manuscript preparation. EFM contributed to data curation, methodology, statistical analysis, and manuscript preparation. AM-C, GV-F, and JAB-A contributed to methodology, data analysis, interpretation of results, and critical revision of the manuscript. All authors reviewed and approved the final version.

## References

1. Vaeyens R, Lenoir M, Williams AM, et al. Talent identification and development programmes in sport: current models and future directions. *Sports Medicine* 2008; 38: 703-714. DOI: 10.2165/00007256-200838090-00001.

2. Till K and Baker J. Challenges and possible solutions to optimizing talent identification and development in sport. *Frontiers in Psychology* 2020; 11: 664. DOI: 10.3389/fpsyg.2020.00664.

3. Malina RM, Rogol AD, Cumming SP, et al. Biological maturation of youth athletes: assessment and implications. *British Journal of Sports Medicine* 2015; 49: 852-859. DOI: 10.1136/bjsports-2015-094623.

4. International Association of Athletics Federations. *World Athletics Competition and Technical Rules 2020 Edition*. International Association of Athletics Federations, 2019.

5. Zuvela F, Skoric J and Matijasevic P. Influence of different strength and power dimensions on success in throwing disciplines. *Human Movement* 2025; 26: 79-88. DOI: 10.5114/hm/203540.

6. Shmueli G. To explain or to predict? *Statistical Science* 2010; 25: 289-310. DOI: 10.1214/10-STS330.

7. Fernandez E, Izquierdo J, Zarauz A, et al. Prediction of sports talent in young throwers using machine learning. *Revista Internacional de Medicina y Ciencias de la Actividad Fisica y del Deporte* 2023; 23(93): 185-199.

8. Kapoor S and Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 2023; 4: 100804. DOI: 10.1016/j.patter.2023.100804.

9. Roberts DR, Bahn V, Ciuti S, et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 2017; 40: 913-929. DOI: 10.1111/ecog.02881.

10. Spiriev B and Spiriev A. *IAAF Scoring Tables of Athletics: 2017 Revised Edition*. IAAF, 2017.

11. Johnston K, Wattie N, Schorer J, et al. Talent identification in sport: a systematic review. *Sports Medicine* 2018; 48: 97-109. DOI: 10.1007/s40279-017-0803-2.

12. Bourdon PC, Cardinale M, Murray A, et al. Monitoring athlete training loads: consensus statement. *International Journal of Sports Physiology and Performance* 2017; 12: S2-161-S2-170. DOI: 10.1123/IJSPP.2017-0208.

13. Kiely M, Warrington G, McGoldrick A, et al. Physiological and performance monitoring in competitive sporting environments: a review for elite individual sports. *Strength & Conditioning Journal* 2019; 41: 62-74. DOI: 10.1519/SSC.0000000000000493.

14. Montull L, Slapsinskaite-Dackeviciene A, Kiely J, et al. Integrative proposals of sports monitoring: subjective outperforms objective monitoring. *Sports Medicine - Open* 2022; 8: 41. DOI: 10.1186/s40798-022-00432-z.

15. Judge WL, Bellar D, McAtee G, et al. Predictors of personal best performance in the hammer throw for US collegiate throwers. *International Journal of Performance Analysis in Sport* 2010; 10: 54-65.

16. Suchomel TJ, Nimphius S and Stone MH. The importance of muscular strength in athletic performance. *Sports Medicine* 2016; 46: 1419-1449. DOI: 10.1007/s40279-016-0486-0.

17. Abbott A and Collins D. Eliminating the dichotomy between theory and practice in talent identification and development: considering the role of psychology. *Journal of Sports Sciences* 2004; 22: 395-408. DOI: 10.1080/02640410410001675324.

18. Steyerberg EW. Statistical models for prediction. In: *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating*. Springer, 2019, pp. 59-93.

19. Raschka S. Model evaluation, model selection, and algorithm selection in machine learning. *arXiv* 2018:1811.12808. DOI: 10.48550/arXiv.1811.12808.

## Tables

**Table 1.** Characteristics of the athlete-level baseline cohort (n = 289).

| Characteristic | Value |
|---|---:|
| Age, years, mean +/- SD (range) | 15.42 +/- 1.16 (12.87-18.98) |
| Male, n (%) | 160 (55.4) |
| Female, n (%) | 129 (44.6) |
| U14 / U16 / U18 / U20, n | 4 / 160 / 107 / 18 |
| Discus / hammer / shot put / javelin, n | 77 / 75 / 74 / 63 |
| Current score, mean +/- SD | 780.20 +/- 110.51 |
| Backward overhead throw, m, mean +/- SD | 13.42 +/- 2.14 |
| Vertical jump, cm, mean +/- SD | 45.71 +/- 8.94 |
| Standing long jump, m, mean +/- SD | 2.34 +/- 0.26 |
| Standing triple jump, m, mean +/- SD | 6.90 +/- 0.90 |
| Medicine-ball load, kg, median (range) | 5.0 (3.0-6.0) |
| Implement weight, kg, median (range) | 3.0 (0.5-6.0) |

**Table 2.** Cohort construction, follow-up, and outcome frequencies.

| Cohort element | Value |
|---|---:|
| Records in primary dataset | 1,564 |
| Unique athletes | 502 |
| Athletes with at least 2 / 3 / 5 records | 297 / 196 / 100 |
| Next-observation transitions within 730 days | 1,034 from 289 athletes |
| Future-maximum instances within 730 days | 1,041 from 289 athletes |
| Athlete-level baseline cohort | 289 athletes |
| Future observations, median (IQR; range) | 2 (1-3; 1-13) |
| Observed follow-up days, median (IQR; range) | 371 (210-546; 21-730) |
| Improvement at least 50 points, n (%) | 113 (39.1) |
| Future score at least 900 points, n (%) | 79 (27.3) |
| Future score at least 950 points, n (%) | 41 (14.2) |

**Table 3.** Corrected best-algorithm performance across predictor sets for continuous outcomes. The best algorithm was selected descriptively by the highest mean cross-validated R2 among the three pre-specified candidates. `delta_days` was excluded from every predictor set.

| Predictor set | Next algorithm | Next R2 | Next MAE / RMSE | Future-max algorithm | Future max R2 | Future max MAE / RMSE |
|---|---|---:|---:|---|---:|---:|
| Physical tests only | Ridge | 0.213 | 80.35 / 101.65 | Ridge | 0.250 | 82.22 / 100.80 |
| Context + physical tests, no current score | Ridge | 0.302 | 74.93 / 95.93 | Ridge | 0.351 | 76.82 / 94.05 |
| Current score only | Ridge | 0.611 | 43.49 / 69.59 | Ridge | 0.559 | 55.74 / 76.39 |
| Current score + athlete/event context | Random forest | 0.635 | 41.15 / 68.25 | Ridge | 0.600 | 53.44 / 73.02 |
| Current + context + implements | Random forest | 0.635 | 41.74 / 68.29 | Ridge | 0.598 | 53.42 / 73.18 |
| Current + context + implements + physical tests | Ridge | 0.628 | 43.71 / 68.20 | Gradient boosting | 0.616 | 51.31 / 71.88 |
| Full broader sensitivity set | Random forest | 0.629 | 43.16 / 68.88 | Ridge | 0.606 | 53.75 / 72.71 |

**Table 4.** Paired within-algorithm incremental value of adding physical tests. Positive Delta R2 and positive RMSE reduction indicate improvement; intervals are 2,000-replicate athlete-cluster bootstrap percentile intervals.

| Outcome | Algorithm | Delta R2 (95% interval) | RMSE reduction, points (95% interval) |
|---|---|---:|---:|
| Next observation | Ridge | -0.000 (-0.008 to 0.007) | -0.02 (-0.81 to 0.73) |
| Next observation | Random forest | -0.007 (-0.016 to 0.002) | -0.70 (-1.62 to 0.18) |
| Next observation | Gradient boosting | 0.001 (-0.031 to 0.035) | 0.09 (-3.04 to 3.22) |
| Maximum within 730 days | Ridge | 0.010 (-0.006 to 0.026) | 0.97 (-0.58 to 2.57) |
| Maximum within 730 days | Random forest | 0.006 (-0.012 to 0.025) | 0.59 (-1.17 to 2.33) |
| Maximum within 730 days | Gradient boosting | 0.040 (0.012 to 0.068) | 3.79 (1.16 to 6.45) |

**Table 5.** Same-algorithm classification performance without and with physical tests.

| Outcome / model | Physical tests | ROC-AUC | PR-AUC | Balanced accuracy | F1 | Brier |
|---|---|---:|---:|---:|---:|---:|
| Improvement at least 50 / logistic | Without | 0.774 | 0.707 | 0.730 | 0.681 | 0.198 |
| Improvement at least 50 / logistic | With | 0.775 | 0.709 | 0.702 | 0.651 | 0.197 |
| Future at least 900 / random forest | Without | 0.859 | 0.676 | 0.818 | 0.705 | 0.143 |
| Future at least 900 / random forest | With | 0.866 | 0.717 | 0.813 | 0.714 | 0.138 |
| Future at least 950 / logistic | Without | 0.853 | 0.606 | 0.746 | 0.460 | 0.155 |
| Future at least 950 / logistic | With | 0.862 | 0.628 | 0.771 | 0.495 | 0.147 |

**Table 6.** Strict chronological holdout with a full 730-day gap. Every model used the same `g1_current` predictor set: current score, age, sex, age category, event, medicine-ball weight, competition-implement weight, and the four physical tests.

| Outcome / model | Train / test n | Train / test positives | Performance |
|---|---:|---:|---|
| Future max / ridge | 168 / 82 | Not applicable | R2 0.320; MAE 79.12; RMSE 100.95 |
| Future max / random forest | 168 / 82 | Not applicable | R2 0.129; MAE 89.74; RMSE 114.28 |
| Future max / gradient boosting | 168 / 82 | Not applicable | R2 0.295; MAE 75.22; RMSE 102.78 |
| Future at least 900 / logistic | 168 / 82 | 30 / 36 | ROC-AUC 0.892; PR-AUC 0.873; Brier 0.150 |
| Future at least 900 / random forest | 168 / 82 | 30 / 36 | ROC-AUC 0.860; PR-AUC 0.779; Brier 0.200 |
| Future at least 900 / gradient boosting | 168 / 82 | 30 / 36 | ROC-AUC 0.905; PR-AUC 0.864; Brier 0.164 |
| Future at least 950 / logistic | 168 / 82 | 7 / 28 | ROC-AUC 0.837; PR-AUC 0.772; Brier 0.179 |

**Table 7.** Permutation importance for future attainment of at least 900 points, expressed as mean decrease in held-out ROC-AUC.

| Variable | Mean decrease in ROC-AUC |
|---|---:|
| Current score | 0.1052 |
| Sex | 0.0407 |
| Age | 0.0234 |
| Implement weight | 0.0125 |
| Event | 0.0098 |
| Age category | 0.0068 |
| Standing long jump | 0.0047 |
| Medicine-ball weight | 0.0025 |
| Standing triple jump | 0.0024 |
| Vertical jump | 0.0020 |
| Backward overhead throw | 0.0004 |

**Table 8.** Follow-up-exposure sensitivity for `future_max_points` using Gradient Boosting and five-fold athlete-grouped cross-validation. R2, MAE, and RMSE are calculated from pooled out-of-fold predictions; Delta R2 uses the same paired out-of-fold predictions.

| Minimum future observations | Instances / athletes | Predictor set | R2 | MAE | RMSE | Delta R2 (95% bootstrap interval) |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1,041 / 289 | Without physical tests | 0.589 | 53.37 | 76.49 | Reference |
| 1 | 1,041 / 289 | With physical tests | 0.629 | 51.31 | 72.70 | 0.040 (0.012 to 0.068) |
| 2 | 737 / 185 | Without physical tests | 0.574 | 54.35 | 73.84 | Reference |
| 2 | 737 / 185 | With physical tests | 0.560 | 54.93 | 75.12 | -0.015 (-0.062 to 0.024) |
| 3 | 524 / 128 | Without physical tests | 0.568 | 53.53 | 71.31 | Reference |
| 3 | 524 / 128 | With physical tests | 0.561 | 55.25 | 71.90 | -0.007 (-0.046 to 0.027) |

**Table 9.** Calendar-period sensitivity summaries from full-cohort out-of-fold predictions. Intervals are 2,000-replicate paired bootstrap percentile intervals; regression resampling was clustered by athlete. These summaries are not independent temporal validations.

| Outcome / algorithm | Period | Instances / athletes (positive cases) | Without physical tests | With physical tests | Paired increment (95% bootstrap interval) |
|---|---|---:|---:|---:|---:|
| `future_max_points` / Gradient Boosting (R2) | 1997-2005 | 290 / 127 | 0.489 | 0.563 | 0.074 (0.012 to 0.151) |
| `future_max_points` / Gradient Boosting (R2) | 2006-2012 | 221 / 93 | 0.371 | 0.471 | 0.100 (0.041 to 0.172) |
| `future_max_points` / Gradient Boosting (R2) | 2013-2020 | 530 / 94 | 0.580 | 0.599 | 0.019 (-0.021 to 0.064) |
| `future_ge900` / Random Forest (ROC-AUC) | 1997-2005 | 127 / 127 (23) | 0.848 | 0.849 | 0.001 (-0.031 to 0.031) |
| `future_ge900` / Random Forest (ROC-AUC) | 2006-2012 | 87 / 87 (24) | 0.833 | 0.833 | 0.000 (-0.037 to 0.040) |
| `future_ge900` / Random Forest (ROC-AUC) | 2013-2020 | 75 / 75 (32) | 0.921 | 0.909 | -0.012 (-0.044 to 0.015) |

## Figure captions

**Figure 1.** Athlete-aware study design and cohort construction. The observed interval to a future record was retained only as a follow-up descriptor and excluded from predictors.

**Figure 2.** Cross-validated ROC-AUC for future classification outcomes, comparing current score and context with the same predictor set plus physical tests.

**Figure 3.** Permutation importance for future attainment of at least 900 points. Importance is expressed as the mean decrease in held-out ROC-AUC after permutation.

**Figure 4.** Paired incremental change in R2 after adding physical tests within each fixed algorithm. Error bars are 2,000-replicate athlete-cluster bootstrap 95% percentile intervals.

**Figure 5.** Out-of-fold calibration curves for future scores of at least 900 and 950 points, comparing models without and with physical tests.
