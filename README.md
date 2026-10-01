---
license: cc-by-4.0
tags: [tabular-classification, xgboost, teaching]
---
# Study-office drop-out model (teaching)

Predicts the probability that a first-year student who is still enrolled at the end of **week 6** leaves later in
the first semester, from what the study office knows at week 6.

- **Data:** synthetic students resampled from UCI *Predict students' dropout and academic success* (CC BY 4.0).
- **Model:** scikit-learn preprocessing (median fill, scaling, one-hot) + XGBoost (600 trees, depth 2).
- **Split:** by cohort. Train 2023-2024, validation 2025.
- **Validation:** AUC 0.802; top-40 rule: precision 35%, recall 22%.
- **Removed as leakage:** ects_passed_sem1, deregistration_form_opened, last_login_week (recorded after week 6).
- **Limits:** recall is lower for international students (few leavers, logins mean less for them). Risks are
  associations, not causes. Use only to **offer** help, decided by an adviser; never to decide anything about a student.
