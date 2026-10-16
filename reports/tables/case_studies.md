# Case studies (temporal test set)

## healthy_correct
- Repo: nodejs_node, prediction month 2026-01
- True: Healthy, predicted: Healthy
- Top drivers:
  - commits_cv6 = 0.20 (SHAP +0.150)
  - contributors_mean3 = 40.00 (SHAP +0.141)
  - commits_last = 168.00 (SHAP +0.095)
  - contributors_last = 42.00 (SHAP +0.087)
  - contributors_mean6 = 45.00 (SHAP +0.058)

## abandoned_correct
- Repo: rethinkdb_rethinkdb, prediction month 2024-03
- True: Abandoned, predicted: Abandoned
- Top drivers:
  - contributors_mean3 = 0.00 (SHAP +0.099)
  - age_months_log = 5.16 (SHAP -0.065)
  - commits_last = 0.00 (SHAP +0.051)
  - contributors_mean6 = 0.33 (SHAP +0.047)
  - commits_cv6 = 1.44 (SHAP +0.047)

## false_alarm
- Repo: sherlock-project_sherlock, prediction month 2025-01
- True: Healthy, predicted: Declining
- Top drivers:
  - prs_opened_last = 4.00 (SHAP +0.029)
  - prs_opened_mean3 = 5.67 (SHAP +0.022)
  - prs_opened_mean6 = 11.33 (SHAP +0.022)
  - commit_zero_months6 = 0.00 (SHAP +0.018)
  - commits_cv6 = 0.63 (SHAP +0.018)
