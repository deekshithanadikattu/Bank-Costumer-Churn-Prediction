# Bank Customer Churn: Analysis and Prediction

Which bank customers are about to leave, and what should the bank do about it? This project
explores churn across 10,000 customers, compares three classifiers, and ranks customers by churn
risk so a retention team knows whom to contact first.

## Key results (held-out test set: 2,000 customers)
- **The top 20% riskiest customers contain 63% of all churners**, 3.1× better than contacting
  customers at random. In that group, 64% of customers actually churn, against 20% overall.
- **XGBoost** was the best of three models on 5-fold cross-validation (PR-AUC 0.70, vs 0.68 for
  Random Forest and 0.46 for Logistic Regression).
- On the test set it identifies **75% of churners** (recall) with **ROC-AUC 0.87** and **PR-AUC 0.72**.

![Cumulative gains](figures/cumulative_gains.png)

## What drives churn
![Churn by segment](figures/churn_by_segment.png)

| Finding | Evidence |
|---|---|
| Customers with 3–4 products almost always leave | 83% churn with 3 products and 100% with 4, vs 8% with 2 |
| Churn peaks in middle age | 56% churn at ages 50–59 and 31% at 40–49, vs 8% for 18–29. Ages 40–59 are 35% of customers but 64% of churners |
| Inactive members leave twice as often | 27% churn vs 14% for active members. Inactive members are 64% of all churners |
| Germany is a problem market | 32% churn vs 16–17% in France and Spain. Germany is 25% of customers but 40% of churners |

Number of products and age are also the model's two most important features:

![Feature importance](figures/feature_importance.png)

## Recommended retention actions
1. **Audit 3+ product bundles.** Nearly every customer with 3–4 products leaves, which suggests
   these bundles are mis-sold or poorly priced. Stop pushing a third product until the cause is
   found, and steer cross-selling toward the 2-product sweet spot, which has the lowest churn.
2. **Re-engage inactive members.** Trigger outreach (offers, check-in calls) when a customer goes
   inactive. That group produces almost two-thirds of churners.
3. **Give 40–59-year-olds a relationship offer.** This age group is the core of churn. Wealth or
   retirement planning and a named relationship manager fit their life stage.
4. **Investigate Germany.** Twice the churn of other markets points to a local cause (competitor
   pricing or service quality) worth a dedicated review.

`at_risk_customers.csv` lists every test-set customer with a churn-risk score and a risk tier
(High = top 20%, Medium, Low), ready for a retention team or a Tableau dashboard.

## Method
- **Split once, stratified:** 80% train and 20% test. The test set is touched only at the end.
- **Pipelines:** scaling and one-hot encoding are fit on training data only.
- **Imbalance:** handled with class weights instead of oversampling, so no duplicated customers
  leak between train and test.
- **Model selection:** 5-fold stratified cross-validation on the training set, ranked by PR-AUC,
  which suits a 20% minority class better than accuracy.
- **Explainability:** permutation importance on the test set.

![Model comparison](figures/model_comparison.png)

### Why accuracy isn't the headline
20% of customers churn, so predicting "stays" for everyone already scores 79.7% accuracy. The model
scores 81.0% accuracy, but that number hides what matters: it catches 75% of churners, where the
"always stays" approach catches none. Class weighting deliberately trades some accuracy for recall,
because missing a churner costs more than an unneeded retention call.

### Fixes over the first version
An earlier notebook reported 98% accuracy. It oversampled churners *before* splitting, so copies of
the same customers landed in both train and test, and the model memorised them. It also used
`customer_id` as a feature and re-fit the scaler on the test set. This version fixes all three.

## Run it
```bash
pip install pandas numpy scikit-learn xgboost matplotlib
python churn_analysis.py
```
Put `Bank Customer Churn Prediction.csv` in the same folder first. The dataset is
[Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
on Kaggle. Outputs: `figures/`, `at_risk_customers.csv`, `results.json`.
