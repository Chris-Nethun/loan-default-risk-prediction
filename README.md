##  Loan Default Risk Prediction

Application-stage credit risk model using Random Forest.

##  Project Overview

This project builds a machine learning model to identify high-risk loan applicants at the application stage. The goal is to detect potential defaulters before loan approval, enabling proactive credit risk management.

The model was trained on 45,000 loan records containing applicant demographics, financial indicators, and credit history information.

## Business Objective

To improve early-stage default detection by:

- Maximizing recall for defaulters

- Evaluating precision–recall trade-offs

- Optimizing classification threshold for risk-based decision making

##  Methodology

- 80/20 stratified train-test split

- One-hot encoding for categorical variables

- Random Forest Classifier (400 trees)

- Class imbalance handled using class_weight="balanced"

- Threshold tuning for business optimization

##  Model Performance
Metric	Value
ROC-AUC	0.856
PR-AUC	0.760

At default threshold (0.50):

- Recall (defaulters): 58%

After threshold optimization (0.35):

- Recall improved to 78%

Significant reduction in undetected defaulters

##  Visual Results
- ROC Curve

- Precision-Recall Curve

Feature Importance

##  Key Insights

- Credit score and loan-to-income ratio are major drivers of default risk.

- Lowering the threshold significantly improves detection of risky applicants.

- In imbalanced credit risk problems, optimizing recall is more important than maximizing accuracy.

##  Tech Stack

- Python

- Pandas

- Scikit-learn

- Matplotlib

- Seaborn

##  Repository Structure

```
loan-default-risk-prediction/
│
├── README.md
├── loan_default_model.py
├── requirements.txt
└── images/
    ├── roc_curve.png
    └── feature_importance.png
```


## 🚀 Future Improvements

- Compare with XGBoost

- Apply SHAP for model interpretability

- Perform cost-sensitive learning analysis
