# 🔮 ChurnLens — Customer Churn Prediction System

> **Task ID:** ML-2 &nbsp;|&nbsp; **Domain:** Machine Learning &nbsp;|&nbsp; **Batch:** Teyzix Core Internship · June 2026  
> **Assigned:** 20 June 2026 &nbsp;|&nbsp; **Deadline:** 29 June 2026

---

## Overview

ChurnLens is an end-to-end machine learning system that predicts whether a customer is likely to stop using a service, based on their historical behaviour and engagement patterns. It includes a full data pipeline, multiple trained models, explainability via SHAP, customer segmentation via K-Means, and an interactive Streamlit dashboard for real-time predictions.

---

## Project Structure

```
Task-02/
│
├── customer_churn_dataset.csv     # Self-created dataset (1,500+ records)
│
├── churn_prediction_ML2.py        # Full ML pipeline (training script)
├── app.py                         # Streamlit dashboard
│
├── churn_model.pkl                # Saved Random Forest model
├── scaler.pkl                     # Saved StandardScaler
├── shap_explainer.pkl             # Saved SHAP TreeExplainer
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── Report_ML2.docx                # Full analysis report
```

---

## Dataset

- **Records:** 1,500+
- **Target Variable:** `ChurnStatus` (0 = Retained, 1 = Churned)
- **Features include:**

| Feature | Type | Description |
|---|---|---|
| CustomerID | ID | Unique customer identifier |
| Age | Numerical | Customer age |
| Gender | Categorical | Male / Female |
| City | Categorical | Customer city |
| SubscriptionType | Categorical | Basic / Standard / Premium |
| MonthlySpending | Numerical | Average monthly spend ($) |
| Tenure | Numerical | Months as a customer |
| NumberOfPurchases | Numerical | Total purchases made |
| CustomerSupportRequests | Numerical | Number of support tickets |
| LoginFrequency | Numerical | Logins per month |
| LastActivityDate | Date | Most recent activity date |
| SatisfactionScore | Numerical | Score from 1–10 |
| PaymentMethod | Categorical | Credit Card / Debit Card / PayPal / Bank Transfer |
| DeviceType | Categorical | Mobile / Desktop / Tablet |
| AutoRenew | Categorical | Yes / No |
| ChurnStatus | Target | 0 = Retained, 1 = Churned |

**Engineered Features (added in pipeline):**

| Feature | Formula |
|---|---|
| DaysSinceLastActivity | `today − LastActivityDate` |
| SpendingPerLogin | `MonthlySpending / (LoginFrequency + 1)` |
| SupportPerTenure | `CustomerSupportRequests / (Tenure + 1)` |
| PurchasePerMonth | `NumberOfPurchases / (Tenure + 1)` |

---

## Machine Learning Pipeline

### Data Preparation
- Fill missing values with column medians
- Remove duplicate rows
- Label encode all categorical columns
- Feature engineering (date conversion + 3 derived ratios)
- Train/test split: **80% / 20%** (stratified)
- Feature scaling: **StandardScaler**

### Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Non-linear, interpretable |
| Random Forest | Best performer — 100 trees |
| Gradient Boosting | Sequential ensemble |

### Hyperparameter Tuning
GridSearchCV (3-fold CV) on Random Forest across:
- `n_estimators`: [50, 100, 150]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 5]

### Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion Matrix
- ROC Curves (all models overlaid)
- 5-Fold Cross-Validation

---

## Bonus Features

| Feature | Description |
|---|---|
| ✅ Hyperparameter Optimization | GridSearchCV tunes the best RF config |
| ✅ Feature Importance Analysis | Bar chart with top features highlighted |
| ✅ Cross-Validation | 5-fold CV scores for all 4 models |
| ✅ SHAP Explainability (XAI) | TreeExplainer bar + beeswarm plots |
| ✅ Customer Segmentation | K-Means (k=4) with elbow method |
| ✅ Churn Risk Dashboard | 6-panel matplotlib dashboard |
| ✅ Multiple Model Comparison | Accuracy / F1 / ROC-AUC bar charts |

---

## Streamlit App

The interactive dashboard has three pages:

**🔮 Predict**
Enter any customer's details and get an instant churn probability, risk band (High / Medium / Low), probability breakdown chart, feature importance chart, and tailored business recommendations.

**📊 EDA & Insights**
Explore the dataset through distribution plots, demographic breakdowns, correlation heatmap, and churn driver box plots — all in a dark themed 4-tab layout.

**🧠 Model Info**
View model architecture, pipeline steps, feature importance table, evaluation metrics on the held-out test set, and the confusion matrix.

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Run the training script first to generate the `.pkl` files:

```bash
python churn_prediction_ML2.py
```

This will:
- Load and clean the dataset
- Run EDA and generate all plots
- Train 4 models, tune the best one, and run SHAP + K-Means
- Save `churn_model.pkl`, `scaler.pkl`, `shap_explainer.pkl`

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

> **Note:** `churn_model.pkl`, `scaler.pkl`, and `customer_churn_dataset.csv` must be in the same directory as `app.py`.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| ML | Scikit-learn |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Serialisation | Pickle |

---

## Evaluation Criteria Coverage

| Criterion | Weight | Coverage |
|---|---|---|
| Dataset Quality & Realism | 15% | 1,500+ records, 15+ features, realistic distributions |
| Data Preparation & Feature Engineering | 20% | Missing values, encoding, scaling, 4 engineered features |
| Model Performance & Evaluation | 30% | 4 models, tuning, SHAP, CV, ROC curves |
| Problem-Solving Approach | 20% | Segmentation, risk banding, business recommendations |
| Documentation & Presentation | 15% | README, report, Streamlit dashboard |

---

## Author

**Sara Ahmad**  
Computer Science Student · Virtual University of Pakistan  
Teyzix Core Internship · June 2026 Batch