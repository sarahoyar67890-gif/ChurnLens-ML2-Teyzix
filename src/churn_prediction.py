# =====================================================
#         Customer Churn Prediction System
#                    Task : ML-2
#             Teyzix Core Internship 2026
# =====================================================
#   Intern    : Sara Ahmad
#   Domain    : Machine Learning
#   Deadline  : 29th June 2026
# =====================================================

# ┌─────────────────────────────────────────────────┐
# │              IMPORT LIBRARIES                   │
# └─────────────────────────────────────────────────┘
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

import shap
import pickle
import warnings
warnings.filterwarnings("ignore")


# ── Aesthetic Plot Settings ──────────────────────────

plt.rcParams.update({
    "figure.facecolor"  : "#0f0f1a",
    "axes.facecolor"    : "#161625",
    "axes.edgecolor"    : "#2e2e4a",
    "axes.labelcolor"   : "#c9c9e3",
    "axes.titlecolor"   : "#e0e0ff",
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "axes.grid"         : True,
    "grid.color"        : "#2a2a40",
    "grid.linewidth"    : 0.6,
    "text.color"        : "#c9c9e3",
    "xtick.color"       : "#9090b8",
    "ytick.color"       : "#9090b8",
    "legend.facecolor"  : "#1a1a2e",
    "legend.edgecolor"  : "#2e2e4a",
    "font.family"       : "DejaVu Sans",
    "figure.dpi"        : 120,
})

PALETTE   = ["#7c83fd", "#fd7cba", "#7cfdd9", "#fdcd7c", "#fd7c7c"]
ACCENT    = "#7c83fd"
ACCENT2   = "#fd7cba"
DARK_BG   = "#0f0f1a"
CARD_BG   = "#161625"


def section(title):
    """ Print a styled section header to the console. """
    bar = "─" * 56
    print(f"\n┌{bar}┐")
    print(f"│  {title:<54}│")
    print(f"└{bar}┘")


# =====================================================
#  PART 1 :  DATA LOADING, CLEANING & EDA
# =====================================================

# ┌─────────────────────────────────────────────────┐
# │  STEP 1 : Load Dataset                          │
# └─────────────────────────────────────────────────┘

section("STEP 1 : Load Dataset")

df = pd.read_csv("customer_churn_dataset.csv")

print("\n  First Five Rows\n")
print(df.head())
print(f"\n  Shape  : {df.shape[0]} rows × {df.shape[1]} columns")


# ┌─────────────────────────────────────────────────┐
# │  STEP 2 : Basic Exploration                     │
# └─────────────────────────────────────────────────┘

section("STEP 2 : Basic Exploration")

print("\n  Dataset Info\n")
df.info()

print("\n  Statistical Summary\n")
print(df.describe())


# ┌─────────────────────────────────────────────────┐
# │  STEP 3 : Missing Values                        │
# └─────────────────────────────────────────────────┘

section("STEP 3 : Missing Values")

missing = df.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    print("\n  ✓  No missing values found.")
else:
    print("\n  Missing Values Detected :\n")
    print(missing)

    df["MonthlySpending"].fillna(df["MonthlySpending"].median(),   inplace=True)
    df["LoginFrequency"].fillna(df["LoginFrequency"].median(),     inplace=True)
    df["SatisfactionScore"].fillna(df["SatisfactionScore"].median(), inplace=True)

    print("\n  ✓  Filled with column medians.")


# ┌─────────────────────────────────────────────────┐
# │  STEP 4 : Remove Duplicates                     │
# └─────────────────────────────────────────────────┘

section("STEP 4 : Remove Duplicates")

print(f"\n  Before : {df.shape[0]} rows")
df.drop_duplicates(inplace=True)
print(f"  After  : {df.shape[0]} rows")


# ┌─────────────────────────────────────────────────┐
# │  STEP 5 : Feature Engineering                   │
# └─────────────────────────────────────────────────┘

section("STEP 5 : Feature Engineering")

# Convert date & derive days-since feature
df["LastActivityDate"]      = pd.to_datetime(df["LastActivityDate"])
today                       = pd.Timestamp.today()
df["DaysSinceLastActivity"] = (today - df["LastActivityDate"]).dt.days
df.drop("LastActivityDate", axis=1, inplace=True)

# Derived behavioural ratios
df["SpendingPerLogin"]   = df["MonthlySpending"]  / (df["LoginFrequency"]  + 1)
df["SupportPerTenure"]   = df["CustomerSupportRequests"] / (df["Tenure"]   + 1)
df["PurchasePerMonth"]   = df["NumberOfPurchases"] / (df["Tenure"]         + 1)

print("\n  New Features Created :")
print("   • DaysSinceLastActivity")
print("   • SpendingPerLogin")
print("   • SupportPerTenure")
print("   • PurchasePerMonth")


# ┌─────────────────────────────────────────────────┐
# │  STEP 6 : Encoding & Scaling                    │
# └─────────────────────────────────────────────────┘

section("STEP 6 : Label Encoding")

encoder = LabelEncoder()

categorical_columns = [
    "Gender", "City", "SubscriptionType",
    "PaymentMethod", "DeviceType", "AutoRenew", "ChurnStatus"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

df.drop("CustomerID", axis=1, inplace=True)

print("\n  ✓  Categorical columns encoded.")
print("\n  Preview After Encoding :\n")
print(df.head())


# =====================================================
#  EXPLORATORY DATA ANALYSIS
# =====================================================

section("EDA : Churn Distribution")

fig, axes = plt.subplots(1, 2, figsize=(8,4))
fig.patch.set_facecolor(DARK_BG)

# Count plot
churn_counts = df["ChurnStatus"].value_counts()
labels       = ["Not Churned", "Churned"]
colors       = [ACCENT, ACCENT2]

axes[0].bar(labels, churn_counts.values, color=colors, width=0.5, edgecolor="none")
axes[0].set_title("Churn Distribution")
axes[0].set_ylabel("Count")

# Pie chart
axes[1].pie(
    churn_counts.values,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": DARK_BG, "linewidth": 2}
)
axes[1].set_title("Churn Ratio")

plt.tight_layout()
plt.show()


# ── Demographics ─────────────────────────────────────

section("EDA : Demographics")

fig, axes = plt.subplots(1, 3, figsize=(8,4))
fig.patch.set_facecolor(DARK_BG)

sns.countplot(x="Gender",           data=df, ax=axes[0], palette=PALETTE, edgecolor="none")
sns.countplot(x="SubscriptionType", data=df, ax=axes[1], palette=PALETTE, edgecolor="none")
sns.countplot(x="DeviceType",       data=df, ax=axes[2], palette=PALETTE, edgecolor="none")

axes[0].set_title("Gender Distribution")
axes[1].set_title("Subscription Types")
axes[2].set_title("Device Types")

for ax in axes:
    ax.set_xlabel("")

plt.tight_layout()
plt.show()


# ── Spending & Activity ──────────────────────────────

section("EDA : Spending & Activity Distributions")

fig, axes = plt.subplots(1, 2, figsize=(8,4))
fig.patch.set_facecolor(DARK_BG)

sns.histplot(df["MonthlySpending"],       bins=30, ax=axes[0], color=ACCENT,  kde=True)
sns.histplot(df["DaysSinceLastActivity"], bins=30, ax=axes[1], color=ACCENT2, kde=True)

axes[0].set_title("Monthly Spending Distribution")
axes[1].set_title("Days Since Last Activity")

plt.tight_layout()
plt.show()


# ── Churn vs Key Features ────────────────────────────

section("EDA : Churn vs Key Features")

features = [
    "SatisfactionScore", "LoginFrequency",
    "MonthlySpending",   "DaysSinceLastActivity"
]

fig, axes = plt.subplots(1, 4, figsize=(8,5))
fig.patch.set_facecolor(DARK_BG)

for ax, feat in zip(axes, features):
    sns.boxplot(
        x="ChurnStatus", y=feat, data=df,
        ax=ax, palette=[ACCENT, ACCENT2],
        flierprops={"marker": ".", "color": "#555580"}
    )
    ax.set_title(feat.replace("_", " "))
    ax.set_xlabel("Churn Status")

plt.tight_layout()
plt.show()


# ── Correlation Heatmap ──────────────────────────────

section("EDA : Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(DARK_BG)

cmap = sns.diverging_palette(260, 330, s=80, l=55, as_cmap=True)

sns.heatmap(
    df.corr(), annot=True, fmt=".2f",
    cmap=cmap, ax=ax, linewidths=0.4,
    linecolor="#0f0f1a", annot_kws={"size": 8}
)

ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


# ── Customer Behaviour Scatter ───────────────────────

section("EDA : Customer Behaviour Scatter")

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(DARK_BG)

for churn_val, label, color in zip([0, 1], ["Retained", "Churned"], [ACCENT, ACCENT2]):
    subset = df[df["ChurnStatus"] == churn_val]
    ax.scatter(
        subset["LoginFrequency"],
        subset["MonthlySpending"],
        label=label, color=color,
        alpha=0.5, s=20, edgecolors="none"
    )

ax.set_xlabel("Login Frequency")
ax.set_ylabel("Monthly Spending")
ax.set_title("Customer Behaviour : Login vs Spending")
ax.legend()

plt.tight_layout()
plt.show()


# =====================================================
#  PART 2 : MACHINE LEARNING MODELS
# =====================================================

# ┌─────────────────────────────────────────────────┐
# │  STEP 7 : Train / Test Split                    │
# └─────────────────────────────────────────────────┘

section("STEP 7 : Train / Test Split")

X = df.drop("ChurnStatus", axis=1)
y = df["ChurnStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train size : {X_train.shape[0]} samples")
print(f"  Test  size : {X_test.shape[0]} samples")


# ┌─────────────────────────────────────────────────┐
# │  STEP 8 : Feature Scaling                       │
# └─────────────────────────────────────────────────┘

section("STEP 8 : Feature Scaling")

scaler  = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n  ✓  StandardScaler applied.")


# ┌─────────────────────────────────────────────────┐
# │  STEP 9 : Train Models                          │
# └─────────────────────────────────────────────────┘

section("STEP 9 : Model Training")

# ── Logistic Regression ──────────────────────────────
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print("  ✓  Logistic Regression trained.")

# ── Decision Tree ────────────────────────────────────
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
dt_pred = dt.predict(X_test_scaled)
print("  ✓  Decision Tree trained.")

# ── Random Forest ────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
print("  ✓  Random Forest trained.")

# ── Gradient Boosting ────────────────────────────────
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)
print("  ✓  Gradient Boosting trained.")


# ┌─────────────────────────────────────────────────┐
# │  STEP 10 : Model Evaluation                     │
# └─────────────────────────────────────────────────┘

section("STEP 10 : Model Evaluation")


def evaluate_model(name, y_true, y_pred):
    print(f"\n  ── {name} ──")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"  ROC AUC   : {roc_auc_score(y_true, y_pred):.4f}")
    print(f"\n{classification_report(y_true, y_pred)}")  

evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("Decision Tree",       y_test, dt_pred)
evaluate_model("Random Forest",       y_test, rf_pred)
evaluate_model("Gradient Boosting",   y_test, gb_pred)


# ┌─────────────────────────────────────────────────┐
# │  STEP 11 : Model Comparison Chart               │
# └─────────────────────────────────────────────────┘

section("STEP 11 : Model Comparison")

model_names  = ["Logistic\nRegression", "Decision\nTree", "Random\nForest", "Gradient\nBoosting"]
predictions  = [lr_pred, dt_pred, rf_pred, gb_pred]

metrics = {
    "Accuracy"  : [accuracy_score(y_test,  p)  for p in predictions],
    "F1 Score"  : [f1_score(y_test,        p)  for p in predictions],
    "ROC AUC"   : [roc_auc_score(y_test,   p)  for p in predictions],
}

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Model Comparison", fontsize=16, fontweight="bold", color="#e0e0ff", y=1.02)

for ax, (metric, values) in zip(axes, metrics.items()):
    bars = ax.bar(model_names, values, color=PALETTE[:4], edgecolor="none", width=0.5)
    ax.set_title(metric)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom",
            fontsize=9, color="#e0e0ff"
        )

plt.tight_layout()
plt.show()


# ┌─────────────────────────────────────────────────┐
# │  STEP 12 : ROC Curves                           │
# └─────────────────────────────────────────────────┘

section("STEP 12 : ROC Curves")

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(DARK_BG)

roc_models = [
    ("Logistic Regression", lr,  PALETTE[0]),
    ("Decision Tree",       dt,  PALETTE[1]),
    ("Random Forest",       rf,  PALETTE[2]),
    ("Gradient Boosting",   gb,  PALETTE[3]),
]

for name, model, color in roc_models:
    proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.3f})", color=color, linewidth=2)

ax.plot([0, 1], [0, 1], linestyle="--", color="#555580", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(loc="lower right")

plt.tight_layout()
plt.show()


# ┌─────────────────────────────────────────────────┐
# │  STEP 13 : Confusion Matrix (Best Model)        │
# └─────────────────────────────────────────────────┘

section("STEP 13 : Confusion Matrix — Random Forest")

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(DARK_BG)

cm = confusion_matrix(y_test, rf_pred)

sns.heatmap(
    cm, annot=True, fmt="d",
    cmap=sns.light_palette(ACCENT, as_cmap=True),
    ax=ax, linewidths=1, linecolor=DARK_BG,
    annot_kws={"size": 14, "weight": "bold"}
)

ax.set_title("Confusion Matrix — Random Forest")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_xticklabels(["Not Churned", "Churned"])
ax.set_yticklabels(["Not Churned", "Churned"], rotation=0)

plt.tight_layout()
plt.show()


# =====================================================
#  BONUS FEATURES
# =====================================================

# ┌─────────────────────────────────────────────────┐
# │  BONUS 1 : Hyperparameter Optimization          │
# └─────────────────────────────────────────────────┘

section("BONUS 1 : Hyperparameter Optimization — Random Forest")

param_grid = {
    "n_estimators"     : [50, 100, 150],
    "max_depth"        : [None, 10, 20],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_scaled, y_train)

best_rf      = grid_search.best_estimator_
best_rf_pred = best_rf.predict(X_test_scaled)

print(f"\n  Best Parameters  : {grid_search.best_params_}")
print(f"  Best CV F1 Score : {grid_search.best_score_:.4f}")
print(f"  Test Accuracy    : {accuracy_score(y_test, best_rf_pred):.4f}")
print(f"  Test F1 Score    : {f1_score(y_test, best_rf_pred):.4f}")


# ┌─────────────────────────────────────────────────┐
# │  BONUS 2 : Feature Importance Analysis          │
# └─────────────────────────────────────────────────┘

section("BONUS 2 : Feature Importance Analysis")

importance_df = pd.DataFrame({
    "Feature"   : X.columns,
    "Importance": best_rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n  Top Features :\n")
print(importance_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(DARK_BG)

colors = [
    ACCENT if i < 3 else PALETTE[2] if i < 7 else "#555580"
    for i in range(len(importance_df))
]

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color=colors, edgecolor="none"
)

ax.invert_yaxis()
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance — Tuned Random Forest")

for i, (feat, score) in enumerate(
    zip(importance_df["Feature"], importance_df["Importance"])
):
    ax.text(score + 0.001, i, f"{score:.4f}", va="center",
            fontsize=8, color="#c9c9e3")

plt.tight_layout()
plt.show()


# ┌─────────────────────────────────────────────────┐
# │  BONUS 3 : Cross-Validation                     │
# └─────────────────────────────────────────────────┘

section("BONUS 3 : Cross-Validation (5-Fold)")

cv_models = {
    "Logistic Regression": lr,
    "Decision Tree"      : dt,
    "Random Forest"      : best_rf,
    "Gradient Boosting"  : gb,
}

print(f"\n  {'Model':<25} {'Mean F1':>10}  {'Std':>8}")
print(f"  {'─'*25} {'─'*10}  {'─'*8}")

cv_means  = []
cv_labels = []

for name, model in cv_models.items():
    scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=5, scoring="f1"
    )
    cv_means.append(scores.mean())
    cv_labels.append(name)
    print(f"  {name:<25} {scores.mean():>10.4f}  ±{scores.std():.4f}")

# Visualise CV scores
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(DARK_BG)

bars = ax.bar(
    [n.replace(" ", "\n") for n in cv_labels],
    cv_means, color=PALETTE[:4],
    edgecolor="none", width=0.5
)
ax.set_title("5-Fold Cross-Validation F1 Scores")
ax.set_ylabel("Mean F1 Score")
ax.set_ylim(0, 1.05)

for bar, val in zip(bars, cv_means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:.3f}", ha="center", va="bottom",
        fontsize=9, color="#e0e0ff"
    )

plt.tight_layout()
plt.show()


# ┌─────────────────────────────────────────────────┐
# │  BONUS 4 : SHAP — Explainable AI                │
# └─────────────────────────────────────────────────┘

section("BONUS 4 : SHAP — Explainable AI")

explainer   = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test_scaled)

# Summary plot (bar)
print("\n  Generating SHAP Summary Plot ...")

shap_values = explainer.shap_values(X_test_scaled)

# Handle both old and new SHAP output formats
sv = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

shap.summary_plot(
    sv,
    X_test_scaled,
    feature_names=X.columns.tolist(),
    plot_type="bar",
    show=True,
    max_display=12
)

shap.summary_plot(
    sv,
    X_test_scaled,
    feature_names=X.columns.tolist(),
    show=True,
    max_display=12
)

print("\n  ✓  SHAP Analysis Complete.")


# ┌─────────────────────────────────────────────────┐
# │  BONUS 5 : Customer Segmentation (K-Means)      │
# └─────────────────────────────────────────────────┘

section("BONUS 5 : Customer Segmentation — K-Means")

# Find optimal k using inertia (Elbow Method)
inertias = []
K_range  = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(DARK_BG)

ax.plot(K_range, inertias, marker="o", color=ACCENT, linewidth=2, markersize=7)
ax.set_title("Elbow Method — Optimal Number of Clusters")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")

plt.tight_layout()
plt.show()

# Fit with k = 4
K_BEST = 4

kmeans = KMeans(n_clusters=K_BEST, random_state=42, n_init=10)
df["Segment"] = kmeans.fit_predict(scaler.transform(X.values))

segment_profiles = df.groupby("Segment").agg({
    "MonthlySpending"  : "mean",
    "LoginFrequency"   : "mean",
    "SatisfactionScore": "mean",
    "Tenure"           : "mean",
    "ChurnStatus"      : "mean",
}).round(2)

segment_profiles.columns = [
    "Avg Spending", "Avg Logins",
    "Avg Satisfaction", "Avg Tenure", "Churn Rate"
]

print("\n  Customer Segment Profiles :\n")
print(segment_profiles.to_string())

# Segment scatter
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK_BG)

for seg in range(K_BEST):
    subset = df[df["Segment"] == seg]
    ax.scatter(
        subset["LoginFrequency"],
        subset["MonthlySpending"],
        label=f"Segment {seg}",
        color=PALETTE[seg], alpha=0.5,
        s=18, edgecolors="none"
    )

ax.set_xlabel("Login Frequency")
ax.set_ylabel("Monthly Spending")
ax.set_title(f"Customer Segments — K-Means (k={K_BEST})")
ax.legend()

plt.tight_layout()
plt.show()

# Drop temp column
df.drop("Segment", axis=1, inplace=True)


# ┌─────────────────────────────────────────────────┐
# │  BONUS 6 : Churn Risk Dashboard                 │
# └─────────────────────────────────────────────────┘

section("BONUS 6 : Churn Risk Dashboard")

churn_prob = best_rf.predict_proba(X_test_scaled)[:, 1]

risk_df = X_test.copy()
risk_df["ChurnProbability"] = churn_prob
risk_df["ActualChurn"]      = y_test.values
risk_df["RiskBand"] = pd.cut(
    churn_prob,
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

fig = plt.figure(figsize=(8,5))
fig.patch.set_facecolor(DARK_BG)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1 : Risk Band Distribution ────────────────
ax1 = fig.add_subplot(gs[0, 0])
band_counts = risk_df["RiskBand"].value_counts()
band_colors = [PALETTE[2], PALETTE[3], PALETTE[4]]
ax1.bar(band_counts.index, band_counts.values,
        color=band_colors, edgecolor="none")
ax1.set_title("Customers by Risk Band")
ax1.set_ylabel("Count")

# ── Panel 2 : Churn Probability Histogram ───────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(churn_prob, bins=30, color=ACCENT, edgecolor="none", alpha=0.85)
ax2.set_title("Churn Probability Distribution")
ax2.set_xlabel("Probability")
ax2.set_ylabel("Count")

# ── Panel 3 : High-Risk Spending vs Logins ──────────
ax3 = fig.add_subplot(gs[0, 2])
high_risk = risk_df[risk_df["RiskBand"] == "High Risk"]
low_risk  = risk_df[risk_df["RiskBand"] == "Low Risk"]
ax3.scatter(low_risk["LoginFrequency"],  low_risk["MonthlySpending"],
            color=PALETTE[2], alpha=0.4, s=12, label="Low Risk",  edgecolors="none")
ax3.scatter(high_risk["LoginFrequency"], high_risk["MonthlySpending"],
            color=PALETTE[4], alpha=0.6, s=15, label="High Risk", edgecolors="none")
ax3.set_title("Risk : Login vs Spending")
ax3.set_xlabel("Login Frequency")
ax3.set_ylabel("Monthly Spending")
ax3.legend(fontsize=8)

# ── Panel 4 : Satisfaction by Risk Band ─────────────
ax4 = fig.add_subplot(gs[1, 0])
for band, color in zip(["Low Risk", "Medium Risk", "High Risk"], band_colors):
    vals = risk_df[risk_df["RiskBand"] == band]["SatisfactionScore"]
    ax4.hist(vals, bins=15, alpha=0.7, label=band, color=color, edgecolor="none")
ax4.set_title("Satisfaction Score by Risk Band")
ax4.set_xlabel("Satisfaction Score")
ax4.legend(fontsize=8)

# ── Panel 5 : Tenure by Risk Band ───────────────────
ax5 = fig.add_subplot(gs[1, 1])
risk_df.boxplot(
    column="Tenure", by="RiskBand", ax=ax5,
    patch_artist=True,
    boxprops=dict(facecolor=CARD_BG, color=ACCENT),
    whiskerprops=dict(color=ACCENT),
    capprops=dict(color=ACCENT),
    medianprops=dict(color=ACCENT2, linewidth=2),
    flierprops=dict(marker=".", color="#555580")
)
ax5.set_title("Tenure by Risk Band")
ax5.set_xlabel("Risk Band")
plt.sca(ax5)
plt.title("Tenure by Risk Band")

# ── Panel 6 : KPI Text ──────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(CARD_BG)
ax6.axis("off")

total      = len(risk_df)
high_count = (risk_df["RiskBand"] == "High Risk").sum()
avg_prob   = churn_prob.mean()
best_acc   = accuracy_score(y_test, best_rf_pred)

kpis = [
    ("Total Customers",    f"{total}"),
    ("High Risk Count",    f"{high_count}  ({high_count/total*100:.1f}%)"),
    ("Avg Churn Probability", f"{avg_prob:.3f}"),
    ("Model Accuracy",     f"{best_acc:.4f}"),
]

ax6.text(0.5, 0.92, "Dashboard KPIs", ha="center", va="top",
         fontsize=13, fontweight="bold", color="#e0e0ff",
         transform=ax6.transAxes)

for i, (label, value) in enumerate(kpis):
    y_pos = 0.72 - i * 0.18
    ax6.text(0.5, y_pos,       label, ha="center", fontsize=9,
             color="#9090b8", transform=ax6.transAxes)
    ax6.text(0.5, y_pos - 0.06, value, ha="center", fontsize=14,
             fontweight="bold", color=PALETTE[i % len(PALETTE)],
             transform=ax6.transAxes)

fig.suptitle(
    "Churn Risk Dashboard",
    fontsize=18, fontweight="bold", color="#e0e0ff", y=1.01
)

plt.tight_layout()
plt.show()

print("\n  ✓  Dashboard Rendered.")


# =====================================================
#  SAVE MODELS & WRAP UP
# =====================================================

# ┌─────────────────────────────────────────────────┐
# │  Save Trained Artifacts                         │
# └─────────────────────────────────────────────────┘

section("Save Model Artifacts")

pickle.dump(best_rf,  open("churn_model.pkl",   "wb"))
pickle.dump(scaler,   open("scaler.pkl",         "wb"))
pickle.dump(explainer, open("shap_explainer.pkl","wb"))

print("\n  ✓  churn_model.pkl    saved")
print("  ✓  scaler.pkl         saved")
print("  ✓  shap_explainer.pkl saved")


# ┌─────────────────────────────────────────────────┐
# │  Example Prediction (CLI Interface)             │
# └─────────────────────────────────────────────────┘

section("Example Prediction")

# Build a sample input dict matching column order
feature_names  = X.columns.tolist()
sample_values  = {
    "Age"                     : 35,
    "Gender"                  : 1,
    "City"                    : 2,
    "SubscriptionType"        : 0,
    "MonthlySpending"         : 25,
    "Tenure"                  : 5,
    "NumberOfPurchases"       : 8,
    "CustomerSupportRequests" : 7,
    "LoginFrequency"          : 2,
    "SatisfactionScore"       : 2,
    "PaymentMethod"           : 3,
    "DeviceType"              : 1,
    "AutoRenew"               : 0,
    "DaysSinceLastActivity"   : 120,
    "SpendingPerLogin"        : 25 / (2 + 1),
    "SupportPerTenure"        : 7  / (5 + 1),
    "PurchasePerMonth"        : 8  / (5 + 1),
}

# Align to training column order
sample_row = np.array([[sample_values[f] for f in feature_names]])
sample_scaled = scaler.transform(sample_row)

prediction  = best_rf.predict(sample_scaled)[0]
probability = best_rf.predict_proba(sample_scaled)[0]

print(f"\n  Churn Prediction  : {'⚠  LIKELY TO CHURN' if prediction == 1 else '✓  NOT LIKELY TO CHURN'}")
print(f"  Churn Probability : {probability[1]:.4f}")
print(f"  Retain Probability: {probability[0]:.4f}")


# ── Final Summary ────────────────────────────────────

section("Project Summary")

print("""
  ┌────────────────────────────────────────────┐
  │  Task     : ML-2 — Customer Churn          │
  │  Models   : LR · DT · RF · GB             │
  │  Best     : Tuned Random Forest            │
  │  Bonuses  : Hyperparameter Tuning          │
  │             Feature Importance             │
  │             Cross-Validation               │
  │             SHAP Explainability            │
  │             K-Means Segmentation           │
  │             Churn Risk Dashboard           │
  └────────────────────────────────────────────┘
""")

print("  ✓  Project Completed Successfully\n")