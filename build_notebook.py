from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks" / "homework2_error_analysis.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r"""
# Homework 2: Regression and Classification Error Analysis

**Course:** Introduction to Data Science  
**Assignment:** Systematic error analysis with k-fold cross-validation  
**Student:** Tamir Eddy

This notebook investigates not only which models perform best, but **where and why they fail**. It uses an industrial LiFePO4 battery dataset containing 3,000 sequential cycles.

## Research questions

1. **Regression:** Can battery capacity (Ah) be estimated from cycle and operational measurements?
2. **Classification:** Can the same information identify a degraded state, defined as $SOH < 80\%$?
3. Are failures caused primarily by data quality, model assumptions, rare cases, or problem formulation?

All performance estimates and error analyses use **out-of-fold (OOF) predictions**. No observation is evaluated by a model that was trained on that observation.
"""),
    code(r"""
from pathlib import Path
import os
import warnings

# Keep Matplotlib's cache inside the project so the notebook also works in
# restricted environments where the default user cache is not writable.
MPL_CACHE = Path.cwd() / ".matplotlib-cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, auc, confusion_matrix,
    f1_score, fbeta_score, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 30)
RANDOM_STATE = 42
"""),
    md(r"""
## 1. Data and problem formulation

The raw data includes several deterministic transformations:

- $SOH = Capacity / 40 \times 100$
- $RUL = 3000 - CycleNumber$
- $DoD = CycleNumber / 30$

Using these together as predictors would create target leakage. For regression, `Capacity(Ah)` is the target and `SOH(%)`, `RUL(cycles)`, and `DoD(%)` are excluded. For classification, `Capacity(Ah)` is additionally excluded because it directly determines SOH and therefore the class label. `CycleNumber` is retained as a legitimate time-in-service feature, but it is a strong degradation proxy; this limitation is revisited in the discussion.
"""),
    code(r"""
def locate_data():
    candidates = [
        Path("../data/raw/industrial_lifepo4_bms.csv"),
        Path("data/raw/industrial_lifepo4_bms.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Place the dataset at data/raw/industrial_lifepo4_bms.csv")

DATA_PATH = locate_data()
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
display(df.head())
"""),
    code(r"""
quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "unique": df.nunique(),
})
display(quality)
print("Duplicate rows:", df.duplicated().sum())
"""),
    code(r"""
REG_TARGET = "Capacity(Ah)"
CLASS_TARGET = "degraded"
FEATURES = [
    "CycleNumber", "Voltage(V)", "Current(A)", "Temperature(C)",
    "Cell1_V", "Cell2_V", "Cell3_V", "Cell4_V",
    "Cell5_V", "Cell6_V", "Cell7_V", "Cell8_V",
]

df[CLASS_TARGET] = (df["SOH(%)"] < 80).astype(int)
X = df[FEATURES].copy()
y_reg = df[REG_TARGET].copy()
y_cls = df[CLASS_TARGET].copy()

display(pd.DataFrame({
    "count": y_cls.value_counts().sort_index(),
    "percent": y_cls.value_counts(normalize=True).sort_index().mul(100).round(2)
}).rename(index={0: "not degraded", 1: "degraded"}))
"""),
    md(r"""
## 2. Cross-validation design

We use **5 folds**. With 3,000 observations, each validation fold contains about 600 rows, which is large enough for stable metrics and still leaves 2,400 training observations. Five folds offer a practical balance:

- compared with a single split, every observation is evaluated and the estimate is less dependent on one lucky partition;
- compared with 10 folds, computation is approximately halved, especially for random forests;
- the training fraction (80%) provides moderate bias while avoiding the highly correlated estimates and higher variance/cost of leave-one-out validation.

`KFold` is used for regression, and `StratifiedKFold` preserves the degraded/non-degraded ratio for classification. Shuffling with a fixed seed makes the comparison reproducible. Because cycles are temporally ordered, random k-fold estimates interpolation performance, not forecasting to unseen future cycles; a time-aware split is recommended as a robustness extension.
"""),
    code(r"""
reg_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cls_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

reg_models = {
    "Linear Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=250, max_depth=12, min_samples_leaf=3,
        max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE
    ),
}

cls_models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, max_depth=10, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    ),
}

print("Regression hyperparameters")
for name, model in reg_models.items():
    print(name, model.get_params())
"""),
    md(r"""
## 3. Regression model comparison

Each model receives the same feature matrix. We compare MSE, RMSE, MAE, and $R^2$ using OOF predictions.
"""),
    code(r"""
reg_predictions = {}
reg_rows = []
for name, model in reg_models.items():
    pred = cross_val_predict(model, X, y_reg, cv=reg_cv, n_jobs=-1)
    reg_predictions[name] = pred
    reg_rows.append({
        "Model": name,
        "MSE": mean_squared_error(y_reg, pred),
        "RMSE": mean_squared_error(y_reg, pred) ** 0.5,
        "MAE": mean_absolute_error(y_reg, pred),
        "R2": r2_score(y_reg, pred),
    })

reg_results = pd.DataFrame(reg_rows).sort_values("RMSE").reset_index(drop=True)
display(reg_results.style.format({"MSE": "{:.5f}", "RMSE": "{:.5f}", "MAE": "{:.5f}", "R2": "{:.5f}"}))

best_reg_name = reg_results.loc[0, "Model"]
best_reg_pred = reg_predictions[best_reg_name]
residuals = y_reg.to_numpy() - best_reg_pred
abs_errors = np.abs(residuals)
print("Selected regression model for detailed error analysis:", best_reg_name)
"""),
    code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(data=reg_results, x="Model", y="RMSE", ax=axes[0], color="#4C78A8")
axes[0].set_title("Out-of-fold RMSE (lower is better)")
axes[0].tick_params(axis="x", rotation=20)

for name, pred in reg_predictions.items():
    axes[1].scatter(y_reg, pred, s=10, alpha=.35, label=name)
lo, hi = y_reg.min(), y_reg.max()
axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")
axes[1].set(xlabel="Actual capacity (Ah)", ylabel="OOF predicted capacity (Ah)", title="Actual vs predicted")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
"""),
    md(r"""
### 3.1 Residual analysis

Residuals are defined as $y-\hat{y}$. A well-specified model should have residuals centered near zero with no systematic curve or funnel shape.
"""),
    code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(best_reg_pred, residuals, s=13, alpha=.45)
axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0].set(xlabel="OOF predicted capacity (Ah)", ylabel="Residual (actual - predicted)",
            title=f"Residuals vs predictions: {best_reg_name}")
sns.histplot(residuals, bins=35, kde=True, ax=axes[1], color="#F58518")
axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
axes[1].set(title="Distribution of residuals", xlabel="Residual (Ah)")
plt.tight_layout()
plt.show()

print(f"Mean residual: {residuals.mean():.5f} Ah")
print(f"Correlation between |error| and prediction: {np.corrcoef(abs_errors, best_reg_pred)[0,1]:.3f}")
"""),
    md(r"""
**Interpretation guide.** A mean close to zero suggests little global bias. Curvature indicates missing non-linearity; repeated bands can reflect tree partitions; and a widening vertical spread indicates heteroscedasticity. The absolute-error correlation and feature-level plots below help distinguish these patterns.
"""),
    md(r"""
### 3.2 Error as a function of features

The plots compare each feature with signed and absolute errors. A LOESS-like visual summary is approximated with binned means to keep the result robust and readable.
"""),
    code(r"""
error_frame = X.copy()
error_frame["residual"] = residuals
error_frame["absolute_error"] = abs_errors

plot_features = ["CycleNumber", "Voltage(V)", "Current(A)", "Temperature(C)"]
fig, axes = plt.subplots(len(plot_features), 2, figsize=(13, 15))
for row, feature in enumerate(plot_features):
    sample = error_frame.sample(min(1800, len(error_frame)), random_state=RANDOM_STATE)
    axes[row, 0].scatter(sample[feature], sample["residual"], s=9, alpha=.28)
    axes[row, 0].axhline(0, color="black", linestyle="--", linewidth=.8)
    axes[row, 0].set(xlabel=feature, ylabel="Residual")
    axes[row, 1].scatter(sample[feature], sample["absolute_error"], s=9, alpha=.28, color="#E45756")
    axes[row, 1].set(xlabel=feature, ylabel="Absolute error")
fig.suptitle(f"Feature-dependent errors: {best_reg_name}", y=1.01, fontsize=15)
plt.tight_layout()
plt.show()

feature_error_corr = error_frame[FEATURES + ["residual", "absolute_error"]].corr()[["residual", "absolute_error"]]
display(feature_error_corr.sort_values("absolute_error", ascending=False))
"""),
    md(r"""
### 3.3 Extreme errors

The largest 5% of absolute OOF errors are examined individually. Because there are no missing or duplicated rows, unusual measurements, boundary cycles, and model limitations are the main candidate explanations.
"""),
    code(r"""
cutoff = np.quantile(abs_errors, .95)
extreme = df.loc[abs_errors >= cutoff, FEATURES + [REG_TARGET, "SOH(%)"]].copy()
extreme["predicted_capacity"] = best_reg_pred[abs_errors >= cutoff]
extreme["residual"] = residuals[abs_errors >= cutoff]
extreme["absolute_error"] = abs_errors[abs_errors >= cutoff]
extreme = extreme.sort_values("absolute_error", ascending=False)

print(f"95th-percentile absolute-error cutoff: {cutoff:.4f} Ah")
print(f"Extreme observations: {len(extreme)} ({len(extreme)/len(df):.1%})")
display(extreme.head(20))
"""),
    code(r"""
summary_compare = pd.concat({
    "all observations": df[FEATURES].describe().loc[["mean", "std", "min", "max"]],
    "top 5% errors": extreme[FEATURES].describe().loc[["mean", "std", "min", "max"]],
}, axis=1)
display(summary_compare)

extreme_cycle_bins = pd.cut(extreme["CycleNumber"], bins=[0, 600, 1200, 1800, 2400, 3000], include_lowest=True)
display(extreme_cycle_bins.value_counts(sort=False).rename("extreme_error_count").to_frame())
"""),
    md(r"""
### 3.4 Statistical properties of errors

Positive skew indicates a longer positive tail (underprediction in extreme cases), while negative skew indicates a longer negative tail. Excess kurtosis above zero indicates heavier tails than a normal distribution.
"""),
    code(r"""
error_stats = pd.Series({
    "MAE": mean_absolute_error(y_reg, best_reg_pred),
    "Mean residual": residuals.mean(),
    "Residual standard deviation": residuals.std(ddof=1),
    "Residual skewness": skew(residuals, bias=False),
    "Residual excess kurtosis": kurtosis(residuals, fisher=True, bias=False),
})
display(error_stats.to_frame("value").style.format("{:.6f}"))
"""),
    md(r"""
### 3.5 Regression discussion

- **Relative performance:** The Decision Tree is the clear winner (`RMSE = 0.02663 Ah`, `MAE = 0.02173 Ah`, $R^2 = 0.99994$). Linear Regression is second (`RMSE = 0.37628 Ah`), while the deliberately regularized Random Forest is third (`RMSE = 0.59051 Ah`). MAE and RMSE agree on the ranking, so it is not driven only by a few extreme errors.
- **Model assumptions:** Linear regression assumes an additive linear response. Battery degradation is curved over cycle life, so systematic residual curvature is expected. Trees capture non-linearity and interactions without requiring homoskedastic Gaussian residuals.
- **Bias-variance trade-off:** The linear model has high bias but comparatively low variance. An unrestricted tree would have low training bias and high variance; depth and leaf-size constraints reduce this risk. The random forest averages many decorrelated trees, usually reducing variance at the cost of interpretability and computation.
- **Outliers and noise:** Linear regression can be pulled by unusual capacity values. A single tree is unstable to local perturbations, while a forest is more robust through averaging. None of these models automatically resolves measurement error or concept drift.
- **Where linear models fail:** A single slope cannot represent accelerating or stage-dependent degradation. Tree-based models have an advantage when the cycle-capacity relationship changes across regions.
- **Interpretability vs performance:** Linear coefficients are globally interpretable. A tree provides readable rules when shallow. A forest often improves predictive performance but requires feature importance or local explanations.
- **Observed failure pattern:** The selected tree has almost no global bias (mean residual `-0.00498 Ah`). Its top 5% errors are strongly concentrated late in life: 82 of 150 occur in cycles 2401-3000 and 46 occur in cycles 1801-2400. Absolute error correlates with cycle number (`r = 0.449`), while correlations with voltage, current, temperature, and cell voltages are close to zero. This points to model approximation error in the steeper late-life degradation region rather than obvious sensor anomalies or missing data.
- **Preferred model:** The Decision Tree is preferred for this interpolation task because it has by far the lowest OOF errors. Its piecewise predictions are less smooth than the physical process, so the result should not be interpreted as proof of future-cycle forecasting ability.
- **Main lesson:** Aggregate scores alone hide that most large errors occur in later degradation stages. OOF residual analysis turns model comparison into a diagnosis.
"""),
    md(r"""
## 4. Classification model comparison

The positive class is a degraded battery (`SOH < 80%`). We compare accuracy, precision, recall, F1, MCC, and ROC-AUC at a default threshold of 0.5 using OOF probabilities.
"""),
    code(r"""
cls_probabilities = {}
cls_rows = []
for name, model in cls_models.items():
    proba = cross_val_predict(model, X, y_cls, cv=cls_cv, method="predict_proba", n_jobs=-1)[:, 1]
    pred = (proba >= .5).astype(int)
    cls_probabilities[name] = proba
    cls_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_cls, pred),
        "Precision": precision_score(y_cls, pred, zero_division=0),
        "Recall": recall_score(y_cls, pred, zero_division=0),
        "F1": f1_score(y_cls, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_cls, pred),
        "ROC_AUC": roc_auc_score(y_cls, proba),
    })

cls_results = pd.DataFrame(cls_rows).sort_values(["MCC", "ROC_AUC"], ascending=False).reset_index(drop=True)
display(cls_results.style.format({c: "{:.4f}" for c in cls_results.columns if c != "Model"}))

best_cls_name = cls_results.loc[0, "Model"]
best_cls_proba = cls_probabilities[best_cls_name]
best_cls_pred = (best_cls_proba >= .5).astype(int)
correct = best_cls_pred == y_cls.to_numpy()
print("Selected classifier for detailed error analysis:", best_cls_name)
"""),
    md(r"""
### 4.1 Confusion matrix and error implications

A **false negative (FN)** labels a degraded battery as healthy, potentially delaying inspection or replacement. A **false positive (FP)** triggers unnecessary inspection or replacement. In this safety and maintenance context, FN is generally more critical, although the final operating threshold should reflect real financial and safety costs.
"""),
    code(r"""
cm = confusion_matrix(y_cls, best_cls_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["not degraded", "degraded"]).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title(f"OOF confusion matrix: {best_cls_name} (threshold = 0.5)")
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"False-positive rate: {fp/(fp+tn):.3%}")
print(f"False-negative rate: {fn/(fn+tp):.3%}")
"""),
    code(r"""
error_type = np.select(
    [(y_cls.to_numpy() == 0) & (best_cls_pred == 1),
     (y_cls.to_numpy() == 1) & (best_cls_pred == 0)],
    ["FP", "FN"], default="Correct"
)
cls_error_frame = X.copy()
cls_error_frame["actual"] = y_cls.to_numpy()
cls_error_frame["probability_degraded"] = best_cls_proba
cls_error_frame["error_type"] = error_type

display(cls_error_frame.loc[cls_error_frame["error_type"] != "Correct"].sort_values("CycleNumber"))
"""),
    md(r"""
### 4.2 Probability-based analysis

The probability distributions reveal calibration and confidence. High-confidence errors are FPs with probabilities near 1 or FNs with probabilities near 0.
"""),
    code(r"""
prob_frame = pd.DataFrame({
    "Predicted probability of degradation": best_cls_proba,
    "Prediction": np.where(correct, "Correct", "Incorrect"),
})
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(data=prob_frame, x="Predicted probability of degradation", hue="Prediction",
             bins=25, stat="density", common_norm=False, element="step", ax=ax)
ax.axvline(.5, color="black", linestyle="--", linewidth=1)
ax.set_title(f"OOF probability distribution: {best_cls_name}")
plt.show()

confidence = np.where(best_cls_pred == 1, best_cls_proba, 1 - best_cls_proba)
high_conf_errors = cls_error_frame.loc[(~correct) & (confidence >= .8)].copy()
high_conf_errors["prediction_confidence"] = confidence[(~correct) & (confidence >= .8)]
print("High-confidence errors (confidence >= 0.80):", len(high_conf_errors))
display(high_conf_errors.sort_values("prediction_confidence", ascending=False).head(20))
"""),
    md(r"""
### 4.3 Error as a function of features

Feature distributions for correct and incorrect predictions locate regions that are prone to misclassification.
"""),
    code(r"""
feature_compare = X.copy()
feature_compare["Prediction"] = np.where(correct, "Correct", "Incorrect")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, feature in zip(axes.flat, ["CycleNumber", "Voltage(V)", "Current(A)", "Temperature(C)"]):
    sns.histplot(data=feature_compare, x=feature, hue="Prediction", stat="density",
                 common_norm=False, element="step", bins=25, ax=ax)
    ax.set_title(feature)
plt.tight_layout()
plt.show()

display(feature_compare.groupby("Prediction")[FEATURES].agg(["mean", "std"]).T)
"""),
    md(r"""
### 4.4 Threshold sensitivity

Thresholds from 0.1 through 0.9 show the precision-recall trade-off. We also track F1 and MCC. ROC-AUC is threshold-independent and summarizes ranking quality.
"""),
    code(r"""
threshold_rows = []
for threshold in np.arange(.1, 1.0, .1):
    pred = (best_cls_proba >= threshold).astype(int)
    threshold_rows.append({
        "Threshold": threshold,
        "Precision": precision_score(y_cls, pred, zero_division=0),
        "Recall": recall_score(y_cls, pred, zero_division=0),
        "F1": f1_score(y_cls, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_cls, pred),
        "FP": ((y_cls.to_numpy() == 0) & (pred == 1)).sum(),
        "FN": ((y_cls.to_numpy() == 1) & (pred == 0)).sum(),
    })
threshold_results = pd.DataFrame(threshold_rows)
display(threshold_results.style.format({c: "{:.3f}" for c in ["Threshold", "Precision", "Recall", "F1", "MCC"]}))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for metric in ["Precision", "Recall", "F1", "MCC"]:
    axes[0].plot(threshold_results["Threshold"], threshold_results[metric], marker="o", label=metric)
axes[0].set(xlabel="Classification threshold", ylabel="Score", ylim=(0, 1.05), title="Threshold sensitivity")
axes[0].legend()

axes[1].plot(threshold_results["Threshold"], threshold_results["FP"], marker="o", label="False positives")
axes[1].plot(threshold_results["Threshold"], threshold_results["FN"], marker="o", label="False negatives")
axes[1].set(xlabel="Classification threshold", ylabel="Count", title="Error trade-off")
axes[1].legend()
plt.tight_layout()
plt.show()
"""),
    code(r"""
betas = np.linspace(.25, 3, 40)
beta_scores = []
for beta in betas:
    beta_scores.append(fbeta_score(y_cls, best_cls_pred, beta=beta, zero_division=0))

fpr, tpr, _ = roc_curve(y_cls, best_cls_proba)
roc_auc = roc_auc_score(y_cls, best_cls_proba)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(betas, beta_scores, color="#54A24B")
axes[0].set(xlabel=r"$\beta$", ylabel=r"$F_\beta$ score", title=r"$F_\beta$ at threshold 0.5")
axes[0].axvline(1, color="black", linestyle="--", linewidth=1)

axes[1].plot(fpr, tpr, label=f"{best_cls_name} (AUC={roc_auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Random ranking")
axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="OOF ROC curve")
axes[1].legend()
plt.tight_layout()
plt.show()
"""),
    code(r"""
best_f1_row = threshold_results.loc[threshold_results["F1"].idxmax()]
best_mcc_row = threshold_results.loc[threshold_results["MCC"].idxmax()]
recall_candidates = threshold_results[threshold_results["Recall"] >= .95]
safe_row = recall_candidates.sort_values(["Precision", "MCC"], ascending=False).head(1)

print(f"Best tested F1 threshold: {best_f1_row['Threshold']:.1f} (F1={best_f1_row['F1']:.3f})")
print(f"Best tested MCC threshold: {best_mcc_row['Threshold']:.1f} (MCC={best_mcc_row['MCC']:.3f})")
if not safe_row.empty:
    row = safe_row.iloc[0]
    print(f"High-recall candidate: {row['Threshold']:.1f} (recall={row['Recall']:.3f}, precision={row['Precision']:.3f})")
"""),
    md(r"""
### 4.5 Classification discussion

- **Strengths and limitations:** The Random Forest and Decision Tree both achieve `99.97%` accuracy and make one FN with no FP at threshold 0.5. The forest has `MCC = 0.9991` and `ROC-AUC = 1.0000`. Logistic Regression is less accurate (`98.33%`) but still ranks cases extremely well (`ROC-AUC = 0.9999`), showing that most of its errors come from the chosen decision boundary rather than poor ordering.
- **Critical error:** A false negative is more consequential because an actually degraded battery is treated as healthy. A lower threshold can reduce FNs, but increases unnecessary alerts.
- **Systematic failure region:** The only error at threshold 0.5 is cycle 2222, the first observation just across the `SOH = 80%` boundary. It is predicted with degradation probability `0.442`, so it is a low-confidence boundary error, not an unexplained high-confidence failure. This is mainly a formulation issue: a continuous process is converted into a hard label.
- **Stable operating regions:** Thresholds 0.4-0.8 are very stable, with F1 and MCC near 0.999. Threshold 0.4 removes the FN at the cost of one FP and maximizes both tested F1 and MCC. Threshold 0.9 is unstable for recall and creates 27 FNs.
- **High-confidence errors:** There are no incorrect predictions with confidence at least 0.80. This reduces concern about confidently wrong regions in this dataset, but does not establish calibration on new battery packs.
- **Proxy risk:** `CycleNumber` can dominate because the dataset contains one battery pack following one degradation trajectory. Performance may not generalize to other packs or operating regimes.
"""),
    md(r"""
## 5. Final reflection

1. **Where do the models fail most?** Regression errors concentrate late in battery life: 128 of the 150 largest errors occur after cycle 1800. The classifier's single error is exactly at cycle 2222, the first degraded case across the 80% SOH boundary.
2. **Data, model, or formulation?** All three contribute, but the observed failures are mainly model/formulation effects. The tree approximates a smooth, steepening degradation curve with constant-valued leaves, while the binary target creates an artificial boundary. The single-pack sequential dataset limits the strength of any generalization claim, and deterministic derived columns create severe leakage risk if used carelessly.
3. **Proposed improvements:** Collect multiple battery packs under varied loads and temperatures; validate with grouped or forward-chaining splits; add physically meaningful rolling/lag features available at prediction time; tune hyperparameters within nested CV; calibrate classification probabilities; and choose the threshold from explicit FN/FP costs.
4. **Main insights:** Cross-validated aggregate metrics are only a starting point. Residual shape, extreme cases, probability confidence, feature-conditioned errors, and threshold sensitivity reveal whether a good average score represents robust behavior. Leakage control and validation design can matter more than choosing a more complex algorithm.

## Reproducibility note

Every random component uses seed 42, all predictions used for reported metrics are OOF, and all models use the same predictor set within each task. The regression and classification targets require different exclusions because capacity directly determines the degradation label.
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
