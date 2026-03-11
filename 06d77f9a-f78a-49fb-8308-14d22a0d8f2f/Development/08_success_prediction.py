import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ─────────────────────────────────────────────────────────────
# 1. Create Behavior Intensity (Ratio) Features
# ─────────────────────────────────────────────────────────────

print("="*60)
print("CREATING BEHAVIOR INTENSITY FEATURES")
print("="*60)

# total events per user
_total_evt = behavior_df[valid_event_cols].sum(axis=1).replace(0, 1)

# create ratio features
for col in valid_event_cols:
    behavior_df[col + "_ratio"] = behavior_df[col] / _total_evt

ratio_cols = [c + "_ratio" for c in valid_event_cols]

print(f"Created {len(ratio_cols)} ratio features")


# ─────────────────────────────────────────────────────────────
# 2. Feature Selection (with leakage filter)
# ─────────────────────────────────────────────────────────────

# Define leakage features — these are outcome proxies that
# directly encode the target and must be excluded
leakage_features = ['total_events', 'credits_used']

clean_event_cols = [f for f in valid_event_cols if f not in leakage_features]
clean_ratio_cols = [f for f in ratio_cols if f not in [lf + "_ratio" for lf in leakage_features]]

extra_features = [
    "events_first_session",
    "event_types_first_session",
    "number_of_sessions"
]

feature_cols = clean_event_cols + clean_ratio_cols + extra_features

excluded = [f for f in leakage_features if f in valid_event_cols]
print(f"\nLeakage filter removed: {excluded if excluded else 'none (not present in valid_event_cols)'}")
print(f"Features after leakage filter: {len(feature_cols)} "
      f"(was {len(valid_event_cols + ratio_cols + extra_features)})")

X = behavior_df[feature_cols].copy().astype(float)
y = behavior_df["successful_user"].astype(int)

print("\nDATASET OVERVIEW")
print("-"*60)
print(f"Dataset shape: {X.shape}")
print(f"""
Class distribution
Successful      : {y.sum()} ({y.mean()*100:.2f}%)
Not Successful  : {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.2f}%)
""")


# ─────────────────────────────────────────────────────────────
# 3. Train / Test Split
# ─────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size : {len(X_test)}")


# ─────────────────────────────────────────────────────────────
# 4. Train Random Forest
# ─────────────────────────────────────────────────────────────

rf_model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("\nModel trained successfully ✓")


# ─────────────────────────────────────────────────────────────
# 5. Prediction Probabilities & ROC-AUC
# ─────────────────────────────────────────────────────────────

rf_probs = rf_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, rf_probs)

print("\nMODEL PERFORMANCE")
print("="*50)
print(f"ROC-AUC : {roc_auc:.4f}")


# ─────────────────────────────────────────────────────────────
# 6. Optimal Threshold (Youden's J)
# ─────────────────────────────────────────────────────────────

fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
youden_index = tpr - fpr
best_index = np.argmax(youden_index)
best_threshold = thresholds[best_index]

print(f"\nOptimal Probability Threshold: {best_threshold:.3f}")

rf_preds = (rf_probs >= best_threshold).astype(int)
accuracy = accuracy_score(y_test, rf_preds)

print(f"Accuracy (optimized threshold): {accuracy*100:.2f}%")
print("\nClassification Report")
print(classification_report(y_test, rf_preds))


# ─────────────────────────────────────────────────────────────
# 7. Feature Importance
# ─────────────────────────────────────────────────────────────

feature_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

top10_features = feature_importance_df.head(10)

print("\nTop 10 Most Predictive Behaviors")
print("-"*50)
for _i, _row in top10_features.iterrows():
    print(f"{_i+1:2d}. {_row['feature']}  (importance={_row['importance']:.4f})")


# ─────────────────────────────────────────────────────────────
# 8. Visualization — Zerve dark theme
# ─────────────────────────────────────────────────────────────

BG    = "#1D1D20"
TEXT  = "#fbfbff"
SEC   = "#909094"
BLUE  = "#A1C9F4"
CORAL = "#FF9F9B"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   SEC,
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
    "grid.color":       SEC,
    "grid.alpha":       0.3,
})

# --- Plot 1: Top-10 Feature Importance ---
feature_importance_fig, ax_fi = plt.subplots(figsize=(12, 7))
feature_importance_fig.patch.set_facecolor(BG)
ax_fi.set_facecolor(BG)

_labels   = top10_features["feature"][::-1].tolist()
_values   = top10_features["importance"][::-1].tolist()
_bar_colors = [BLUE] * 10

_bars = ax_fi.barh(_labels, _values, color=_bar_colors, edgecolor="none", height=0.65)

# Add value labels
for _b in _bars:
    _w = _b.get_width()
    ax_fi.text(_w + 0.0005, _b.get_y() + _b.get_height()/2,
               f"{_w:.4f}", va="center", ha="left", color=TEXT, fontsize=9)

ax_fi.set_xlabel("Feature Importance (Gini)", color=TEXT, fontsize=11)
ax_fi.set_title("Top 10 Most Predictive Behaviors — Success Prediction",
                color=TEXT, fontsize=13, pad=14)
for _sp in ax_fi.spines.values():
    _sp.set_edgecolor(SEC)
ax_fi.tick_params(colors=TEXT)
ax_fi.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# --- Plot 2: Confusion Matrix ---
confusion_matrix_fig, ax_cm = plt.subplots(figsize=(7, 6))
confusion_matrix_fig.patch.set_facecolor(BG)
ax_cm.set_facecolor(BG)

cm = confusion_matrix(y_test, rf_preds)
im = ax_cm.imshow(cm, cmap="Blues")

ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["Not Successful", "Successful"], color=TEXT)
ax_cm.set_yticklabels(["Not Successful", "Successful"], color=TEXT)
ax_cm.set_xlabel("Predicted", color=TEXT, fontsize=11)
ax_cm.set_ylabel("Actual", color=TEXT, fontsize=11)
ax_cm.set_title("Confusion Matrix", color=TEXT, fontsize=13, pad=14)

for _i in range(2):
    for _j in range(2):
        ax_cm.text(_j, _i, str(cm[_i, _j]),
                   ha="center", va="center", color=TEXT, fontsize=14, fontweight="bold")

for _sp in ax_cm.spines.values():
    _sp.set_edgecolor(SEC)
plt.tight_layout()
plt.show()


# --- Plot 3: ROC Curve ---
roc_curve_fig, ax_roc = plt.subplots(figsize=(8, 6))
roc_curve_fig.patch.set_facecolor(BG)
ax_roc.set_facecolor(BG)

ax_roc.plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
ax_roc.plot([0, 1], [0, 1], color=SEC, lw=1.2, linestyle="--", label="Random baseline")
ax_roc.scatter([fpr[best_index]], [tpr[best_index]], color="#ffd400", zorder=5,
               s=100, label=f"Optimal threshold ({best_threshold:.2f})")

ax_roc.set_xlabel("False Positive Rate", color=TEXT, fontsize=11)
ax_roc.set_ylabel("True Positive Rate", color=TEXT, fontsize=11)
ax_roc.set_title("ROC Curve — Success Prediction", color=TEXT, fontsize=13, pad=14)
ax_roc.legend(facecolor=BG, edgecolor=SEC, labelcolor=TEXT, fontsize=10)
for _sp in ax_roc.spines.values():
    _sp.set_edgecolor(SEC)
ax_roc.tick_params(colors=TEXT)
ax_roc.grid(linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

print("\nVisualization complete. ✓")


# ─────────────────────────────────────────────────────────────
# 9. Save Outputs for Deployment
# ─────────────────────────────────────────────────────────────

print("\nSaving model artifacts for Streamlit deployment...")

feature_importance_df.to_csv("feature_importance.csv", index=False)
user_features.to_csv("user_features.csv", index=False)
joblib.dump(rf_model, "rf_model.pkl")

print("Saved:")
print("✓ feature_importance.csv")
print("✓ user_features.csv")
print("✓ rf_model.pkl")

print(feature_importance_df.head(10).to_string())
