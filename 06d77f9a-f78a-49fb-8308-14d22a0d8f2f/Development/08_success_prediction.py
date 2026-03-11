import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
print("=" * 60)
print("CREATING BEHAVIOR INTENSITY FEATURES")
print("=" * 60)

# total events per user (for normalisation only — NOT a feature)
_total_evt = behavior_df[valid_event_cols].sum(axis=1).replace(0, 1)

# Create ratio features: event_x / total_events
# These capture *composition* of activity, NOT volume → no leakage
for col in valid_event_cols:
    behavior_df[col + "_ratio"] = behavior_df[col] / _total_evt

ratio_cols = [c + "_ratio" for c in valid_event_cols]
print(f"Created {len(ratio_cols)} ratio features")


# ─────────────────────────────────────────────────────────────
# 2. Leakage Audit & Feature Selection
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LEAKAGE AUDIT")
print("=" * 60)

# The target is defined as: total_events >= p80_threshold (80th percentile)
# Therefore ANY feature that encodes total event volume will perfectly predict
# the target — this is the source of the near-perfect 0.9996 ROC-AUC.
#
# LEAKAGE VECTORS IDENTIFIED:
# (A) Raw event count cols: their SUM = total_events = the target numerator
# (B) number_of_sessions: r > 0.9 with total_events (users with more events
#     have more sessions); also in extra_features
# (C) events_per_session: = total_events / sessions → direct proxy
# (D) number_of_event_types: monotonically increases with total_events
# (E) event_diversity_ratio: = event_types / total_events (encodes total_events)
# (F) total_credits_used + avg_credits_per_event: credits scale with events
#
# SAFE FEATURES (no volume leakage):
# - Ratio features: event_x / total_events → encode *composition*, not volume
# - events_first_session: only first session, isolated before observation window
# - event_types_first_session: same temporal isolation

leakage_features = [
    'total_events',          # A: IS the target numerator
    'credits_used',          # A: raw count (sum encodes total_events)
    'total_credits_used',    # B: aggregate correlated with total_events
    'credits_used_ratio',    # already excluded via credits_used
    'number_of_sessions',    # C: r > 0.85 with total_events
    'events_per_session',    # D: = total_events / sessions
    'number_of_event_types', # E: increases monotonically with events
    'event_diversity_ratio', # F: = event_types / total_events
    'avg_credits_per_event', # G: credits / total_events — encodes volume
    'days_active',           # H: active span scales with total_events
    'active_multiple_days',  # I: boolean proxy for days_active
]

# Verify which leaky features are actually in behavior_df
_all_behav_cols = behavior_df.columns.tolist()
_confirmed_leaky = [f for f in leakage_features if f in _all_behav_cols]

print("Confirmed leakage features (excluded from model):")
for _lf in _confirmed_leaky:
    _corr = behavior_df[_lf].corr(behavior_df['successful_user'].astype(int))
    print(f"  {_lf:<35s}  r(target) = {_corr:.4f}")

# Correlation of number_of_sessions with total_events (key diagnostic)
_r_sessions_events = behavior_df['number_of_sessions'].corr(behavior_df['total_events'])
_r_ntypes_events = behavior_df['number_of_event_types'].corr(behavior_df['total_events'])
print(f"\nCorrelations confirming volume leakage:")
print(f"  number_of_sessions vs total_events    r = {_r_sessions_events:.4f}")
print(f"  number_of_event_types vs total_events r = {_r_ntypes_events:.4f}")

# ── Clean feature set ──────────────────────────────────────────
# ONLY ratio features + first-session features (temporally isolated)
# Raw event counts excluded entirely — their sum reconstructs the target

leakage_features = set(leakage_features)

clean_event_cols = [
    f for f in valid_event_cols
    if f not in leakage_features
]

clean_ratio_cols = [
    f for f in ratio_cols
    if f.replace('_ratio', '') not in leakage_features
    and f not in leakage_features
]

# Keep only RATIO features + early-session features (no volume features)
feature_cols = clean_ratio_cols + [
    'events_first_session',
    'event_types_first_session',
]

excluded_raw = [f for f in clean_event_cols if f in behavior_df.columns]

print(f"\nFeature summary:")
print(f"  Ratio features (safe, compositional)  : {len(clean_ratio_cols)}")
print(f"  First-session features (safe)         : 2")
print(f"  Raw event count features              : EXCLUDED (all {len(clean_event_cols)})")
print(f"  Volume/aggregate features             : EXCLUDED ({len(_confirmed_leaky)})")
print(f"  TOTAL features for model              : {len(feature_cols)}")


# ─────────────────────────────────────────────────────────────
# 3. Build X, y
# ─────────────────────────────────────────────────────────────
X = behavior_df[feature_cols].copy().astype(float)
y = behavior_df["successful_user"].astype(int)

print("\nDATASET OVERVIEW")
print("-" * 60)
print(f"Dataset shape: {X.shape}")
print(f"""
Class distribution
Successful      : {y.sum()} ({y.mean()*100:.2f}%)
Not Successful  : {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.2f}%)
""")

# Sanity check: verify no feature correlates > 0.6 with total_events
_vol_check = pd.Series({
    f: X[f].corr(behavior_df['total_events'])
    for f in feature_cols
}).abs().sort_values(ascending=False)

_high_vol_corr = _vol_check[_vol_check > 0.6]
if len(_high_vol_corr) > 0:
    print(f"⚠️  WARNING: {len(_high_vol_corr)} features have |r| > 0.6 with total_events:")
    print(_high_vol_corr.to_string())
else:
    print(f"✓ Leakage check passed: no feature has |r| > 0.6 with total_events")
    print(f"  Max |r| with total_events: {_vol_check.max():.4f} ({_vol_check.idxmax()})")


# ─────────────────────────────────────────────────────────────
# 4. Train / Test Split (stratified)
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size : {len(X_test)}")


# ─────────────────────────────────────────────────────────────
# 5. Stratified K-Fold Cross-Validation (before final fit)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STRATIFIED 5-FOLD CROSS-VALIDATION")
print("=" * 60)

# Regularised RF to prevent overfitting on small-volume ratio features
rf_cv = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,           # Limit depth — prevents memorising low-count ratio patterns
    min_samples_leaf=10,   # Require meaningful leaf populations
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
_cv_auc_scores = cross_val_score(rf_cv, X_train, y_train, cv=_skf, scoring="roc_auc", n_jobs=-1)

print(f"CV ROC-AUC scores (5-fold): {_cv_auc_scores.round(4).tolist()}")
print(f"Mean AUC : {_cv_auc_scores.mean():.4f} ± {_cv_auc_scores.std():.4f}")

_gap_ok = _cv_auc_scores.std() < 0.05
_range_ok = 0.65 <= _cv_auc_scores.mean() <= 0.95
print(f"\nConsistency check (std < 0.05)  : {'✓ PASS' if _gap_ok else '✗ FAIL'} (std = {_cv_auc_scores.std():.4f})")
print(f"Realistic range (0.65–0.95)     : {'✓ PASS' if _range_ok else '⚠  CHECK'} (mean = {_cv_auc_scores.mean():.4f})")


# ─────────────────────────────────────────────────────────────
# 6. Train Final Model on Full Training Set
# ─────────────────────────────────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("\nFinal model trained successfully ✓")


# ─────────────────────────────────────────────────────────────
# 7. Prediction Probabilities & ROC-AUC
# ─────────────────────────────────────────────────────────────
rf_probs = rf_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, rf_probs)

# Train AUC for gap analysis
_train_probs = rf_model.predict_proba(X_train)[:, 1]
_train_auc = roc_auc_score(y_train, _train_probs)

print("\nMODEL PERFORMANCE")
print("=" * 50)
print(f"Train ROC-AUC : {_train_auc:.4f}")
print(f"Test  ROC-AUC : {roc_auc:.4f}")
print(f"Train-Test Gap: {_train_auc - roc_auc:.4f}  ({'✓ Acceptable' if (_train_auc - roc_auc) < 0.1 else '⚠ Large gap — possible overfit'})")


# ─────────────────────────────────────────────────────────────
# 8. Optimal Threshold (Youden's J)
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
# 9. Feature Importance
# ─────────────────────────────────────────────────────────────
feature_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

top10_features = feature_importance_df.head(10)

print("Top 10 Most Predictive Behaviors")
print("-" * 50)
for _i, _row in top10_features.iterrows():
    print(f"{_i+1:2d}. {_row['feature']}  (importance={_row['importance']:.4f})")


# ─────────────────────────────────────────────────────────────
# 10. Visualization — Zerve dark theme
# ─────────────────────────────────────────────────────────────
BG    = "#1D1D20"
TEXT  = "#fbfbff"
SEC   = "#909094"
BLUE  = "#A1C9F4"
CORAL = "#FF9F9B"
GREEN = "#8DE5A1"
GOLD  = "#ffd400"

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
_fi_colors = [BLUE] * 10

_bars = ax_fi.barh(_labels, _values, color=_fi_colors, edgecolor="none", height=0.65)

for _b in _bars:
    _w = _b.get_width()
    ax_fi.text(_w + 0.0005, _b.get_y() + _b.get_height()/2,
               f"{_w:.4f}", va="center", ha="left", color=TEXT, fontsize=9)

ax_fi.set_xlabel("Feature Importance (Gini)", color=TEXT, fontsize=11)
ax_fi.set_title("Top 10 Most Predictive Behaviors — Success Prediction\n(Leakage-free: ratio features only)",
                color=TEXT, fontsize=12, pad=14)
for _sp in ax_fi.spines.values():
    _sp.set_edgecolor(SEC)
ax_fi.tick_params(colors=TEXT)
ax_fi.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# --- Plot 2: CV Scores + Train/Test AUC ---
cv_scores_fig, ax_cv = plt.subplots(figsize=(9, 5))
cv_scores_fig.patch.set_facecolor(BG)
ax_cv.set_facecolor(BG)

_fold_labels = [f"Fold {i+1}" for i in range(5)]
_bar_colors = [GREEN if s >= 0.70 else CORAL for s in _cv_auc_scores]
_cv_bars = ax_cv.bar(_fold_labels, _cv_auc_scores, color=_bar_colors, edgecolor="none", width=0.55)

for _b, _s in zip(_cv_bars, _cv_auc_scores):
    ax_cv.text(_b.get_x() + _b.get_width()/2, _b.get_height() + 0.005,
               f"{_s:.4f}", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")

ax_cv.axhline(_cv_auc_scores.mean(), color=GOLD, lw=2, linestyle="--",
              label=f"Mean CV AUC = {_cv_auc_scores.mean():.4f}")
ax_cv.axhline(roc_auc, color=BLUE, lw=1.5, linestyle=":",
              label=f"Test AUC = {roc_auc:.4f}")

ax_cv.set_ylim(max(0, _cv_auc_scores.min() - 0.05), min(1.0, _cv_auc_scores.max() + 0.10))
ax_cv.set_ylabel("ROC-AUC", color=TEXT, fontsize=11)
ax_cv.set_title("5-Fold Stratified CV — AUC Consistency Check", color=TEXT, fontsize=13, pad=14)
ax_cv.legend(facecolor=BG, edgecolor=SEC, labelcolor=TEXT, fontsize=10)
for _sp in ax_cv.spines.values():
    _sp.set_edgecolor(SEC)
ax_cv.tick_params(colors=TEXT)
ax_cv.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# --- Plot 3: Confusion Matrix ---
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

for _ci in range(2):
    for _cj in range(2):
        ax_cm.text(_cj, _ci, str(cm[_ci, _cj]),
                   ha="center", va="center", color=TEXT, fontsize=14, fontweight="bold")

for _sp in ax_cm.spines.values():
    _sp.set_edgecolor(SEC)
plt.tight_layout()
plt.show()


# --- Plot 4: ROC Curve ---
roc_curve_fig, ax_roc = plt.subplots(figsize=(8, 6))
roc_curve_fig.patch.set_facecolor(BG)
ax_roc.set_facecolor(BG)

ax_roc.plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
ax_roc.plot([0, 1], [0, 1], color=SEC, lw=1.2, linestyle="--", label="Random baseline")
ax_roc.scatter([fpr[best_index]], [tpr[best_index]], color=GOLD, zorder=5,
               s=100, label=f"Optimal threshold ({best_threshold:.2f})")

ax_roc.set_xlabel("False Positive Rate", color=TEXT, fontsize=11)
ax_roc.set_ylabel("True Positive Rate", color=TEXT, fontsize=11)
ax_roc.set_title("ROC Curve — Success Prediction (Leakage-free)", color=TEXT, fontsize=13, pad=14)
ax_roc.legend(facecolor=BG, edgecolor=SEC, labelcolor=TEXT, fontsize=10)
for _sp in ax_roc.spines.values():
    _sp.set_edgecolor(SEC)
ax_roc.tick_params(colors=TEXT)
ax_roc.grid(linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

print("\nVisualization complete. ✓")


# ─────────────────────────────────────────────────────────────
# 11. Final Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LEAKAGE-FIX SUMMARY")
print("=" * 60)
print(f"""
Root Cause Identified:
  Target = total_events >= {p80_threshold:.0f} (80th percentile)
  Raw event count features SUM to total_events → trivial leakage
  Volume-correlated features (sessions, event types, etc.) also leaked

Fix Applied:
  ✓ Removed ALL raw event count features (sum = target numerator)
  ✓ Removed all volume/aggregate features from user_features
  ✓ Kept only RATIO features (event composition, not volume)
  ✓ Kept first-session features (temporally isolated)
  ✓ Applied regularisation: max_depth=8, min_samples_leaf=10
  ✓ Validated with 5-fold stratified cross-validation

Results:
  CV ROC-AUC : {_cv_auc_scores.mean():.4f} ± {_cv_auc_scores.std():.4f}
  Test AUC   : {roc_auc:.4f}
  Train-Test gap: {_train_auc - roc_auc:.4f}
""")


# ─────────────────────────────────────────────────────────────
# 12. Save Outputs for Deployment
# ─────────────────────────────────────────────────────────────
print("Saving model artifacts for deployment...")

feature_importance_df.to_csv("feature_importance.csv", index=False)
user_features.to_csv("user_features.csv", index=False)
joblib.dump(rf_model, "rf_model.pkl")

print("Saved:")
print("✓ feature_importance.csv")
print("✓ user_features.csv")
print("✓ rf_model.pkl")

print(feature_importance_df.head(10).to_string())
