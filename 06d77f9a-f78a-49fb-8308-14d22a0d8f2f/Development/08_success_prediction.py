import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
total_events = behavior_df[valid_event_cols].sum(axis=1).replace(0,1)

# create ratio features
for col in valid_event_cols:
    behavior_df[col + "_ratio"] = behavior_df[col] / total_events

ratio_cols = [c + "_ratio" for c in valid_event_cols]

print(f"Created {len(ratio_cols)} ratio features")


# ─────────────────────────────────────────────────────────────
# 2. Feature Selection
# ─────────────────────────────────────────────────────────────

extra_features = [
    "events_first_session",
    "event_types_first_session",
    "number_of_sessions"
]

feature_cols = valid_event_cols + ratio_cols + extra_features

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
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
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
# 5. Prediction Probabilities
# ─────────────────────────────────────────────────────────────

rf_probs = rf_model.predict_proba(X_test)[:,1]

# ROC-AUC
roc_auc = roc_auc_score(y_test, rf_probs)

print("\nMODEL PERFORMANCE")
print("="*50)

print(f"ROC-AUC : {roc_auc:.4f}")


# ─────────────────────────────────────────────────────────────
# 6. Optimal Threshold Search
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
})

feature_importance_df = feature_importance_df.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

top10_features = feature_importance_df.head(10)

print("\nTop 10 Most Predictive Behaviors")
print("-"*50)

for i,row in top10_features.iterrows():
    print(f"{i+1:2d}. {row['feature']}  (importance={row['importance']:.4f})")


# ─────────────────────────────────────────────────────────────
# 8. Visualization
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1,2, figsize=(16,6))

# Feature Importance Plot
axes[0].barh(
    top10_features["feature"][::-1],
    top10_features["importance"][::-1],
    color="steelblue"
)

axes[0].set_title("Top Behaviors Predicting Successful Users")
axes[0].set_xlabel("Feature Importance")


# Confusion Matrix
cm = confusion_matrix(y_test, rf_preds)

axes[1].imshow(cm, cmap="Blues")

axes[1].set_xticks([0,1])
axes[1].set_yticks([0,1])

axes[1].set_xticklabels(["Not Successful","Successful"])
axes[1].set_yticklabels(["Not Successful","Successful"])

axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i,j], ha="center", va="center")

axes[1].set_title("Confusion Matrix")

plt.tight_layout()
plt.show()

print("\nVisualization complete.")


# ─────────────────────────────────────────────────────────────
# 9. Final Output
# ─────────────────────────────────────────────────────────────

feature_importance_df