import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# ── 1. Prepare features & target ───────────────────────────────────────────
X = behavior_df[valid_event_cols].copy().astype(float)
y = behavior_df["successful_user"].astype(int)

print(f"Dataset shape: {X.shape}")
print(f"""
Class distribution
Successful      : {y.sum()} ({y.mean()*100:.1f}%)
Not Successful  : {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)
""")

# ── 2. Train / test split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── 3. Train Random Forest ──────────────────────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("\nModel trained successfully ✓")

# ── 4. Evaluate performance ─────────────────────────────────────────────────
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, rf_preds)
roc_auc = roc_auc_score(y_test, rf_probs)

print("\n" + "="*50)
print("Random Forest Performance")
print("="*50)

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"ROC-AUC  : {roc_auc:.3f}")

print("\nClassification Report")
print(classification_report(y_test, rf_preds))

# ── 5. Feature Importance ───────────────────────────────────────────────────
feature_importance_df = pd.DataFrame({
    "feature": valid_event_cols,
    "importance": rf_model.feature_importances_
})

feature_importance_df = feature_importance_df.sort_values(
    "importance", ascending=False
).reset_index(drop=True)

top10_features = feature_importance_df.head(10)

print("\nTop 10 Most Predictive Behaviors")
print("-"*50)

for i,row in top10_features.iterrows():
    print(f"{i+1:2d}. {row['feature']}  (importance={row['importance']:.4f})")

# ── 6. Visualization ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1,2, figsize=(16,6))

# Feature importance
axes[0].barh(
    top10_features["feature"][::-1],
    top10_features["importance"][::-1],
    color="steelblue"
)

axes[0].set_title("Top Behaviors Predicting Successful Users")
axes[0].set_xlabel("Feature Importance")

# Confusion matrix
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
        axes[1].text(j,i,cm[i,j],ha="center",va="center")

axes[1].set_title("Confusion Matrix")

plt.tight_layout()
plt.show()

print("\nVisualization complete.")

# Return model outputs
feature_importance_df