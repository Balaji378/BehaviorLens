import pandas as pd

# ── 1. Build user-level behavioral features ───────────────────────────────────
user_features = (
    cleaned_df
    .groupby("distinct_id")
    .agg(
        total_events=("event", "count"),
        total_credits_used=("prop_credits_used", "sum"),
        number_of_sessions=("prop_$session_id", "nunique"),
        number_of_event_types=("event", "nunique"),
        first_activity=("timestamp", "min"),
        last_activity=("timestamp", "max"),
    )
    .reset_index()
)

# ── 2. Activity span ─────────────────────────────────────────────────────────
user_features["days_active"] = (
    user_features["last_activity"] - user_features["first_activity"]
).dt.total_seconds() / 86400

# ── 3. Derived behavioral metrics ────────────────────────────────────────────

# average credits per event
user_features["avg_credits_per_event"] = (
    user_features["total_credits_used"] /
    user_features["total_events"].replace(0, 1)
)

# events per session
user_features["events_per_session"] = (
    user_features["total_events"] /
    user_features["number_of_sessions"].replace(0, 1)
)

# event diversity ratio
user_features["event_diversity_ratio"] = (
    user_features["number_of_event_types"] /
    user_features["total_events"].replace(0, 1)
)

# activity span indicator
user_features["active_multiple_days"] = user_features["days_active"] > 1

# ── 4. Summary stats ─────────────────────────────────────────────────────────
print("=" * 60)
print("USER FEATURES DATASET")
print("=" * 60)
print(f"Shape: {user_features.shape[0]:,} users × {user_features.shape[1]} features")

print("\nSample rows:")
print(user_features.head(10).to_string(index=False))

# ── 5. Feature summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FEATURE SUMMARY")
print("=" * 60)

print(user_features.describe().to_string())

# ── 6. Key behavioral insights preview ───────────────────────────────────────
print("\n" + "=" * 60)
print("ENGAGEMENT INSIGHTS")
print("=" * 60)

print(f"""
Average events per user: {user_features['total_events'].mean():.2f}
Median events per user : {user_features['total_events'].median():.2f}

Average sessions per user: {user_features['number_of_sessions'].mean():.2f}

Users active multiple days: 
{user_features['active_multiple_days'].sum():,} / {len(user_features):,}
""")

user_features