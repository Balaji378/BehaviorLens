import matplotlib.pyplot as plt

# ── Define Long-Term Successful Users ────────────────────────────────────────
# Success metric: Top 20% of users by total_events (engagement level)

# 1. Calculate threshold
p80_threshold = user_features["total_events"].quantile(0.80)

# 2. Label users
user_features["successful_user"] = (
    user_features["total_events"] >= p80_threshold
)

# 3. Count outcomes
n_successful = user_features["successful_user"].sum()
n_not_successful = (~user_features["successful_user"]).sum()
total = len(user_features)

# ── Display Results ──────────────────────────────────────────────────────────
print("=" * 60)
print("        Long-Term User Success Definition")
print("=" * 60)
print(f"  80th Percentile Threshold : {p80_threshold:,.2f} events")
print(f"  Successful Users  (top 20%): {n_successful:,} ({n_successful/total*100:.1f}%)")
print(f"  Other Users               : {n_not_successful:,} ({n_not_successful/total*100:.1f}%)")
print(f"  Total Users               : {total:,}")
print("=" * 60)

# ── Visualize Class Distribution ─────────────────────────────────────────────
plt.figure(figsize=(6,4))

user_features["successful_user"].value_counts().plot(
    kind="bar",
    color=["#4CAF50", "#90CAF9"]
)

plt.title("Successful vs Non-Successful Users")
plt.xlabel("Successful User")
plt.ylabel("Number of Users")

plt.xticks([0,1], ["False","True"], rotation=0)

plt.tight_layout()
plt.show()

# ── Rationale ────────────────────────────────────────────────────────────────
print("""
Why total_events is a strong proxy for long-term success:
---------------------------------------------------------
1. Engagement depth — users performing more actions are actively interacting
   with the product rather than just exploring.

2. Workflow completion — higher event counts often indicate that users are
   executing real workflows instead of isolated actions.

3. Behavioral signal — events capture actual product usage behavior,
   which is directly relevant when identifying successful user journeys.

4. Relative threshold — using the 80th percentile ensures that the definition
   adapts to the dataset rather than relying on an arbitrary cutoff.

5. Power users — in most software products, a small portion of users
   performs the majority of meaningful activity.
---------------------------------------------------------
""")

# Return dataframe for next pipeline steps
user_features