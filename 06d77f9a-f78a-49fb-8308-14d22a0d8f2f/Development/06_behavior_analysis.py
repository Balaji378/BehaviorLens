import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import mannwhitneyu

# ── 1. Build event frequency matrix (pivot table) ────────────────────────────
event_matrix = (
    cleaned_df
    .groupby(['distinct_id', 'event'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# ── 2. Merge with user_features ──────────────────────────────────────────────
behavior_df = user_features.merge(event_matrix, on='distinct_id', how='left')

# Fill NaNs
event_cols = [c for c in event_matrix.columns if c != 'distinct_id']
behavior_df[event_cols] = behavior_df[event_cols].fillna(0)

# ── 3. Remove very rare events (noise reduction) ─────────────────────────────
min_event_count = 20

valid_event_cols = [
    col for col in event_cols
    if behavior_df[col].sum() >= min_event_count
]

print(f"\nEvents retained after filtering rare events: {len(valid_event_cols)}")

# ── 4. Compute correlations with successful_user ─────────────────────────────
target = behavior_df['successful_user'].astype(int)

event_correlations = (
    behavior_df[valid_event_cols]
    .corrwith(target)
    .dropna()
    .sort_values(ascending=False)
)

# ── 5. Top predictive behaviors ──────────────────────────────────────────────
top15_corr = event_correlations.head(15)

print("\n" + "=" * 60)
print("Top 15 Events Most Positively Correlated with Success")
print("=" * 60)

for rank, (evt, corr) in enumerate(top15_corr.items(), 1):
    print(f"{rank:2d}. {evt:<45s}  r = {corr:.4f}")

# ── 6. Negative correlations (anti-patterns) ─────────────────────────────────
bottom10_corr = event_correlations.tail(10)

print("\n" + "=" * 60)
print("Events Negatively Correlated with Success")
print("=" * 60)

for rank, (evt, corr) in enumerate(bottom10_corr.items(), 1):
    print(f"{rank:2d}. {evt:<45s}  r = {corr:.4f}")

# ── 7. Event success rates ───────────────────────────────────────────────────
event_success_rates_dict = {}

for evt in valid_event_cols:
    users_with_event = behavior_df[behavior_df[evt] > 0]
    if len(users_with_event) > 0:
        sr = users_with_event["successful_user"].mean()
        event_success_rates_dict[evt] = sr

event_success_rates = (
    pd.Series(event_success_rates_dict)
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("Events with Highest Success Rate")
print("=" * 60)
print(event_success_rates.head(10))

# ── 8. Split into successful / non-successful groups ─────────────────────────
successful_users = behavior_df[behavior_df['successful_user'] == True]
nonsuccessful_users = behavior_df[behavior_df['successful_user'] == False]

print(f"\nSuccessful users: {len(successful_users):,}  |  Non-successful: {len(nonsuccessful_users):,}")

# ── 9. Visualization: Top Predictive Behaviors (Zerve theme) ─────────────────
top10_corr = event_correlations.head(10).iloc[::-1]

BG = '#1D1D20'
TEXT = '#fbfbff'
COLORS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF',
          '#ffd400', '#17b26a', '#f04438', '#9467BD', '#C49C94']

bar_colors = COLORS[:len(top10_corr)]

fig_behavior, ax_behavior = plt.subplots(figsize=(11, 6), facecolor=BG)
ax_behavior.set_facecolor(BG)

bars = ax_behavior.barh(
    range(len(top10_corr)),
    top10_corr.values,
    color=bar_colors,
    edgecolor='none',
    height=0.65,
)

# Label bars
for bar_patch, val in zip(bars, top10_corr.values):
    ax_behavior.text(
        val + 0.002,
        bar_patch.get_y() + bar_patch.get_height() / 2,
        f'{val:.3f}',
        va='center',
        ha='left',
        fontsize=9,
        fontweight='bold',
        color=TEXT,
    )

ax_behavior.set_yticks(range(len(top10_corr)))
ax_behavior.set_yticklabels(
    [e.replace('_', ' ').title() for e in top10_corr.index],
    fontsize=10,
    color=TEXT,
)
ax_behavior.tick_params(axis='x', colors=TEXT)

ax_behavior.set_xlabel("Pearson Correlation with Successful User", fontsize=11, color=TEXT)
ax_behavior.set_title(
    "Top 10 User Behaviors That Predict Long-Term Success",
    fontsize=14,
    fontweight='bold',
    color=TEXT,
    pad=14,
)

ax_behavior.axvline(0, color='#909094', linewidth=0.8, linestyle='--')
ax_behavior.set_xlim(0, top10_corr.max() * 1.22)

for spine in ax_behavior.spines.values():
    spine.set_edgecolor('#909094')
ax_behavior.spines[['top', 'right']].set_visible(False)
ax_behavior.grid(axis='x', alpha=0.2, linestyle='--', color='#909094')

plt.tight_layout()
plt.savefig("top10_success_behaviors.png", dpi=150, facecolor=BG, bbox_inches='tight')
plt.show()

print("\nChart saved → top10_success_behaviors.png")

# ── 10. Statistical Significance Tests (Mann–Whitney U) ───────────────────────
behavior_metrics = [
    "number_of_sessions",
    "total_events",
    "number_of_event_types"
]

print("\nStatistical Significance Tests (Mann–Whitney U)")
print("=" * 50)

for m in behavior_metrics:
    s = successful_users[m]
    ns = nonsuccessful_users[m]

    stat, p = mannwhitneyu(s, ns, alternative="two-sided")

    sig = "✓ Significant (p < 0.05)" if p < 0.05 else "✗ Not significant"
    print(f"{m}")
    print(f"  Successful median: {s.median():.1f}  |  Non-successful median: {ns.median():.1f}")
    print(f"  U-statistic: {stat:.2f}, p-value: {p:.5f}  — {sig}")
    print()

# ── Return dataframe for next blocks ─────────────────────────────────────────
behavior_df
