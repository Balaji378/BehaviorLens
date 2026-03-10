import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

for rank, (event, corr) in enumerate(top15_corr.items(), 1):
    print(f"{rank:2d}. {event:<45s}  r = {corr:.4f}")

# ── 6. Negative correlations (anti-patterns) ─────────────────────────────────
bottom10_corr = event_correlations.tail(10)

print("\n" + "=" * 60)
print("Events Negatively Correlated with Success")
print("=" * 60)

for rank, (event, corr) in enumerate(bottom10_corr.items(), 1):
    print(f"{rank:2d}. {event:<45s}  r = {corr:.4f}")

# ── 7. Event success rates ───────────────────────────────────────────────────
event_success_rates = {}

for event in valid_event_cols:

    users_with_event = behavior_df[behavior_df[event] > 0]

    if len(users_with_event) > 0:
        success_rate = users_with_event["successful_user"].mean()
        event_success_rates[event] = success_rate

event_success_rates = (
    pd.Series(event_success_rates)
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("Events with Highest Success Rate")
print("=" * 60)

print(event_success_rates.head(10))

# ── 8. Visualization: Top Predictive Behaviors ───────────────────────────────
top10_corr = event_correlations.head(10).iloc[::-1]

colors = plt.cm.RdYlGn(
    np.linspace(0.45, 0.85, len(top10_corr))
)

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.barh(
    range(len(top10_corr)),
    top10_corr.values,
    color=colors,
    edgecolor='white',
    linewidth=0.6,
    height=0.65,
)

# Label bars
for bar_patch, val in zip(bars, top10_corr.values):

    ax.text(
        val + 0.002,
        bar_patch.get_y() + bar_patch.get_height() / 2,
        f'{val:.3f}',
        va='center',
        ha='left',
        fontsize=9,
        fontweight='bold',
    )

ax.set_yticks(range(len(top10_corr)))
ax.set_yticklabels(
    [e.replace('_', ' ').title() for e in top10_corr.index],
    fontsize=10,
)

ax.set_xlabel("Pearson Correlation with Successful User", fontsize=11)

ax.set_title(
    "Top 10 User Behaviors That Predict Long-Term Success",
    fontsize=14,
    fontweight='bold',
)

ax.axvline(0, color='#aaaaaa', linewidth=0.8, linestyle='--')

ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

ax.set_xlim(0, top10_corr.max() * 1.22)

ax.spines[['top', 'right']].set_visible(False)

ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()

plt.savefig("top10_success_behaviors.png", dpi=150)

plt.show()

print("\nChart saved → top10_success_behaviors.png")

# Return dataframe for next blocks
behavior_df