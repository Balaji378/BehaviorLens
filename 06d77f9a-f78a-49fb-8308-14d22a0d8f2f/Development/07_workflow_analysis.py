import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. Prepare event timeline ───────────────────────────────────────────────
wf = cleaned_df[['distinct_id','event','timestamp']].copy()

wf['timestamp'] = pd.to_datetime(wf['timestamp'], utc=True, errors='coerce')
wf = wf.dropna(subset=['timestamp','event','distinct_id'])

wf = wf.sort_values(['distinct_id','timestamp']).reset_index(drop=True)

# ── 2. Create next-event transitions ────────────────────────────────────────
wf['next_event'] = wf.groupby('distinct_id')['event'].shift(-1)
wf = wf.dropna(subset=['next_event'])

# ── 3. Attach success label ─────────────────────────────────────────────────
wf = wf.merge(
    user_features[['distinct_id','successful_user']],
    on='distinct_id',
    how='left'
)

# ── 4. Compute transition frequencies ───────────────────────────────────────
transitions = (
    wf.groupby(['event','next_event'])
    .size()
    .reset_index(name='count')
)

# Remove self loops
transitions = transitions[transitions['event'] != transitions['next_event']]

# Remove very rare transitions
min_transition_count = 15
transitions = transitions[transitions['count'] >= min_transition_count]

transitions = transitions.sort_values('count', ascending=False).reset_index(drop=True)

# ── 5. Compute success rate for each transition ─────────────────────────────
transition_success = (
    wf.groupby(['event','next_event'])['successful_user']
    .mean()
    .reset_index(name='success_rate')
)

transitions = transitions.merge(
    transition_success,
    on=['event','next_event'],
    how='left'
)

# ── 6. Top transitions ──────────────────────────────────────────────────────
top20 = transitions.head(20).copy()

top20['transition'] = top20['event'] + " → " + top20['next_event']
top20['pct_of_all'] = (
    top20['count'] / transitions['count'].sum() * 100
).round(2)

# ── 7. Print results ────────────────────────────────────────────────────────
print("=" * 70)
print("TOP WORKFLOWS ASSOCIATED WITH USER BEHAVIOR")
print("=" * 70)

for i, row in top20.iterrows():
    print(
        f"{i+1:2d}. {row['event']} → {row['next_event']}  "
        f"(count={row['count']:,}, success_rate={row['success_rate']:.2f})"
    )

# ── 8. Visualisation ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12,8))

bars = ax.barh(
    range(len(top20)),
    top20['count'],
    color="#5DADE2"
)

ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['transition'], fontsize=9)

ax.set_xlabel("Transition Count")
ax.set_title("Top User Workflow Transitions")

ax.invert_yaxis()

for i, val in enumerate(top20['count']):
    ax.text(val + 2, i, f"{val:,}", va='center')

plt.tight_layout()
plt.show()

# ── 9. Highlight workflows with highest success rate ───────────────────────
high_success = transitions.sort_values(
    "success_rate", ascending=False
).head(10)

print("\n" + "=" * 70)
print("WORKFLOWS WITH HIGHEST SUCCESS RATE")
print("=" * 70)

for i, row in high_success.iterrows():
    print(
        f"{row['event']} → {row['next_event']} "
        f"(success_rate={row['success_rate']:.2f}, count={row['count']})"
    )

# ── 10. Export results ──────────────────────────────────────────────────────
transitions.to_csv("workflow_transitions_analysis.csv", index=False)

print("\nTransition dataset saved → workflow_transitions_analysis.csv")

# Return for next blocks
transitions