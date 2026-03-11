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

# ── 8. Visualisation: Top User Workflow Transitions ─────────────────────────
_BG   = "#1D1D20"
_TEXT = "#fbfbff"
_SUB  = "#909094"

fig, ax = plt.subplots(figsize=(12, 8), facecolor=_BG)
ax.set_facecolor(_BG)

bars = ax.barh(
    range(len(top20)),
    top20['count'],
    color="#A1C9F4",
    edgecolor="none",
    height=0.7
)

ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['transition'], fontsize=9, color=_TEXT)

ax.set_xlabel("Transition Count", color=_SUB, fontsize=10)
ax.set_title("Top 20 User Workflow Transitions", color=_TEXT, fontsize=13, fontweight='bold', pad=12)

ax.invert_yaxis()
ax.tick_params(axis='x', colors=_SUB)
ax.tick_params(axis='y', colors=_TEXT)
for _spine in ax.spines.values():
    _spine.set_edgecolor("#333337")

for i, val in enumerate(top20['count']):
    ax.text(val + top20['count'].max() * 0.01, i, f"{val:,}",
            va='center', ha='left', fontsize=8, color=_TEXT)

ax.set_xlim(0, top20['count'].max() * 1.18)

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

# ── 11. Transition analysis: split by successful_user ───────────────────────
success_trans = (
    wf[wf['successful_user'] == True]
    .groupby(['event', 'next_event'])
    .size()
    .reset_index(name='count')
)
success_trans = success_trans[success_trans['event'] != success_trans['next_event']]
success_trans = success_trans[success_trans['count'] >= min_transition_count]
success_trans = success_trans.sort_values('count', ascending=False).reset_index(drop=True)

nonsuccess_trans = (
    wf[wf['successful_user'] == False]
    .groupby(['event', 'next_event'])
    .size()
    .reset_index(name='count')
)
nonsuccess_trans = nonsuccess_trans[nonsuccess_trans['event'] != nonsuccess_trans['next_event']]
nonsuccess_trans = nonsuccess_trans[nonsuccess_trans['count'] >= min_transition_count]
nonsuccess_trans = nonsuccess_trans.sort_values('count', ascending=False).reset_index(drop=True)

# Rename for downstream clarity
success_transitions = success_trans.copy()
nonsuccess_transitions = nonsuccess_trans.copy()

# ── 11b. Transition success rate: group by event_name and next_event ─────────
transition_success = (
    wf.groupby(['event', 'next_event'])['successful_user']
    .mean()
    .reset_index(name='success_rate')
    .sort_values('success_rate', ascending=False)
)

print("\n" + "=" * 70)
print("TRANSITION SUCCESS RATES (sorted descending)")
print("=" * 70)
print(f"Total transition pairs: {len(transition_success):,}")
print(transition_success.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("TRANSITION ANALYSIS BY USER SUCCESS")
print("=" * 70)
print(f"Successful user transitions (≥{min_transition_count} occurrences): {len(success_transitions):,} unique pairs")
print(f"Non-successful user transitions (≥{min_transition_count} occurrences): {len(nonsuccess_transitions):,} unique pairs")

print("\nTop 10 transitions — SUCCESSFUL users:")
print(success_transitions.head(10).to_string(index=False))

print("\nTop 10 transitions — NON-SUCCESSFUL users:")
print(nonsuccess_transitions.head(10).to_string(index=False))

# ── 12. Successful users transitions chart ──────────────────────────────────
_COLOR_SUCCESS = "#8DE5A1"   # green  — successful users
_N = 15

_s_top = success_transitions.head(_N).copy()
_s_top['label'] = _s_top['event'] + " → " + _s_top['next_event']

transitions_comparison_fig, _ax_s = plt.subplots(figsize=(12, 8), facecolor=_BG)
_ax_s.set_facecolor(_BG)

_ax_s.barh(
    range(len(_s_top)),
    _s_top['count'],
    color=_COLOR_SUCCESS,
    edgecolor="none",
    height=0.7
)
_ax_s.set_yticks(range(len(_s_top)))
_ax_s.set_yticklabels(_s_top['label'], fontsize=9, color=_TEXT)
_ax_s.invert_yaxis()
_ax_s.set_xlabel("Transition Count", color=_SUB, fontsize=10)
_ax_s.set_title("✅  Top Event Transitions — Successful Users", color=_COLOR_SUCCESS, fontsize=13, fontweight='bold', pad=12)
_ax_s.tick_params(axis='x', colors=_SUB)
_ax_s.tick_params(axis='y', colors=_TEXT)
for _spine in _ax_s.spines.values():
    _spine.set_edgecolor("#333337")

# Value labels
for _idx, _v in enumerate(_s_top['count']):
    _ax_s.text(_v + _s_top['count'].max() * 0.01, _idx, f"{_v:,}",
               va='center', ha='left', fontsize=8, color=_TEXT)
_ax_s.set_xlim(0, _s_top['count'].max() * 1.18)

plt.tight_layout()
plt.show()

# Return for next blocks
transitions
