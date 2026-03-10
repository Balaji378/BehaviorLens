import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Zerve dark theme ────────────────────────────────────────────────────────
BG   = "#1D1D20"
FG   = "#fbfbff"
SUB  = "#909094"
GREEN = "#8DE5A1"
RED   = "#FF9F9B"
BLUE  = "#A1C9F4"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   SUB,
    "axes.labelcolor":  FG,
    "xtick.color":      FG,
    "ytick.color":      FG,
    "text.color":       FG,
    "grid.color":       "#333338",
    "legend.facecolor": "#2a2a2e",
    "legend.edgecolor": SUB,
    "legend.labelcolor": FG,
})

# ── Prepare dataset ────────────────────────────────────────────────────────
vis_df = user_features.copy()

vis_successful = vis_df[vis_df["successful_user"] == True]
vis_not_successful = vis_df[vis_df["successful_user"] == False]

success_label = "Successful"
non_success_label = "Not Successful"

vis_colors = {True: GREEN, False: RED}

# ── Figure layout ──────────────────────────────────────────────────────────
vis_fig, vis_axes = plt.subplots(2, 3, figsize=(18, 12))
vis_fig.patch.set_facecolor(BG)

vis_fig.suptitle(
    "User Engagement Patterns: Successful vs Non-Successful Users",
    fontsize=16,
    fontweight="bold",
    color=FG,
    y=1.01
)

# ── 1. Sessions boxplot ─────────────────────────────────────────────────────
_ax = vis_axes[0, 0]

_data = [
    vis_successful["number_of_sessions"],
    vis_not_successful["number_of_sessions"]
]

_bp = _ax.boxplot(_data, labels=[success_label, non_success_label],
                  patch_artist=True, showfliers=False,
                  medianprops=dict(color=FG, linewidth=2),
                  whiskerprops=dict(color=SUB),
                  capprops=dict(color=SUB))

_bp["boxes"][0].set_facecolor(vis_colors[True])
_bp["boxes"][1].set_facecolor(vis_colors[False])

_ax.set_title("Number of Sessions by Success", color=FG, fontweight="bold")
_ax.set_ylabel("Sessions", color=FG)

# ── 2. Total events boxplot ─────────────────────────────────────────────────
_ax = vis_axes[0, 1]

_data = [
    vis_successful["total_events"],
    vis_not_successful["total_events"]
]

_bp = _ax.boxplot(_data, labels=[success_label, non_success_label],
                  patch_artist=True, showfliers=False,
                  medianprops=dict(color=FG, linewidth=2),
                  whiskerprops=dict(color=SUB),
                  capprops=dict(color=SUB))

_bp["boxes"][0].set_facecolor(vis_colors[True])
_bp["boxes"][1].set_facecolor(vis_colors[False])

_ax.set_title("Total Events by Success", color=FG, fontweight="bold")
_ax.set_ylabel("Events", color=FG)

# ── 3. Event diversity boxplot ─────────────────────────────────────────────
_ax = vis_axes[0, 2]

_data = [
    vis_successful["number_of_event_types"],
    vis_not_successful["number_of_event_types"]
]

_bp = _ax.boxplot(_data, labels=[success_label, non_success_label],
                  patch_artist=True, showfliers=False,
                  medianprops=dict(color=FG, linewidth=2),
                  whiskerprops=dict(color=SUB),
                  capprops=dict(color=SUB))

_bp["boxes"][0].set_facecolor(vis_colors[True])
_bp["boxes"][1].set_facecolor(vis_colors[False])

_ax.set_title("Feature Diversity by Success", color=FG, fontweight="bold")
_ax.set_ylabel("Unique Event Types", color=FG)

# ── 4. Credits distribution ────────────────────────────────────────────────
_ax = vis_axes[1, 0]

_cap = vis_df["total_credits_used"].quantile(0.99)

_ax.hist(
    vis_successful["total_credits_used"].clip(upper=_cap),
    bins=40,
    alpha=0.7,
    label=success_label,
    color=vis_colors[True]
)
_ax.hist(
    vis_not_successful["total_credits_used"].clip(upper=_cap),
    bins=40,
    alpha=0.7,
    label=non_success_label,
    color=vis_colors[False]
)

_ax.set_title("Credits Usage Distribution", color=FG, fontweight="bold")
_ax.set_xlabel("Credits Used", color=FG)
_ax.set_ylabel("Users", color=FG)
_ax.legend()

# ── 5. Scatter: sessions vs events ─────────────────────────────────────────
_ax = vis_axes[1, 1]

for _label, _group in vis_df.groupby("successful_user"):
    _ax.scatter(
        _group["number_of_sessions"],
        _group["total_events"],
        alpha=0.35,
        s=15,
        c=vis_colors[_label],
        label=success_label if _label else non_success_label
    )

_ax.set_title("Sessions vs Events", color=FG, fontweight="bold")
_ax.set_xlabel("Sessions", color=FG)
_ax.set_ylabel("Total Events", color=FG)
_ax.legend()

# ── 6. Success rate by sessions ────────────────────────────────────────────
_ax = vis_axes[1, 2]

_bins = np.linspace(
    vis_df["number_of_sessions"].min(),
    vis_df["number_of_sessions"].quantile(0.95),
    10
)

vis_df_bucketed = vis_df.copy()
vis_df_bucketed["session_bucket"] = pd.cut(vis_df_bucketed["number_of_sessions"], _bins)

_success_by_session = vis_df_bucketed.groupby("session_bucket", observed=True)["successful_user"].mean()

_ax.plot(
    range(len(_success_by_session)),
    _success_by_session.values,
    marker="o",
    color=BLUE,
    linewidth=2,
    markersize=7
)

# Format x-tick labels from bucket intervals
_tick_labels = [str(b) for b in _success_by_session.index]
_ax.set_xticks(range(len(_success_by_session)))
_ax.set_xticklabels(_tick_labels, rotation=30, ha="right", fontsize=7)

_ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
_ax.set_title("Success Rate vs Session Activity", color=FG, fontweight="bold")
_ax.set_ylabel("Success Rate", color=FG)
_ax.set_xlabel("Session Bucket", color=FG)

plt.tight_layout(pad=2.0)
plt.show()

# ── Summary stats ──────────────────────────────────────────────────────────
print("\nENGAGEMENT SUMMARY\n" + "="*50)

for _col in [
    "number_of_sessions",
    "total_events",
    "number_of_event_types",
    "total_credits_used"
]:
    print(f"\n{_col}")
    print(f"  Successful     → median {vis_successful[_col].median():.1f} | mean {vis_successful[_col].mean():.1f}")
    print(f"  Not Successful → median {vis_not_successful[_col].median():.1f} | mean {vis_not_successful[_col].mean():.1f}")
