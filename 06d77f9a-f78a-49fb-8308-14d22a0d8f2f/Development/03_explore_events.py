import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

# ── 1. Core metrics ──────────────────────────────────────────────────────────
total_users  = cleaned_df["distinct_id"].nunique()
total_events = len(cleaned_df)

print("=" * 60)
print("OVERVIEW")
print("=" * 60)
print(f"  Total unique users : {total_users:,}")
print(f"  Total events       : {total_events:,}")

# ── 2. Top 15 most frequent events ───────────────────────────────────────────
top_events = (
    cleaned_df["event"]
    .value_counts()
    .head(15)
    .reset_index()
)

top_events.columns = ["Event", "Count"]
top_events["% of Total"] = (top_events["Count"] / total_events * 100).round(2)

print("\n" + "=" * 60)
print("TOP 15 MOST FREQUENT EVENTS")
print("=" * 60)
print(top_events.to_string(index=False))

# ── 3. Top 10 tools ──────────────────────────────────────────────────────────
top_tools = (
    cleaned_df["prop_tool_name"]
    .dropna()
    .value_counts()
    .head(10)
    .reset_index()
)

top_tools.columns = ["Tool", "Count"]
top_tools["% of Tool Events"] = (
    top_tools["Count"] / top_tools["Count"].sum() * 100
).round(2)

print("\n" + "=" * 60)
print("TOP 10 TOOLS (prop_tool_name)")
print("=" * 60)
print(top_tools.to_string(index=False))

# ── 4. Events per user distribution ──────────────────────────────────────────
events_per_user = cleaned_df.groupby("distinct_id").size()

print("\n" + "=" * 60)
print("EVENTS PER USER SUMMARY")
print("=" * 60)
print(events_per_user.describe())

# ── 5. Event diversity per user ──────────────────────────────────────────────
event_diversity = cleaned_df.groupby("distinct_id")["event"].nunique()

print("\n" + "=" * 60)
print("EVENT TYPES PER USER")
print("=" * 60)
print(event_diversity.describe())

# ── 6. Core workflow concentration ───────────────────────────────────────────
core_event_share = top_events.head(5)["Count"].sum() / total_events * 100

print("\n" + "=" * 60)
print("CORE WORKFLOW CONCENTRATION")
print("=" * 60)
print(f"Top 5 events account for {core_event_share:.2f}% of all activity.")

# ── 7. Visualisations ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle(
    "BehaviorLens — Event & Tool Distribution",
    fontsize=16,
    fontweight="bold",
    y=1.01
)

_PALETTE_EVENTS = plt.cm.Blues_r([i / 15 for i in range(15)])
_PALETTE_TOOLS  = plt.cm.Oranges_r([i / 10 for i in range(10)])

# — Bar chart: top events —
ax1 = axes[0]
bars1 = ax1.barh(
    top_events["Event"][::-1],
    top_events["Count"][::-1],
    color=_PALETTE_EVENTS,
    edgecolor="white",
    linewidth=0.5,
)

ax1.set_xlabel("Number of Events", fontsize=11)
ax1.set_title("Top 15 Most Frequent Events", fontsize=13, fontweight="bold")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax1.tick_params(axis="y", labelsize=9)
ax1.spines[["top", "right"]].set_visible(False)

for bar in bars1:
    width = bar.get_width()
    ax1.text(
        width + total_events * 0.003,
        bar.get_y() + bar.get_height() / 2,
        f"{width:,.0f}",
        va="center",
        ha="left",
        fontsize=8
    )

# — Bar chart: top tools —
ax2 = axes[1]
bars2 = ax2.barh(
    top_tools["Tool"][::-1],
    top_tools["Count"][::-1],
    color=_PALETTE_TOOLS,
    edgecolor="white",
    linewidth=0.5,
)

ax2.set_xlabel("Number of Events", fontsize=11)
ax2.set_title(
    "Top 10 Most Used Tools (prop_tool_name)",
    fontsize=13,
    fontweight="bold"
)

ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax2.tick_params(axis="y", labelsize=9)
ax2.spines[["top", "right"]].set_visible(False)

for bar in bars2:
    width = bar.get_width()
    ax2.text(
        width + top_tools["Count"].max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{width:,.0f}",
        va="center",
        ha="left",
        fontsize=8
    )

plt.tight_layout()
plt.show()

# ── 8. Log-transformed distribution of engagement ────────────────────────────
events_per_user = cleaned_df.groupby("distinct_id").size()
log_events = np.log1p(events_per_user)

plt.figure(figsize=(10, 5))
plt.hist(log_events, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.5)
plt.title("Log Distribution of Events per User", fontsize=14, fontweight="bold")
plt.xlabel("log(events + 1)", fontsize=11)
plt.ylabel("Users", fontsize=11)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# ── 9. Interpretation ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("INTERPRETATION — Event Distribution")
print("=" * 60)

top1      = top_events.iloc[0]
top1_pct  = top1["% of Total"]
top3_pct  = top_events.head(3)["% of Total"].sum()

print(f"""
• The most common event is '{top1['Event']}' ({top1_pct:.1f}% of all events),
  indicating it represents a core interaction in the product.

• The top three events together account for {top3_pct:.1f}% of total activity,
  showing that user engagement is concentrated around a few key actions.

• The top five events alone represent {core_event_share:.1f}% of total activity,
  suggesting that most users rely on a small number of primary workflows.

• Event distribution across users shows significant variation in engagement,
  where some users perform only a few actions while others perform hundreds.

• Users with higher event counts and more diverse event usage are likely
  to represent power users who engage more deeply with the platform.
""")
