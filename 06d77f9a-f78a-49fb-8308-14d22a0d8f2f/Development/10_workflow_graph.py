import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ─── Color palette ─────────────────────────────────────────────────────────
BG      = "#0d1117"
EDGE_C  = "#58a6ff"
ACCENT  = "#f78166"
TEXT    = "#e6edf3"
WHITE   = "#ffffff"

# ─── Ensure success_rate exists ────────────────────────────────────────────
if "success_rate" not in top20.columns:
    top20["success_rate"] = 0.5

top20_graph = top20.copy()

# ─── Build adjacency / node data from top20 ────────────────────────────────
nodes_set = sorted(set(top20_graph["event"]).union(set(top20_graph["next_event"])))
node_idx  = {n: i for i, n in enumerate(nodes_set)}
n_nodes   = len(nodes_set)

# Node flow (sum of counts as source or target)
counts_src  = top20_graph.groupby("event")["count"].sum().to_dict()
counts_tgt  = top20_graph.groupby("next_event")["count"].sum().to_dict()
node_flow   = {n: counts_src.get(n, 0) + counts_tgt.get(n, 0) for n in nodes_set}
max_flow    = max(node_flow.values()) if node_flow else 1

# Node avg success rate (across all adjacent edges)
node_success = {}
for n in nodes_set:
    edges_out = top20_graph[top20_graph["event"] == n]["success_rate"].tolist()
    edges_in  = top20_graph[top20_graph["next_event"] == n]["success_rate"].tolist()
    all_edges = edges_out + edges_in
    node_success[n] = float(np.mean(all_edges)) if all_edges else 0.5

# ─── Shorten event names ───────────────────────────────────────────────────
def _shorten(name):
    replacements = {
        "agent_tool_call_create_block_tool": "agent:create_block",
        "agent_tool_call_run_block_tool":    "agent:run_block",
        "agent_tool_call_get_block_tool":    "agent:get_block",
        "agent_tool_call_edit_block_tool":   "agent:edit_block",
        "addon_credits_used":                "addon:credits_used",
    }
    for long, short in replacements.items():
        name = name.replace(long, short)
    if len(name) > 22:
        name = name[:20] + "…"
    return name

label_map = {n: _shorten(n) for n in nodes_set}

# ─── Force-directed layout (simple spring layout) ─────────────────────────
np.random.seed(42)
pos = {n: np.array([np.cos(2 * np.pi * i / n_nodes),
                    np.sin(2 * np.pi * i / n_nodes)], dtype=float)
       for i, n in enumerate(nodes_set)}

# Run simplified force-directed iterations
k = 2.8 / np.sqrt(n_nodes) if n_nodes > 0 else 1.0
for _ in range(80):
    disp = {n: np.zeros(2) for n in nodes_set}
    nlist = list(nodes_set)
    # Repulsive forces
    for i in range(len(nlist)):
        for j in range(i + 1, len(nlist)):
            u, v = nlist[i], nlist[j]
            delta = pos[u] - pos[v]
            dist  = max(np.linalg.norm(delta), 0.01)
            force = k ** 2 / dist
            disp[u] += force * delta / dist
            disp[v] -= force * delta / dist
    # Attractive forces along edges
    for _, row in top20_graph.iterrows():
        u, v  = row["event"], row["next_event"]
        if u not in pos or v not in pos:
            continue
        delta = pos[u] - pos[v]
        dist  = max(np.linalg.norm(delta), 0.01)
        force = dist ** 2 / k
        disp[u] -= force * delta / dist
        disp[v] += force * delta / dist
    # Apply displacement with cooling
    temp = 0.1
    for n in nodes_set:
        d = np.linalg.norm(disp[n])
        if d > 0:
            pos[n] += disp[n] / d * min(d, temp)

# Scale to [-2, 2]
all_pts = np.array(list(pos.values()))
scale = max(np.abs(all_pts).max(), 1e-6)
pos = {n: p / scale * 2 for n, p in pos.items()}

# ─── Figure ───────────────────────────────────────────────────────────────
workflow_fig, workflow_ax = plt.subplots(figsize=(20, 14))
workflow_fig.patch.set_facecolor(BG)
workflow_ax.set_facecolor(BG)

# Edge widths
max_count = top20_graph["count"].max()

# Draw edges (arrows)
for _, row in top20_graph.iterrows():
    u, v     = row["event"], row["next_event"]
    if u not in pos or v not in pos:
        continue
    p0, p1   = pos[u], pos[v]
    weight   = row["count"] / max_count
    lw       = 0.5 + 6 * weight
    alpha    = 0.4 + 0.5 * weight

    # Offset arrow endpoints toward center to avoid overlap with node circles
    delta  = p1 - p0
    dist   = np.linalg.norm(delta)
    if dist < 1e-6:
        continue
    unit   = delta / dist
    node_r = 0.05 + 0.12 * (node_flow.get(u, 0) / max_flow)
    p0s    = p0 + unit * node_r
    p1e    = p1 - unit * (0.05 + 0.12 * (node_flow.get(v, 0) / max_flow))

    workflow_ax.annotate(
        "",
        xy=p1e, xytext=p0s,
        arrowprops=dict(
            arrowstyle="-|>",
            color=EDGE_C,
            lw=lw,
            alpha=alpha,
            connectionstyle="arc3,rad=0.12",
            mutation_scale=18,
        ),
    )

# Draw nodes
cmap     = cm.RdYlGn
cnorm    = mcolors.Normalize(vmin=0, vmax=1)
node_list = list(nodes_set)

for n in node_list:
    x, y    = pos[n]
    flow    = node_flow.get(n, 0) / max_flow
    radius  = 0.05 + 0.12 * flow
    color   = cmap(cnorm(node_success.get(n, 0.5)))
    circle  = plt.Circle(
        (x, y), radius,
        color=color, alpha=0.92,
        linewidth=1.4, edgecolor=TEXT, zorder=3
    )
    workflow_ax.add_patch(circle)

    # Label
    workflow_ax.text(
        x, y, label_map[n],
        ha="center", va="center",
        fontsize=7, fontweight="bold", color=WHITE,
        path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)],
        zorder=4
    )

# ─── Highlight most important node ────────────────────────────────────────
main_node = max(node_flow, key=node_flow.get)
mx, my    = pos[main_node]
mflow     = node_flow[main_node] / max_flow
mradius   = (0.05 + 0.12 * mflow) * 1.4
highlight = plt.Circle(
    (mx, my), mradius,
    fill=False, edgecolor=ACCENT, linewidth=2, zorder=5
)
workflow_ax.add_patch(highlight)

# ─── Colorbar ─────────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
sm.set_array([])
cbar = workflow_fig.colorbar(sm, ax=workflow_ax, fraction=0.025, pad=0.01)
cbar.set_label("Avg Success Rate", color=TEXT)
cbar.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT)

# ─── Title ────────────────────────────────────────────────────────────────
workflow_ax.set_title(
    "User Workflow Network — Top Event Transitions",
    color=TEXT, fontsize=17, fontweight="bold", pad=16,
)

workflow_ax.set_xlim(-2.5, 2.5)
workflow_ax.set_ylim(-2.5, 2.5)
workflow_ax.set_aspect("equal")
workflow_ax.axis("off")

plt.tight_layout()
plt.savefig(
    "workflow_network_graph.png",
    dpi=160, bbox_inches="tight", facecolor=BG
)
plt.show()

print("Workflow network graph saved.")
