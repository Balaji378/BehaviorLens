import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ─── Color palette (Zerve design system) ─────────────────────────────────────
BG      = "#1D1D20"
EDGE_C  = "#A1C9F4"
ACCENT  = "#FFB482"
TEXT    = "#fbfbff"
SUB     = "#909094"
WHITE   = "#ffffff"

# ─── Limit to top 15 transitions by count ──────────────────────────────────
top_transitions = transitions.head(15)

if "success_rate" not in top_transitions.columns:
    top_transitions = top_transitions.copy()
    top_transitions["success_rate"] = 0.5

top_transitions_graph = top_transitions.copy()

# ─── Build adjacency / node data ──────────────────────────────────────────
nodes_set = sorted(set(top_transitions_graph["event"]).union(set(top_transitions_graph["next_event"])))
node_idx  = {n: i for i, n in enumerate(nodes_set)}
n_nodes   = len(nodes_set)

counts_src  = top_transitions_graph.groupby("event")["count"].sum().to_dict()
counts_tgt  = top_transitions_graph.groupby("next_event")["count"].sum().to_dict()
node_flow   = {n: counts_src.get(n, 0) + counts_tgt.get(n, 0) for n in nodes_set}
max_flow    = max(node_flow.values()) if node_flow else 1

node_success = {}
for n in nodes_set:
    edges_out = top_transitions_graph[top_transitions_graph["event"] == n]["success_rate"].tolist()
    edges_in  = top_transitions_graph[top_transitions_graph["next_event"] == n]["success_rate"].tolist()
    all_edges = edges_out + edges_in
    node_success[n] = float(np.mean(all_edges)) if all_edges else 0.5

# ─── Shorten event names ──────────────────────────────────────────────────
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
    if len(name) > 24:
        name = name[:22] + "…"
    return name

label_map = {n: _shorten(n) for n in nodes_set}

# ─── Build adjacency matrix ───────────────────────────────────────────────
adj = np.zeros((n_nodes, n_nodes))
for _, row in top_transitions_graph.iterrows():
    u_idx = node_idx[row["event"]]
    v_idx = node_idx[row["next_event"]]
    adj[u_idx, v_idx] = row["count"]
    adj[v_idx, u_idx] = row["count"]  # symmetric for layout

# ─── Kamada-Kawai layout (pure numpy) ────────────────────────────────────
# Compute shortest path distances via Floyd-Warshall
# Use edge weights inverse to counts (closer nodes = more connected)
INF = 1e9
dist_mat = np.full((n_nodes, n_nodes), INF)
np.fill_diagonal(dist_mat, 0.0)

for i in range(n_nodes):
    for j in range(n_nodes):
        if adj[i, j] > 0:
            # Lighter weight for stronger edges → places them closer
            dist_mat[i, j] = 1.0 / (adj[i, j] + 1e-9)

# Floyd-Warshall
for mid in range(n_nodes):
    for src in range(n_nodes):
        for tgt in range(n_nodes):
            if dist_mat[src, mid] + dist_mat[mid, tgt] < dist_mat[src, tgt]:
                dist_mat[src, tgt] = dist_mat[src, mid] + dist_mat[mid, tgt]

# Replace INF with max finite distance * 1.5 for isolated nodes
finite_max = dist_mat[dist_mat < INF].max() if np.any(dist_mat < INF) else 1.0
dist_mat[dist_mat >= INF] = finite_max * 1.5

# Desired distances = actual graph distances scaled to [0.5, 3.5]
d_min, d_max = dist_mat.min(), dist_mat.max()
if d_max > d_min:
    ideal = 0.8 + 2.5 * (dist_mat - d_min) / (d_max - d_min)
else:
    ideal = np.ones_like(dist_mat) * 2.0
np.fill_diagonal(ideal, 0.0)

# Spring stiffness matrix: 1/d^2
k_ij = np.where(ideal > 0, 1.0 / (ideal ** 2), 0.0)

# Initialise on a circle, more space than before
angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
pos_arr = np.column_stack([np.cos(angles), np.sin(angles)]) * 3.0

# Kamada-Kawai gradient descent
for _outer in range(500):
    # Find node with max gradient magnitude
    grads = []
    for i in range(n_nodes):
        grad = np.zeros(2)
        for j in range(n_nodes):
            if i == j:
                continue
            diff = pos_arr[i] - pos_arr[j]
            dist = max(np.linalg.norm(diff), 1e-6)
            coeff = k_ij[i, j] * (1.0 - ideal[i, j] / dist)
            grad += coeff * diff
        grads.append(np.linalg.norm(grad))
    pick = int(np.argmax(grads))
    if grads[pick] < 1e-4:
        break

    # Newton sub-step for node 'pick'
    for _inner in range(10):
        grad   = np.zeros(2)
        H      = np.zeros((2, 2))
        for j in range(n_nodes):
            if j == pick:
                continue
            diff  = pos_arr[pick] - pos_arr[j]
            dist  = max(np.linalg.norm(diff), 1e-6)
            dist3 = dist ** 3
            coeff = k_ij[pick, j]
            grad += coeff * (1.0 - ideal[pick, j] / dist) * diff
            H[0, 0] += coeff * (1.0 - ideal[pick, j] * diff[1]**2 / dist3)
            H[1, 1] += coeff * (1.0 - ideal[pick, j] * diff[0]**2 / dist3)
            H[0, 1] += coeff * ideal[pick, j] * diff[0] * diff[1] / dist3
            H[1, 0]  = H[0, 1]
        # Solve H * delta = -grad
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if abs(det) < 1e-12:
            break
        inv_H = np.array([[H[1, 1], -H[0, 1]], [-H[1, 0], H[0, 0]]]) / det
        delta = -inv_H @ grad
        pos_arr[pick] += delta
        if np.linalg.norm(delta) < 1e-4:
            break

# Scale to a nice range
all_pts = pos_arr
c = all_pts.mean(axis=0)
pos_arr -= c
s = max(np.abs(pos_arr).max(), 1e-6)
pos_arr = pos_arr / s * 3.8

pos = {n: pos_arr[node_idx[n]] for n in nodes_set}

# ─── Figure (larger canvas) ───────────────────────────────────────────────
workflow_fig, workflow_ax = plt.subplots(figsize=(26, 18))
workflow_fig.patch.set_facecolor(BG)
workflow_ax.set_facecolor(BG)

max_count = top_transitions_graph["count"].max()

# ─── Draw curved edges with arrows ───────────────────────────────────────
for _, row in top_transitions_graph.iterrows():
    u, v   = row["event"], row["next_event"]
    if u not in pos or v not in pos:
        continue
    p0, p1 = pos[u], pos[v]
    weight = row["count"] / max_count
    lw     = 1.0 + 5.0 * weight
    alpha  = 0.35 + 0.55 * weight

    base_r = 0.18
    nr_u   = base_r + 0.22 * (node_flow.get(u, 0) / max_flow)
    nr_v   = base_r + 0.22 * (node_flow.get(v, 0) / max_flow)

    delta = p1 - p0
    dist  = np.linalg.norm(delta)
    if dist < 1e-6:
        continue
    unit = delta / dist
    p0s  = p0 + unit * nr_u
    p1e  = p1 - unit * nr_v

    workflow_ax.annotate(
        "",
        xy=p1e, xytext=p0s,
        arrowprops=dict(
            arrowstyle="-|>",
            color=EDGE_C,
            lw=lw,
            alpha=alpha,
            connectionstyle="arc3,rad=0.28",
            mutation_scale=22,
        ),
    )

# ─── Draw nodes ──────────────────────────────────────────────────────────
cmap      = cm.RdYlGn
cnorm     = mcolors.Normalize(vmin=0, vmax=1)
node_list = list(nodes_set)

for n in node_list:
    nx_pos, ny_pos = pos[n]
    flow    = node_flow.get(n, 0) / max_flow
    radius  = 0.18 + 0.22 * flow
    color   = cmap(cnorm(node_success.get(n, 0.5)))
    circle  = plt.Circle(
        (nx_pos, ny_pos), radius,
        color=color, alpha=0.92,
        linewidth=1.8, edgecolor=TEXT, zorder=3
    )
    workflow_ax.add_patch(circle)

    workflow_ax.text(
        nx_pos, ny_pos, label_map[n],
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=WHITE,
        path_effects=[pe.withStroke(linewidth=3, foreground=BG)],
        zorder=4,
    )

# ─── Highlight most important node ───────────────────────────────────────
main_node = max(node_flow, key=node_flow.get)
mx, my    = pos[main_node]
mflow     = node_flow[main_node] / max_flow
mradius   = (0.18 + 0.22 * mflow) * 1.45
highlight = plt.Circle(
    (mx, my), mradius,
    fill=False, edgecolor=ACCENT, linewidth=2.5, zorder=5
)
workflow_ax.add_patch(highlight)

workflow_ax.text(
    mx, my + mradius + 0.15,
    "★ highest traffic",
    ha="center", va="bottom",
    fontsize=9, color=ACCENT,
    path_effects=[pe.withStroke(linewidth=2, foreground=BG)],
    zorder=6,
)

# ─── Colorbar ─────────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
sm.set_array([])
cbar = workflow_fig.colorbar(sm, ax=workflow_ax, fraction=0.022, pad=0.01)
cbar.set_label("Avg Success Rate", color=TEXT, fontsize=11)
cbar.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT)

# ─── Title ─────────────────────────────────────────────────────────────────
workflow_ax.set_title(
    "User Workflow Network — Top 15 Event Transitions",
    color=TEXT, fontsize=20, fontweight="bold", pad=20,
)

# Auto-compute bounds with padding
all_xy = np.array(list(pos.values()))
pad    = 0.9
workflow_ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
workflow_ax.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
workflow_ax.set_aspect("equal")
workflow_ax.axis("off")

# Legend
size_patch_high = mpatches.Patch(color="#aaaaaa", label="Large circle = high traffic")
size_patch_low  = mpatches.Patch(color="#555555", label="Small circle = low traffic")
workflow_ax.legend(
    handles=[size_patch_high, size_patch_low],
    loc="lower left", framealpha=0.25,
    facecolor=BG, edgecolor=SUB,
    labelcolor=TEXT, fontsize=9,
)

plt.tight_layout()
plt.savefig(
    "workflow_network_graph.png",
    dpi=160, bbox_inches="tight", facecolor=BG
)
plt.show()

print(f"Workflow network graph saved — {n_nodes} nodes, {len(top_transitions)} transitions. Layout: Kamada-Kawai (pure numpy).")
