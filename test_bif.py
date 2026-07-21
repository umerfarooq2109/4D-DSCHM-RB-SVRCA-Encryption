import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import itertools
import os

# ---------------------------------------------------------------
# Custom "ocean teal" colormap to match the reference image
# (deep teal-blue base with lighter cyan/white highlights)
# ---------------------------------------------------------------
ocean_colors = [
    "#062b3d",  # deep navy
    "#0b4f6c",  # dark teal
    "#137a8f",  # mid teal
    "#3aa9c0",  # teal-cyan
    "#8fd7e0",  # light cyan
    "#e8f8fa",  # near white highlight
]
ocean_cmap = LinearSegmentedColormap.from_list("ocean_teal", ocean_colors, N=256)

# ---------------------------------------------------------------
# Map parameters (identical to prior verified scripts)
# ---------------------------------------------------------------
b, c, d, e, f, g, h = 0.354, 1.287, 0.219, 1.543, 0.412, 1.109, 0.331

def map_step(x, y, z, w, a):
    xn = (np.sin(a * y) + b * np.cos(x) + w) % 1.0
    yn = (np.sin(c * x) + d * np.cos(y)) % 1.0
    zn = (np.sin(e * z) + f * np.cos(w)) % 1.0
    wn = (np.sin(g * w) + h * np.cos(z)) % 1.0
    return xn, yn, zn, wn

# ---------------------------------------------------------------
# Sweep settings
# ---------------------------------------------------------------
start_a, end_a = 0.5, 3.0
alpha_steps = 260      # number of alpha columns
transient   = 800       # discarded iterations per alpha
sampled     = 90        # kept iterations per alpha

alphas = np.linspace(start_a, end_a, alpha_steps)

total_pts = alpha_steps * sampled
A = np.empty(total_pts)
X = np.empty(total_pts)
Y = np.empty(total_pts)
Z = np.empty(total_pts)
W = np.empty(total_pts)

ptr = 0
print("Generating trajectory data across alpha sweep...")
for a in alphas:
    x, y, z, w = 0.1, 0.2, 0.3, 0.4
    for _ in range(transient):
        x, y, z, w = map_step(x, y, z, w, a)
    for _ in range(sampled):
        x, y, z, w = map_step(x, y, z, w, a)
        A[ptr] = a
        X[ptr] = x
        Y[ptr] = y
        Z[ptr] = z
        W[ptr] = w
        ptr += 1

print("Data generation complete:", total_pts, "points.")

state = {'x': X, 'y': Y, 'z': Z, 'w': W}
pairs = list(itertools.combinations(['x', 'y', 'z', 'w'], 2))  # 6 pairs

os.makedirs("results/plots", exist_ok=True)

# ---------------------------------------------------------------
# Build 2x3 grid of 3D bifurcation plots: axes = (alpha, var_i, var_j)
# color mapped by the alpha value using the ocean_teal colormap
# ---------------------------------------------------------------
fig = plt.figure(figsize=(24, 15), dpi=300, facecolor='white')

for i, (vi, vj) in enumerate(pairs):
    ax = fig.add_subplot(2, 3, i + 1, projection='3d')
    ax.set_facecolor('white')

    vi_data = state[vi]
    vj_data = state[vj]

    sc = ax.scatter(
        A, vi_data, vj_data,
        c=A, cmap=ocean_cmap,
        s=0.6, alpha=0.55, linewidths=0,
        depthshade=True
    )

    ax.set_xlabel(r'$\alpha$', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel(f'{vi}$_n$', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_zlabel(f'{vj}$_n$', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title(f'3D Bifurcation: $\\alpha$ - {vi} - {vj}', fontsize=13, fontweight='bold', pad=10)

    ax.set_xlim(start_a, end_a)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=22, azim=-60)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    ax.grid(True, alpha=0.15)

fig.suptitle('3D Bifurcation Diagrams of the 4D-DSCHM (All 6 Pairwise State Projections)',
             fontsize=18, fontweight='bold', y=0.985)

plt.tight_layout(rect=[0, 0, 1, 0.97])
outfile = "results/plots/bifurcation_3d_all_pairs.png"
plt.savefig(outfile, dpi=300, facecolor='white', bbox_inches='tight')
plt.close()
print("Saved combined grid:", outfile)

# ---------------------------------------------------------------
# Also save each of the 6 pairs as its own large standalone
# high-resolution figure
# ---------------------------------------------------------------
for (vi, vj) in pairs:
    fig = plt.figure(figsize=(12, 10), dpi=300, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    vi_data = state[vi]
    vj_data = state[vj]

    ax.scatter(
        A, vi_data, vj_data,
        c=A, cmap=ocean_cmap,
        s=0.8, alpha=0.6, linewidths=0,
        depthshade=True
    )

    ax.set_xlabel(r'Control Parameter $\alpha$', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel(f'State Variable {vi}$_n$', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_zlabel(f'State Variable {vj}$_n$', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_title(f'3D Bifurcation Diagram: $\\alpha$ vs {vi} vs {vj}\n(4D-DSCHM)',
                 fontsize=15, fontweight='bold', pad=16)

    ax.set_xlim(start_a, end_a)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=22, azim=-60)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    outfile = f"results/plots/bifurcation_3d_{vi}{vj}.png"
    plt.savefig(outfile, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print("Saved:", outfile)

print("All 3D bifurcation diagrams generated.")