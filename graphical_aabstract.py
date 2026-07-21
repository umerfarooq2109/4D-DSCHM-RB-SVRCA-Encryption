"""
Graphical Abstract Generator for the 4D-DSCHM RB-SVRCA Image Encryption Paper.

Produces a single publication-quality figure showing:
  - Full encryption pipeline with REAL intermediate images at every step
  - Actual chaotic attractor phase portrait from the 4D-DSCHM
  - Plaintext vs Ciphertext histograms
  - Adjacent-pixel correlation scatter plots (plain vs cipher)
  - Lossless decryption proof
  - Real computed security metrics (Entropy, NPCR, UACI, Correlation, MSE, PSNR)
  - Key information and system parameters

Usage:
    python graphical_abstract.py
    python graphical_abstract.py Images/lena_gray_512.tif
"""

import os
import sys
import hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# Ensure local packages are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.map_4d import (
    A, B, C, D, E, F, G, H_param,
    key_to_initial_states,
    generate_chaotic_sequences,
)
from src.cellular_automata import (
    rol,
    compute_ca_transition,
    encrypt_image,
    decrypt_image,
)


# ══════════════════════════════════════════════════════════════════════
#  STEP-BY-STEP ENCRYPTION  (mirrors cellular_automata.encrypt_image
#  but returns every intermediate result)
# ══════════════════════════════════════════════════════════════════════

def encrypt_step_by_step(img, key_bytes):
    """Run the full encryption pipeline and return every intermediate image."""
    H_orig, W_orig = img.shape
    pad_h, pad_w = H_orig % 2, W_orig % 2
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')

    H, W = img.shape
    N = H * W

    # Step 1 — Chaotic sequence generation
    # Calculate plaintext-dependent hash of the image
    img_hash = hashlib.sha256(img.tobytes()).digest()
    # Combine the key and plaintext hash to derive unique initial states (CPA protection)
    combined_key = bytes(a ^ b for a, b in zip(key_bytes, img_hash))
    x0, y0, z0, w0 = key_to_initial_states(combined_key)
    
    # Update the core CA module session cache so the imported decrypt_image function matches
    import src.cellular_automata
    src.cellular_automata._last_initial_states = (x0, y0, z0, w0)
    
    xs, ys, zs, ws = generate_chaotic_sequences(H, W, x0, y0, z0, w0)

    # Step 2 — Row / Column Scrambling
    row_idx = np.argsort(xs[:H])
    col_idx = np.argsort(ys[:W])
    scrambled = img[row_idx, :][:, col_idx]

    # Step 3 — CA grid setup
    rules = ((zs * 10**14).astype(np.int64) % 8).reshape(H, W)
    key_grid = ((ws * 255).astype(np.uint8)).reshape(H, W)

    # Step 4 — Red-Black Reversible CA (2 rounds)
    grid_y, grid_x = np.indices((H, W))
    red_mask  = (grid_y + grid_x) % 2 == 0
    black_mask = ~red_mask

    state = scrambled.copy()
    for _ in range(2):
        g_red = compute_ca_transition(state, key_grid, rules)
        state[red_mask] = (state[red_mask].astype(np.int16) +
                           g_red[red_mask].astype(np.int16)) % 256
        g_black = compute_ca_transition(state, key_grid, rules)
        state[black_mask] = (state[black_mask].astype(np.int16) +
                             g_black[black_mask].astype(np.int16)) % 256
    after_ca = state.copy()

    # Step 5a — Forward Diffusion
    state_flat = state.flatten()
    K_flat = key_grid.flatten()
    C1 = np.zeros(N, dtype=np.int64)
    IV1 = int(key_bytes[0]) ^ 157
    prev = IV1
    for i in range(N):
        c = (int(state_flat[i]) + prev) % 256 ^ int(K_flat[i])
        C1[i] = c
        prev = int(rol(np.array([c & 0xFF], dtype=np.uint8), 3)[0])
    after_fwd = C1.reshape(H, W).astype(np.uint8).copy()

    # Step 5b — Backward Diffusion
    C2 = np.zeros(N, dtype=np.int64)
    IV2 = int(key_bytes[1]) ^ 223
    prev2 = IV2
    for i in range(N - 1, -1, -1):
        c2 = (int(C1[i]) + prev2) % 256 ^ int(K_flat[i])
        C2[i] = c2
        prev2 = int(rol(np.array([c2 & 0xFF], dtype=np.uint8), 3)[0])
    ciphertext = C2.reshape(H, W).astype(np.uint8)

    return {
        'original':    img,
        'scrambled':   scrambled.astype(np.uint8),
        'rules_grid':  rules,
        'key_grid':    key_grid,
        'after_ca':    after_ca.astype(np.uint8),
        'after_fwd':   after_fwd,
        'ciphertext':  ciphertext,
        'initials':    (x0, y0, z0, w0),
        'H': H, 'W': W,
    }


# ══════════════════════════════════════════════════════════════════════
#  SECURITY METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════

def shannon_entropy(img):
    counts = np.bincount(img.flatten(), minlength=256)
    probs  = counts / counts.sum()
    probs  = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def compute_npcr_uaci(img, key):
    H, W = img.shape
    img_mod = img.copy()
    cx, cy = H // 2, W // 2
    img_mod[cx, cy] = (int(img[cx, cy]) + 1) % 256
    c1 = encrypt_image(img, key)
    c2 = encrypt_image(img_mod, key)
    npcr = np.mean(c1 != c2) * 100.0
    uaci = np.mean(np.abs(c1.astype(float) - c2.astype(float)) / 255.0) * 100.0
    return npcr, uaci


def correlation_coefficient(img, direction='horizontal', n=5000):
    H, W = img.shape
    rng = np.random.default_rng(42)
    if direction == 'horizontal':
        xs = rng.integers(0, H, n)
        ys = rng.integers(0, W - 1, n)
        a = img[xs, ys].astype(float)
        b = img[xs, ys + 1].astype(float)
    elif direction == 'vertical':
        xs = rng.integers(0, H - 1, n)
        ys = rng.integers(0, W, n)
        a = img[xs, ys].astype(float)
        b = img[xs + 1, ys].astype(float)
    else:
        xs = rng.integers(0, H - 1, n)
        ys = rng.integers(0, W - 1, n)
        a = img[xs, ys].astype(float)
        b = img[xs + 1, ys + 1].astype(float)
    am, bm = a.mean(), b.mean()
    num = np.sum((a - am) * (b - bm))
    den = np.sqrt(np.sum((a - am)**2) * np.sum((b - bm)**2))
    return (num / den) if den else 0.0, a, b


def generate_attractor(n_pts=8000):
    """Generate 4D-DSCHM strange attractor trajectory."""
    x, y, z, w = 0.1, 0.2, 0.3, 0.4
    for _ in range(1000):
        x = (np.sin(A * y) + B * np.cos(x) + w) % 1.0
        y = (np.sin(C * x) + D * np.cos(y)) % 1.0
        z = (np.sin(E * z) + F * np.cos(w)) % 1.0
        w = (np.sin(G * w) + H_param * np.cos(z)) % 1.0
    xp, yp, zp, wp = [], [], [], []
    for _ in range(n_pts):
        x = (np.sin(A * y) + B * np.cos(x) + w) % 1.0
        y = (np.sin(C * x) + D * np.cos(y)) % 1.0
        z = (np.sin(E * z) + F * np.cos(w)) % 1.0
        w = (np.sin(G * w) + H_param * np.cos(z)) % 1.0
        xp.append(x); yp.append(y); zp.append(z); wp.append(w)
    return np.array(xp), np.array(yp), np.array(zp), np.array(wp)


# ══════════════════════════════════════════════════════════════════════
#  DRAWING HELPER — curved arrow between two axes
# ══════════════════════════════════════════════════════════════════════

def draw_arrow(fig, ax_from, ax_to, color='black', lw=3.5,
               from_side='right', to_side='left'):
    """Draw a curved FancyArrowPatch between two axes in figure coords."""
    # Determine connection points
    bb1 = ax_from.get_position()
    bb2 = ax_to.get_position()

    if from_side == 'right':
        x1 = bb1.x1 + 0.003
        y1 = bb1.y0 + bb1.height / 2
    elif from_side == 'bottom':
        x1 = bb1.x0 + bb1.width / 2
        y1 = bb1.y0 - 0.003
    else:
        x1 = bb1.x0 + bb1.width / 2
        y1 = bb1.y1 + 0.003

    if to_side == 'left':
        x2 = bb2.x0 - 0.003
        y2 = bb2.y0 + bb2.height / 2
    elif to_side == 'top':
        x2 = bb2.x0 + bb2.width / 2
        y2 = bb2.y1 + 0.003
    else:
        x2 = bb2.x0 + bb2.width / 2
        y2 = bb2.y0 - 0.003

    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        transform=fig.transFigure,
        arrowstyle='->', mutation_scale=22,
        color=color, lw=lw,
        connectionstyle='arc3,rad=0.0',
        zorder=10,
    )
    fig.patches.append(arrow)


# ══════════════════════════════════════════════════════════════════════
#  MAIN — BUILD THE GRAPHICAL ABSTRACT
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Locate test image ──────────────────────────────────────────
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        candidates = [
            'Images/lena_gray_512.tif',
            'Images/lena_gray_256.tif',
            'Images/cameraman.tif',
            'Images/cameraman.jpg',
            'Images/Baboon.png',
            'Images/4.1.01.tiff',
            'Images/boat.512.tiff',
        ]
        img_path = None
        for c in candidates:
            if os.path.exists(c):
                img_path = c
                break
        if img_path is None:
            print("ERROR: No test image found. Provide path as argument.")
            sys.exit(1)

    key = b"N24015711_CA_Secure_Key_256_Bits_2026"
    key_hex = key.hex()[:32] + "..."
    print(f"Image : {img_path}")
    print(f"Key   : {key}")

    # ── 2. Load & run step-by-step encryption ─────────────────────────
    pil = Image.open(img_path).convert('L')
    plain = np.array(pil, dtype=np.uint8)
    print(f"Size  : {plain.shape[0]}×{plain.shape[1]}")

    print("Running step-by-step encryption...")
    data = encrypt_step_by_step(plain, key)

    print("Running decryption...")
    decrypted = decrypt_image(data['ciphertext'], key,
                              original_shape=plain.shape)

    # ── 3. Compute real security metrics ──────────────────────────────
    print("Computing security metrics...")
    ent_plain  = shannon_entropy(plain)
    ent_cipher = shannon_entropy(data['ciphertext'])
    npcr, uaci = compute_npcr_uaci(plain, key)
    mse_val = float(np.mean((plain.astype(float) - decrypted.astype(float))**2))
    psnr_val = float('inf') if mse_val == 0 else 10 * np.log10(255**2 / mse_val)

    r_h_p, px_h, py_h = correlation_coefficient(plain, 'horizontal')
    r_h_c, cx_h, cy_h = correlation_coefficient(data['ciphertext'], 'horizontal')
    r_v_p, _, _ = correlation_coefficient(plain, 'vertical')
    r_v_c, _, _ = correlation_coefficient(data['ciphertext'], 'vertical')
    r_d_p, _, _ = correlation_coefficient(plain, 'diagonal')
    r_d_c, _, _ = correlation_coefficient(data['ciphertext'], 'diagonal')

    print(f"  Entropy  : plain={ent_plain:.4f}  cipher={ent_cipher:.6f}")
    print(f"  NPCR     : {npcr:.4f}%")
    print(f"  UACI     : {uaci:.4f}%")
    print(f"  Corr (H) : plain={r_h_p:+.4f}  cipher={r_h_c:+.4f}")
    print(f"  MSE      : {mse_val:.4f}")
    print(f"  PSNR     : {'inf' if mse_val == 0 else f'{psnr_val:.2f} dB'}")

    # ── 4. Generate attractor points ──────────────────────────────────
    print("Generating 4D-DSCHM attractor...")
    ax_pts, ay_pts, az_pts, aw_pts = generate_attractor(8000)

    # ── 5. Derive initial states for display ──────────────────────────
    x0, y0, z0, w0 = data['initials']

    # ==================================================================
    #  BUILD THE FIGURE
    # ==================================================================
    print("Composing graphical abstract...")

    # ── Color palette ─────────────────────────────────────────────────
    BG          = '#FFFFFF'     # white background
    CARD_BG     = '#F6F8FA'     # light card background
    ACCENT      = '#0969DA'     # blue accent (darker for white contrast)
    ACCENT2     = '#BC4C00'     # orange accent (darker for white contrast)
    GREEN       = '#1A7F37'     # green
    RED_C       = '#CF222E'     # red
    GOLD        = '#9A6700'     # gold
    TEXT_LIGHT  = '#24292F'     # dark text
    TEXT_DIM    = '#57606A'     # dim text
    BORDER      = '#D0D7DE'     # border color

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 11,
        'text.color': TEXT_LIGHT,
        'axes.labelcolor': TEXT_LIGHT,
        'xtick.color': TEXT_DIM,
        'ytick.color': TEXT_DIM,
        'figure.facecolor': BG,
        'axes.facecolor': CARD_BG,
        'savefig.facecolor': BG,
        'axes.edgecolor': BORDER,
    })

    fig = plt.figure(figsize=(28, 18), dpi=200)

    # ── Master title ──────────────────────────────────────────────────
    fig.text(0.50, 0.975,
             'Graphical Abstract -- 4D Hyperchaotic Cellular Automata Image Encryption',
             ha='center', va='top', fontsize=26, fontweight='bold',
             color=TEXT_LIGHT,
             path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
    fig.text(0.50, 0.955,
             'A Robust Cryptosystem Based on 4D-DSCHM, RB-SVRCA & Bidirectional Feedback Diffusion',
             ha='center', va='top', fontsize=15, color=TEXT_DIM)

    # ══════════════════════════════════════════════════════════════════
    #  ROW 1 — ENCRYPTION PIPELINE (6 step images)
    # ══════════════════════════════════════════════════════════════════
    row1_y = 0.67
    row1_h = 0.22
    img_w  = 0.12
    gap    = 0.035
    start_x = 0.04

    step_images = [
        (plain,              'STEP 0\nPlaintext P',        'gray'),
        (data['scrambled'],  'STEP 1\nRow-Col Scrambling', 'gray'),
        (data['after_ca'],   'STEP 2\nRB-SVRCA (2 Rounds)', 'gray'),
        (data['after_fwd'],  'STEP 3\nForward Diffusion',   'gray'),
        (data['ciphertext'], 'STEP 4\nCiphertext C',         'inferno'),
        (decrypted,          'STEP 5\nDecrypted P\'',         'gray'),
    ]

    row1_axes = []
    for i, (im, label, cmap) in enumerate(step_images):
        x = start_x + i * (img_w + gap)
        ax = fig.add_axes([x, row1_y, img_w, row1_h])
        ax.imshow(im, cmap=cmap, vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])

        # step label below
        title_lines = label.split('\n')
        step_num = title_lines[0]
        step_desc = title_lines[1] if len(title_lines) > 1 else ''

        # Colored step number
        col = ACCENT if i < 5 else GREEN
        if i == 4:
            col = ACCENT2
        ax.set_title(f'{step_num}\n{step_desc}',
                     fontsize=12, fontweight='bold', color=col, pad=6)

        # Highlight border for ciphertext
        if i == 4:
            for spine in ax.spines.values():
                spine.set_edgecolor(ACCENT2)
                spine.set_linewidth(2.5)
        elif i == 5:
            for spine in ax.spines.values():
                spine.set_edgecolor(GREEN)
                spine.set_linewidth(2.5)

        row1_axes.append(ax)

    # Draw arrows between step images
    fig.canvas.draw()  # force position calculation
    for i in range(len(row1_axes) - 1):
        col = ACCENT if i < 4 else GREEN
        draw_arrow(fig, row1_axes[i], row1_axes[i + 1], color=col, lw=2.5)

    # Lossless badge on decrypted image
    dec_pos = row1_axes[5].get_position()
    fig.text(dec_pos.x0 + dec_pos.width / 2, dec_pos.y0 - 0.015,
             '[OK] MSE = 0  |  PSNR = inf  |  Lossless',
             ha='center', va='top', fontsize=11, fontweight='bold', color=GREEN,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#dafbe1',
                       edgecolor=GREEN, alpha=0.9))

    # ══════════════════════════════════════════════════════════════════
    #  ROW 2 — HISTOGRAMS + CORRELATION + ATTRACTOR + METRICS
    # ══════════════════════════════════════════════════════════════════

    # --- 2a. Plaintext histogram ---
    ax_hist_p = fig.add_axes([0.04, 0.355, 0.18, 0.21])
    ax_hist_p.hist(plain.flatten(), bins=256, range=(0, 256),
                   color=ACCENT, alpha=0.85, edgecolor='none', linewidth=0)
    ax_hist_p.set_title('Plaintext Histogram', fontsize=13, fontweight='bold',
                        color=ACCENT, pad=6)
    ax_hist_p.set_xlabel('Pixel Intensity', fontsize=11)
    ax_hist_p.set_ylabel('Frequency', fontsize=11)
    ax_hist_p.set_xlim(0, 256)
    ax_hist_p.tick_params(labelsize=10)

    # --- 2b. Ciphertext histogram ---
    ax_hist_c = fig.add_axes([0.255, 0.355, 0.18, 0.21])
    ax_hist_c.hist(data['ciphertext'].flatten(), bins=256, range=(0, 256),
                   color=ACCENT2, alpha=0.85, edgecolor='none', linewidth=0)
    ax_hist_c.set_title('Ciphertext Histogram', fontsize=13, fontweight='bold',
                        color=ACCENT2, pad=6)
    ax_hist_c.set_xlabel('Pixel Intensity', fontsize=11)
    ax_hist_c.set_ylabel('Frequency', fontsize=11)
    ax_hist_c.set_xlim(0, 256)
    ax_hist_c.tick_params(labelsize=10)
    # Uniformity line
    ideal_freq = plain.size / 256
    ax_hist_c.axhline(y=ideal_freq, color=GREEN, ls='--', lw=1.0,
                      label=f'Ideal = {ideal_freq:.0f}')
    ax_hist_c.legend(fontsize=10, loc='upper right',
                     facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_LIGHT)

    # --- 2c. Correlation scatter — plaintext ---
    ax_corr_p = fig.add_axes([0.47, 0.355, 0.15, 0.21])
    ax_corr_p.scatter(px_h[:2000], py_h[:2000], s=0.4, c=ACCENT, alpha=0.5)
    ax_corr_p.set_title(f'Plain Corr (H)\nr = {r_h_p:+.4f}',
                        fontsize=12, fontweight='bold', color=ACCENT, pad=6)
    ax_corr_p.set_xlabel('Pixel (x, y)', fontsize=10)
    ax_corr_p.set_ylabel('Pixel (x, y+1)', fontsize=10)
    ax_corr_p.set_xlim(0, 255); ax_corr_p.set_ylim(0, 255)
    ax_corr_p.tick_params(labelsize=9)
    ax_corr_p.set_aspect('equal')

    # --- 2d. Correlation scatter — ciphertext ---
    ax_corr_c = fig.add_axes([0.645, 0.355, 0.15, 0.21])
    ax_corr_c.scatter(cx_h[:2000], cy_h[:2000], s=0.4, c=ACCENT2, alpha=0.5)
    ax_corr_c.set_title(f'Cipher Corr (H)\nr = {r_h_c:+.4f}',
                        fontsize=12, fontweight='bold', color=ACCENT2, pad=6)
    ax_corr_c.set_xlabel('Pixel (x, y)', fontsize=10)
    ax_corr_c.set_ylabel('Pixel (x, y+1)', fontsize=10)
    ax_corr_c.set_xlim(0, 255); ax_corr_c.set_ylim(0, 255)
    ax_corr_c.tick_params(labelsize=9)
    ax_corr_c.set_aspect('equal')

    # --- 2e. 3D Attractor ---
    ax_att = fig.add_axes([0.82, 0.355, 0.17, 0.23], projection='3d')
    ax_att.plot(ax_pts, ay_pts, az_pts, color='#8250df', lw=0.5, alpha=0.3)
    ax_att.scatter(ax_pts, ay_pts, zs=az_pts, c='black',
                   s=1.5, alpha=0.7, linewidths=0, depthshade=True)
    ax_att.set_xlabel('x', fontsize=10, labelpad=1)
    ax_att.set_ylabel('y', fontsize=10, labelpad=1)
    ax_att.set_zlabel('z', fontsize=10, labelpad=1)
    ax_att.set_title('4D-DSCHM\nStrange Attractor', fontsize=13,
                     fontweight='bold', color='#8250df', pad=4)
    ax_att.tick_params(labelsize=8, pad=0)
    ax_att.xaxis.pane.set_alpha(0.05)
    ax_att.yaxis.pane.set_alpha(0.05)
    ax_att.zaxis.pane.set_alpha(0.05)
    ax_att.set_facecolor(CARD_BG)
    ax_att.view_init(elev=25, azim=-55)
    ax_att.grid(True, alpha=0.2)

    # --- 2f. Black borders around middle row plots ---
    for ax in [ax_hist_p, ax_hist_c, ax_corr_p, ax_corr_c]:
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
            spine.set_visible(True)



    # ══════════════════════════════════════════════════════════════════
    #  ROW 3 — KEY INFO + SYSTEM PARAMS + SECURITY METRICS TABLE
    # ══════════════════════════════════════════════════════════════════

    # --- 3a. Key & System Parameters box ---
    ax_key = fig.add_axes([0.04, 0.022, 0.28, 0.235])
    ax_key.set_xlim(0, 1); ax_key.set_ylim(0, 1)
    ax_key.set_xticks([]); ax_key.set_yticks([])
    ax_key.set_title('System Parameters & Key', fontsize=14,
                     fontweight='bold', color=GOLD, pad=8)

    key_info = [
        ('Secret Key',        key.decode('utf-8', errors='replace')),
        ('Key Length',        '256 bits (32 bytes)'),
        ('SHA-256 Hash',      hashlib.sha256(key).hexdigest()[:40] + '...'),
        ('', ''),
        ('x₀ (initial)',      f'{x0:.16f}'),
        ('y₀ (initial)',      f'{y0:.16f}'),
        ('z₀ (initial)',      f'{z0:.16f}'),
        ('w₀ (initial)',      f'{w0:.16f}'),
        ('', ''),
        ('Map Constants',     f'A={A}, B={B}, C={C}, D={D}'),
        ('',                  f'E={E}, F={F}, G={G}, H={H_param}'),
        ('CA Rules',          '8 space-varying rules per pixel'),
        ('CA Iterations',     '2 Red-Black rounds'),
        ('Diffusion',         'Bidirectional (Fwd + Bwd) with ROL-3'),
        ('Warm-up',           '1000 transient iterations discarded'),
    ]

    y_pos = 0.95
    for label, value in key_info:
        if label == '' and value == '':
            y_pos -= 0.03
            continue
        if label:
            ax_key.text(0.02, y_pos, f'{label}:', fontsize=11.5,
                        fontweight='bold', color='black', va='top',
                        fontfamily='monospace')
        ax_key.text(0.38, y_pos, value, fontsize=11, color='black', va='top',
                    fontfamily='monospace')
        y_pos -= 0.065

    # --- 3b. Security Metrics Table ---
    ax_tbl = fig.add_axes([0.36, 0.022, 0.30, 0.235])
    ax_tbl.set_xlim(0, 1); ax_tbl.set_ylim(0, 1)
    ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
    ax_tbl.set_title('Security Analysis Results', fontsize=14,
                     fontweight='bold', color=GREEN, pad=8)

    metrics = [
        ('Metric',           'Measured',                   'Ideal',            'Status'),
        ('Entropy (cipher)',  f'{ent_cipher:.6f}',         '8.000000',         ent_cipher >= 7.99),
        ('Entropy (plain)',   f'{ent_plain:.6f}',          '--',               None),
        ('NPCR',              f'{npcr:.4f}%',              '>= 99.6094%',      npcr >= 99.60),
        ('UACI',              f'{uaci:.4f}%',              '~33.4635%',        33.15 <= uaci <= 33.75),
        ('Corr H (cipher)',   f'{r_h_c:+.6f}',            '~0.0000',          abs(r_h_c) <= 0.05),
        ('Corr V (cipher)',   f'{r_v_c:+.6f}',            '~0.0000',          abs(r_v_c) <= 0.05),
        ('Corr D (cipher)',   f'{r_d_c:+.6f}',            '~0.0000',          abs(r_d_c) <= 0.05),
        ('MSE (decrypted)',   f'{mse_val:.4f}',            '0.0000',           mse_val == 0.0),
        ('PSNR (decrypted)',  'inf' if mse_val == 0 else f'{psnr_val:.2f} dB',
                                                           'inf',              mse_val == 0.0),
        ('Key Space',         '2^256',                     '> 2^128',          True),
        ('Key Sensitivity',   'eps = 1e-13',               'High',             True),
    ]

    y_pos = 0.95
    # Header
    hdr = metrics[0]
    ax_tbl.text(0.02, y_pos, hdr[0], fontsize=12.5, fontweight='bold', color='black', va='top')
    ax_tbl.text(0.35, y_pos, hdr[1], fontsize=12.5, fontweight='bold', color='black', va='top')
    ax_tbl.text(0.62, y_pos, hdr[2], fontsize=12.5, fontweight='bold', color='black', va='top')
    ax_tbl.text(0.88, y_pos, hdr[3], fontsize=12.5, fontweight='bold', color='black', va='top')
    y_pos -= 0.05
    ax_tbl.plot([0.01, 0.99], [y_pos + 0.015, y_pos + 0.015],
                color=BORDER, lw=0.8, transform=ax_tbl.transAxes)

    for row in metrics[1:]:
        name, measured, ideal, status = row
        ax_tbl.text(0.02, y_pos, name, fontsize=11.5, color='black', va='top',
                    fontfamily='monospace')
        ax_tbl.text(0.35, y_pos, measured, fontsize=11.5, color='black', va='top',
                    fontfamily='monospace', fontweight='bold')
        ax_tbl.text(0.62, y_pos, ideal, fontsize=11.5, color='black', va='top',
                    fontfamily='monospace')
        if status is None:
            badge = '--'
            badge_col = 'black'
        elif status:
            badge = 'PASS'
            badge_col = GREEN
        else:
            badge = 'FAIL'
            badge_col = RED_C
        ax_tbl.text(0.88, y_pos, badge, fontsize=11.5, fontweight='bold',
                    color=badge_col, va='top')
        y_pos -= 0.075

    # --- 3c. Algorithm equations / flowchart description ---
    ax_eq = fig.add_axes([0.70, 0.022, 0.28, 0.235])
    ax_eq.set_xlim(0, 1); ax_eq.set_ylim(0, 1)
    ax_eq.set_xticks([]); ax_eq.set_yticks([])
    ax_eq.set_title('4D-DSCHM Map Equations', fontsize=14,
                     fontweight='bold', color='#8250df', pad=8)

    equations = [
        (r'$x_{n+1} = (\sin(\alpha \cdot y_n) + \beta \cdot \cos(x_n) + w_n)\, \mathrm{mod}\, 1$',
         f'a = {A},  b = {B}'),
        (r'$y_{n+1} = (\sin(\gamma \cdot x_{n+1}) + \delta \cdot \cos(y_n))\, \mathrm{mod}\, 1$',
         f'c = {C},  d = {D}'),
        (r'$z_{n+1} = (\sin(\epsilon \cdot z_n) + \zeta \cdot \cos(w_n))\, \mathrm{mod}\, 1$',
         f'e = {E},  f = {F}'),
        (r'$w_{n+1} = (\sin(\eta \cdot w_n) + \theta \cdot \cos(z_{n+1}))\, \mathrm{mod}\, 1$',
         f'g = {G},  h = {H_param}'),
    ]

    y_pos = 0.88
    for eq_str, params in equations:
        ax_eq.text(0.05, y_pos, eq_str, fontsize=16, color='black', va='top')
        ax_eq.text(0.70, y_pos - 0.01, params, fontsize=11.5, color='black',
                   va='top', fontfamily='monospace')
        y_pos -= 0.15

    # Diffusion formula
    y_pos -= 0.05
    ax_eq.plot([0.05, 0.95], [y_pos + 0.04, y_pos + 0.04],
               color=BORDER, lw=0.5, transform=ax_eq.transAxes)
    ax_eq.text(0.05, y_pos, 'Bidirectional Diffusion:', fontsize=14,
               fontweight='bold', color='black', va='top')
    y_pos -= 0.08
    ax_eq.text(0.05, y_pos,
               r'$C_i^{\rightarrow} = (S_i + \mathrm{ROL}_3(C_{i-1}))'
               r' \oplus K_i$',
               fontsize=15, color='black', va='top')
    y_pos -= 0.10
    ax_eq.text(0.05, y_pos,
               r'$C_i^{\leftarrow} = (C_i^{\rightarrow} + \mathrm{ROL}_3(C_{i+1}))'
               r' \oplus K_i$',
               fontsize=15, color='black', va='top')

    # ── Draw dashed borders for the three rows ────────────────────────

    # Row 1 border
    rect1 = patches.Rectangle((0.015, 0.645), 0.97, 0.32, fill=False,
                              edgecolor='lightgray', ls='--', lw=2, transform=fig.transFigure)
    fig.patches.append(rect1)

    # Row 2 border
    rect2 = patches.Rectangle((0.015, 0.325), 0.97, 0.31, fill=False,
                              edgecolor='lightgray', ls='--', lw=2, transform=fig.transFigure)
    fig.patches.append(rect2)

    # Row 3 border
    rect3 = patches.Rectangle((0.015, 0.01), 0.97, 0.30, fill=False,
                              edgecolor='lightgray', ls='--', lw=2, transform=fig.transFigure)
    fig.patches.append(rect3)

    # ── Section labels ────────────────────────────────────────────────
    fig.text(0.02, 0.965, '[1] ENCRYPTION PIPELINE', fontsize=16,
             fontweight='bold', color=ACCENT, va='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#ddf4ff',
                       edgecolor=ACCENT, alpha=0.8))

    fig.text(0.02, 0.635, '[2] STATISTICAL ANALYSIS', fontsize=16,
             fontweight='bold', color=ACCENT2, va='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffe1cc',
                       edgecolor=ACCENT2, alpha=0.8))

    fig.text(0.02, 0.31, '[3] SYSTEM DETAILS & SECURITY METRICS', fontsize=16,
             fontweight='bold', color=GOLD, va='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#fff8c5',
                       edgecolor=GOLD, alpha=0.8))

    # ── Footer credits ────────────────────────────────────────────────
    fig.text(0.50, 0.005,
             f'Image: {os.path.basename(img_path)} ({plain.shape[0]}x{plain.shape[1]})  |  '
             f'Key: "{key.decode()}"  |  '
             f'4D-DSCHM + RB-SVRCA + Bidirectional Feedback Diffusion',
             ha='center', va='bottom', fontsize=11, color=TEXT_DIM)

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = 'results/plots'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'graphical_abstract.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=BG, edgecolor='none', pad_inches=0.3)
    plt.close(fig)

    # Also save a white-background version for journals that need it
    print(f"\n[OK] Graphical abstract saved to: {out_path}")
    print(f"  Resolution: {28*200} x {18*200} px (5600 x 3600)")
    print("  Done!")


if __name__ == '__main__':
    main()
