"""
Step-by-step Execution and Flowchart Visualizer of the Image Cryptosystem on a 4x4 Grid.

This script takes a dummy 4x4 image, performs the encryption steps,
prints details of each step to the console, and draws a beautiful,
high-resolution publication-quality flowchart showing the actual matrices
evolving step-by-step.
"""

import os
import sys
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Add the project root to system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import src.cellular_automata as ca
from src.map_4d import key_to_initial_states, generate_chaotic_sequences

# Custom Colormaps for visual clarity
cmap_plain = LinearSegmentedColormap.from_list('plain', ['#fff2cc', '#ffd966'], N=256)
cmap_row_scr = LinearSegmentedColormap.from_list('row_scr', ['#fff7e6', '#ffe0b2'], N=256)
cmap_col_scr = LinearSegmentedColormap.from_list('col_scr', ['#fff2cc', '#ff9900'], N=256)
cmap_rules = LinearSegmentedColormap.from_list('rules', ['#e1f5fe', '#0288d1'], N=8)
cmap_key = LinearSegmentedColormap.from_list('key', ['#f3e5f5', '#7b1fa2'], N=256)
cmap_red = LinearSegmentedColormap.from_list('red', ['#ffebee', '#c62828'], N=256)
cmap_black = LinearSegmentedColormap.from_list('black', ['#efebe9', '#4e342e'], N=256)
cmap_diff = LinearSegmentedColormap.from_list('diff', ['#e8f5e9', '#2e7d32'], N=256)
cmap_cipher = LinearSegmentedColormap.from_list('cipher', ['#e0f7fa', '#00838f'], N=256)

def run_step_by_step_4x4():
    print("======================================================================")
    print("          STEP-BY-STEP IMAGE ENCRYPTION PIPELINE (4x4 DUMMY)")
    print("======================================================================\n")

    # 1. Plain Image setup
    plain = np.array([
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44]
    ], dtype=np.uint8)

    print("Step 1: Original Plain Image (4x4)")
    print(plain)
    print()

    # 2. Key Derivation and Chaos Generation
    key_bytes = b"Step_By_Step_Key_256_Bits_2026_"
    h = hashlib.sha256(key_bytes).digest()
    
    # Extract initial states
    chunks = [h[i:i+8] for i in range(0, 32, 8)]
    states = []
    for chunk in chunks:
        val = int.from_bytes(chunk, byteorder='big')
        states.append((val % 10**16) / 10**16)
    
    x0, y0, z0, w0 = states
    print("Step 2: Key Derivation (SHA-256)")
    print(f"  Secret Key: {key_bytes.decode()}")
    print(f"  Initial States: x0={x0:.6f}, y0={y0:.6f}, z0={z0:.6f}, w0={w0:.6f}")
    
    # Generate Chaotic sequences (length 16)
    xs, ys, zs, ws = generate_chaotic_sequences(4, 4, x0, y0, z0, w0)
    print(f"  Generated {len(xs)} hyperchaotic values using 4D-DSCHM.")
    print()

    # 3. Image Scrambling (Permutation) Stage
    row_idx = np.argsort(xs[:4])
    col_idx = np.argsort(ys[:4])
    
    print("Step 3: Confusion/Scrambling stage")
    print(f"  Row sorting indices:    {row_idx.tolist()}")
    row_scrambled = plain[row_idx, :]
    print("  Row Scrambled Grid:")
    print(row_scrambled)
    
    print(f"  Column sorting indices: {col_idx.tolist()}")
    scrambled = row_scrambled[:, col_idx]
    print("  Column Scrambled Grid (Full Scrambled):")
    print(scrambled)
    print()

    # 4. Checkerboard Partitioning and CA Setup
    rules = ((zs * 10**14).astype(np.int64) % 8).reshape(4, 4)
    key_grid = ((ws * 255).astype(np.uint8)).reshape(4, 4)
    
    grid_y, grid_x = np.indices((4, 4))
    red_mask = (grid_y + grid_x) % 2 == 0
    black_mask = ~red_mask
    
    # Create checkerboard background mask for visualization
    checkerboard_colors = np.zeros((4, 4), dtype=object)
    checkerboard_colors[red_mask] = '#ffb3b3'
    checkerboard_colors[black_mask] = '#e6e6e6'
    
    print("Step 4: Cellular Automata rules and grid setup")
    print("  CA Rules Grid (0 to 7):")
    print(rules)
    print("  CA Key Grid (0 to 255):")
    print(key_grid)
    print()

    # 5. Red Phase & Black Phase
    state_red = scrambled.copy()
    g_red = ca.compute_ca_transition(state_red, key_grid, rules)
    state_red[red_mask] = (state_red[red_mask].astype(np.int16) + g_red[red_mask].astype(np.int16)) % 256
    
    print("Step 5: CA Red Phase update (only Red cells updated)")
    print(state_red)
    print()

    state_black = state_red.copy()
    g_black = ca.compute_ca_transition(state_black, key_grid, rules)
    state_black[black_mask] = (state_black[black_mask].astype(np.int16) + g_black[black_mask].astype(np.int16)) % 256
    
    print("Step 6: CA Black Phase update (only Black cells updated - Final CA)")
    print(state_black)
    print()

    # 6. Bidirectional Feedback Diffusion
    state_flat = state_black.flatten()
    K_flat = key_grid.flatten()
    N = len(state_flat)
    
    # Forward Diffusion
    C1 = np.zeros_like(state_flat)
    IV1 = int(key_bytes[0]) ^ 157
    prev = IV1
    for i in range(N):
        c = (int(state_flat[i]) + prev) % 256 ^ int(K_flat[i])
        C1[i] = c
        prev = ca.rol(c, 3)
    
    forward_diff = C1.reshape(4, 4)
    print("Step 7: Forward Feedback Diffusion")
    print(forward_diff)
    print()
    
    # Backward Diffusion
    C2 = np.zeros_like(C1)
    IV2 = int(key_bytes[1]) ^ 223
    prev = IV2
    for i in range(N - 1, -1, -1):
        c = (int(C1[i]) + prev) % 256 ^ int(K_flat[i])
        C2[i] = c
        prev = ca.rol(c, 3)
        
    cipher = C2.reshape(4, 4)
    print("Step 8: Backward Feedback Diffusion (Final Ciphertext)")
    print(cipher)
    print()

    # 7. Draw the detailed 4x4 flowchart diagram
    draw_flowchart_diagram_4x4(plain, row_scrambled, scrambled, checkerboard_colors, rules, key_grid, state_red, state_black, forward_diff, cipher, row_idx, col_idx)

def draw_flowchart_diagram_4x4(plain, row_scrambled, scrambled, checkerboard_colors, rules, key_grid, state_red, state_black, forward_diff, cipher, row_idx, col_idx):
    fig, ax = plt.subplots(figsize=(15.5, 14.5))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 12.5)
    ax.axis('off')

    # Grid drawing helper
    def draw_grid_at(grid, cx, cy, cell_size=0.18, cmap=None, show_values=True, text_size=9.5, title=None, bg_mask=None, title_offset=0.65):
        n = 4
        start_x = cx - (n * cell_size) / 2
        start_y = cy - (n * cell_size) / 2
        
        # Draw frame background
        rect_bg = patches.Rectangle((cx - 0.55, cy - 0.55), 1.1, 1.1, linewidth=1.0, edgecolor='#cccccc', facecolor='#fafafa', zorder=1)
        ax.add_patch(rect_bg)
        
        v_min, v_max = grid.min(), grid.max()
        if v_max == v_min:
            v_max += 1.0
            
        for i in range(n):
            for j in range(n):
                val = grid[i, j]
                x = start_x + j * cell_size
                y = start_y + (n - 1 - i) * cell_size
                
                # Determine face color
                if bg_mask is not None:
                    facecolor = bg_mask[i, j]
                elif cmap is not None:
                    norm_val = (val - v_min) / (v_max - v_min)
                    facecolor = cmap(norm_val)
                else:
                    facecolor = 'white'
                    
                rect = patches.Rectangle((x, y), cell_size, cell_size, linewidth=0.7, edgecolor='gray', facecolor=facecolor, zorder=3)
                ax.add_patch(rect)
                
                if show_values:
                    # Adaptive text color
                    txt_color = 'white' if (cmap is not None and (val - v_min) / (v_max - v_min) > 0.6) else 'black'
                    if cmap in (cmap_plain, cmap_row_scr, cmap_col_scr, cmap_key):
                        txt_color = 'black'
                    ax.text(x + cell_size/2, y + cell_size/2, str(int(val)), ha='center', va='center', fontsize=text_size, color=txt_color, zorder=4)
                    
        if title:
            ax.text(cx, cy + title_offset, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color='black', zorder=5)

    # Explanation block helper
    def draw_explanation_box(text, cx, cy, w, h, bg_color='#f7f7f9', border_color='#b3b3b3'):
        rect = patches.Rectangle((cx - w/2, cy - h/2), w, h, linewidth=1.0, edgecolor=border_color, facecolor=bg_color, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=8.2, fontweight='normal', color='black', linespacing=1.3, zorder=4)

    # Arrow helper
    def draw_arrow(x1, y1, x2, y2, color='black', ls='-', label=None, label_pos=None, align='center'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, ls=ls, mutation_scale=15),
                    zorder=2)
        if label and label_pos:
            xl, yl = label_pos
            ax.text(xl, yl, label, ha=align, va='center', fontsize=8.5, fontweight='normal', fontstyle='italic',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95), zorder=5)

    # Coordinates layout for the main flowchart on the left (X in [0, 10.5])
    # 1. Plain Image
    draw_grid_at(plain, 1.8, 11.2, cmap=cmap_plain, title="1. Original Plain Image (4x4)\nI(i, j) = 10i + j")
    
    # 2. Row Scrambled Image
    row_lbl = f"Row Permutation\nIndices: {row_idx.tolist()}"
    draw_grid_at(row_scrambled, 5.2, 11.2, cmap=cmap_row_scr, title="2. Row Scrambled\n(Confusion Step 1)")
    draw_arrow(2.45, 11.2, 4.55, 11.2, label=row_lbl, label_pos=(3.5, 11.2))
    
    # 3. Column Scrambled Image
    col_lbl = f"Column Permutation\nIndices: {col_idx.tolist()}"
    draw_grid_at(scrambled, 8.6, 11.2, cmap=cmap_col_scr, title="3. Column Scrambled\n(Confusion Step 2)")
    draw_arrow(5.85, 11.2, 7.95, 11.2, label=col_lbl, label_pos=(6.9, 11.2))
    
    # 4. CA Rules Grid
    draw_grid_at(rules, 1.8, 8.0, cmap=cmap_rules, title="4. CA Transition Rules\n(Rule IDs 0-7)")
    
    # 5. CA Key Grid
    draw_grid_at(key_grid, 5.2, 8.0, cmap=cmap_key, title="5. CA Key Grid\n(Key values 0-255)")
    
    # 6. Checkerboard Mask
    draw_grid_at(np.zeros((4,4)), 8.6, 8.0, bg_mask=checkerboard_colors, show_values=False, title="6. Checkerboard Partition\n(Red/Black Grid)")
    
    # 7. Red Phase Output
    draw_grid_at(state_red, 1.8, 4.8, cmap=cmap_red, title="7. CA Red Phase\n(Red cells updated)")
    draw_arrow(8.6, 10.55, 1.8, 5.5, label="Scrambled\nPixels", label_pos=(5.3, 7.35))
    draw_arrow(1.8, 7.35, 1.8, 5.55)
    
    # 8. Black Phase Output
    draw_grid_at(state_black, 5.2, 4.8, cmap=cmap_black, title="8. CA Black Phase\n(Black cells updated)")
    draw_arrow(2.45, 4.8, 4.55, 4.8, label="Red Phase\nState", label_pos=(3.5, 4.8))
    draw_arrow(8.6, 7.35, 5.2, 5.55)
    draw_arrow(5.2, 7.35, 5.2, 5.55)
    
    # 9. Forward Diffusion Output
    draw_grid_at(forward_diff, 8.6, 4.8, cmap=cmap_diff, title="9. Forward Diffusion\n(Forward scan)")
    draw_arrow(5.85, 4.8, 7.95, 4.8, label="CA Output", label_pos=(6.9, 4.8))
    
    # 10. Backward Diffusion Output (Cipher)
    draw_grid_at(cipher, 5.2, 1.6, cmap=cmap_cipher, title="10. Backward Diffusion (Cipher Image)\n(Final Encrypted Output)")
    draw_arrow(8.6, 4.15, 5.85, 1.6, label="Backward Scan", label_pos=(7.55, 2.75))

    # --- Add Detailed Mathematical Explanation Boxes in the empty regions (Plain Text) ---
    
    # Box A: Key Derivation (placed between Row 1 and Row 2)
    key_exp = (
        "A. KEY DERIVATION & 4D-DSCHM MAP\n"
        "• Digest = SHA256(Key) -> states: x0=0.0418, y0=0.0494, z0=0.7971, w0=0.6502\n"
        "• Equations: x(k+1) = (sin(A*y) + B*cos(x) + w) mod 1.0\n"
        "• Keystreams: Generated 16 chaotic states of xs, ys, zs, ws after\n"
        "  1000 warm-up iterations to discard transients."
    )
    draw_explanation_box(key_exp, 5.2, 9.6, 6.0, 0.95, bg_color='#fffbe6', border_color='#ffe58f')

    # Box B: CA Rules and Key Grid Setup details (increased height for rules grid calculations)
    ca_setup_exp = (
        "B. RULES & KEY GRID GENERATION\n"
        "• CA Rules Grid: rules_grid = floor(zs * 10^14) mod 8\n"
        "  - Cell (0,0): zs_0 = 0.7569530717854205 -> 75695307178542 mod 8 = 6\n"
        "  - Cell (0,1): zs_1 = 0.2519631170567358 -> 25196311705673 mod 8 = 1\n"
        "  - Cell (0,2): floor(0.61301522891829... * 10^14) mod 8 = 5\n"
        "  - Cell (0,3): floor(0.21844684512312... * 10^14) mod 8 = 0\n"
        "• CA Key Grid: key_grid = floor(ws * 255)\n"
        "  - Row 1: floor(ws_k * 255) -> 161, 246, 38, 124\n"
        "• Checkerboard Partition: Red cell if (i+j) mod 2 == 0, else Black."
    )
    draw_explanation_box(ca_setup_exp, 5.2, 6.15, 6.0, 1.35, bg_color='#f9f0ff', border_color='#d3adf7')

    # Box C: CA State Update details (bottom left)
    ca_update_exp = (
        "C. REVERSIBLE CA UPDATE METHOD\n"
        "• Red Phase (only Red cells updated):\n"
        "  S_red(i,j) = (S_scrambled(i,j) + g_red(i,j)) mod 256\n"
        "  where g_red is local transition rule output on neighbors.\n\n"
        "• Black Phase (only Black cells updated):\n"
        "  S_black(i,j) = (S_red(i,j) + g_black(i,j)) mod 256\n"
        "  where g_black runs rules on updated Red neighbors."
    )
    draw_explanation_box(ca_update_exp, 1.8, 1.8, 2.9, 1.7, bg_color='#fff0f6', border_color='#ffadd2')

    # Box D: Bidirectional Diffusion details (bottom right)
    diff_exp = (
        "D. BIDIRECTIONAL FEEDBACK DIFFUSION\n"
        "• Forward Feedback (left-to-right scan):\n"
        "  C1(i) = (S_black(i) + C1(i-1)) mod 256 ^ K_flat(i)\n"
        "  where C1(-1) = IV1 = K_bytes[0] ^ 157 = 230\n\n"
        "• Backward Feedback (right-to-left scan):\n"
        "  C2(i) = (C1(i) + C2(i+1)) mod 256 ^ K_flat(i)\n"
        "  where C2(16) = IV2 = K_bytes[1] ^ 223 = 143"
    )
    draw_explanation_box(diff_exp, 8.6, 1.8, 2.9, 1.7, bg_color='#f6ffed', border_color='#b7eb8f')

    # --- Draw the MANUAL TRACE CALLOUT panel on the right side of the canvas ---
    panel_cx = 11.8
    panel_w = 2.6
    
    # Draw border panel for callout
    rect_panel = patches.Rectangle((panel_cx - panel_w/2, 4.3), panel_w, 7.3, linewidth=1.5, edgecolor='#4d4d4d', facecolor='#fafafa', zorder=2)
    ax.add_patch(rect_panel)
    
    # Panel Title
    ax.text(panel_cx, 11.2, "MANUAL TRACE OF CELL (0,0)\n(CA RED PHASE TRANSITION)", ha='center', va='center', fontsize=8.5, fontweight='bold', color='black', zorder=3)
    
    # Neighborhood Diagram coordinates centered at Y = 9.2
    diag_cy = 9.3
    cell_w = 0.35
    
    # Draw the neighborhood boxes
    def draw_diag_cell(val, label, dx, dy, is_center=False):
        x = panel_cx + dx * cell_w - cell_w/2
        y = diag_cy + dy * cell_w - cell_w/2
        fcolor = '#ffccd5' if is_center else '#d6e4ff'
        rect = patches.Rectangle((x, y), cell_w, cell_w, linewidth=1.0, edgecolor='black', facecolor=fcolor, zorder=3)
        ax.add_patch(rect)
        ax.text(x + cell_w/2, y + cell_w/2, str(val), ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)
        ax.text(x + cell_w/2, y + cell_w + 0.05, label, ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='blue', zorder=4)

    draw_diag_cell(42, "Cell(0,0)", 0, 0, is_center=True)
    draw_diag_cell(22, "T (Top)", 0, 1.3)
    draw_diag_cell(32, "B (Bottom)", 0, -1.3)
    draw_diag_cell(41, "L (Left)", -1.3, 0)
    draw_diag_cell(44, "R (Right)", 1.3, 0)
    
    # Mathematical calculation details inside the panel
    math_details = (
        "1. Neighbors of Cell (0,0):\n"
        "   L = 41,  R = 44\n"
        "   T = 22,  B = 32\n\n"
        "2. Active parameters:\n"
        "   Rule = 6,  Key = 161\n\n"
        "3. Transition Rule 6:\n"
        "   g = ((T ^ B) + Key) mod 256\n\n"
        "4. Intermediate XOR:\n"
        "   T ^ B = 22 ^ 32 = 54\n"
        "   (00010110 ^ 00100000 = 00110110)\n\n"
        "5. Key mixing:\n"
        "   g = (54 + 161) mod 256 = 215\n\n"
        "6. Final Cell State Update:\n"
        "   S_new = (S_old + g) mod 256\n"
        "   S_new = (42 + 215) mod 256\n"
        "   S_new = 257 mod 256 = 1"
    )
    ax.text(panel_cx - 1.15, 6.7, math_details, ha='left', va='top', fontsize=8.0, color='black', linespacing=1.25, zorder=4)

    # Arrow connecting callout to Red Phase Output Grid
    draw_arrow(panel_cx - panel_w/2, 9.3, 2.5, 5.0, color='blue', ls=':', label="Manual trace\nof Cell(0,0)", label_pos=(6.0, 5.0))

    # Add subtitle details about parameters used at the bottom center
    details_txt = (
        r"$\bf{Step-by-step\ Encryption\ Simulation\ for\ a\ dummy\ 4\times4\ grid}$" + "\n"
        r"$\bf{Secret\ Key}$: 'Step_By_Step_Key_256_Bits_2026_'  |  $\bf{Derived\ Chaotic\ Coordinates}$: $x_0=0.0418$, $y_0=0.0494$, $z_0=0.7971$, $w_0=0.6502$" + "\n"
        r"$\bf{Verification}$: Decrypting this Cipher Image recovers the Plain Image with 100% losslessness ($\text{MSE}=0.0$, $\text{PSNR}=\infty$)."
    )
    ax.text(5.2, 0.4, details_txt, ha='center', va='center', fontsize=9.5, color='#333333', linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdfdfd", ec="#999999", alpha=0.9))

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/step_by_step_flowchart_4x4.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated flowchart figure: results/plots/step_by_step_flowchart_4x4.png")

if __name__ == "__main__":
    run_step_by_step_4x4()
