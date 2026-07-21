import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def draw_flowchart():
    # Set up figure
    fig, ax = plt.subplots(figsize=(11.5, 11))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 12.5)
    ax.axis('off')

    # Helper function to draw rectangles
    def draw_box(text, x, y, w, h, bg_color='white', border_color='black', text_size=10, bold=False):
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=1.5, edgecolor=border_color, facecolor=bg_color, zorder=3)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=text_size, fontweight=weight, color='black', zorder=4)

    # Helper function for bent arrow
    def draw_bent_arrow(pts, color='black', ls='-', label=None, label_pos=None, align='center'):
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            if i == len(pts) - 2:
                # Add arrowhead at the end of last segment
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, ls=ls, mutation_scale=15),
                            zorder=2)
            else:
                # Draw straight line segment
                ax.plot([x1, x2], [y1, y2], color=color, lw=1.5, ls=ls, zorder=2)
        
        if label and label_pos:
            xl, yl = label_pos
            ax.text(xl, yl, label, ha=align, va='center', fontsize=8.5, fontweight='normal', fontstyle='italic',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95), zorder=5)

    # Coordinates
    col1_x = 2.6
    col2_x = 6.8

    # Draw boxes
    # Left column (Image processing path)
    draw_box("Plain Image\n(M x N)", col1_x, 11.2, 2.2, 0.7, bg_color='#fff2cc', border_color='#d6b656', bold=True)
    draw_box("Padding &\nBlock Setup", col1_x, 9.9, 2.2, 0.7)
    draw_box("Row & Column\nScrambling (Confusion)", col1_x, 8.4, 2.2, 0.7)
    draw_box("RB-SVRCA\nCA Iteration 1", col1_x, 7.0, 2.2, 0.7)
    draw_box("RB-SVRCA\nCA Iteration 2", col1_x, 5.6, 2.2, 0.7)
    draw_box("Bidirectional Feedback\nDiffusion (Forward)", col1_x, 4.2, 2.2, 0.7)
    draw_box("Bidirectional Feedback\nDiffusion (Backward)", col1_x, 2.8, 2.2, 0.7)
    draw_box("Cipher Image\n(M x N)", col1_x, 1.4, 2.2, 0.7, bg_color='#dae8fc', border_color='#6c8ebf', bold=True)

    # Right column (Keystream generator path)
    draw_box("256-bit Secret Key\n(K)", col2_x, 11.2, 2.2, 0.7, bg_color='#f5f5f5', border_color='#b3b3b3', bold=True)
    draw_box("SHA-256 Key Derivation\n(Digest)", col2_x, 9.9, 2.2, 0.7)
    draw_box("Initial States\n(x0, y0, z0, w0)", col2_x, 8.6, 2.2, 0.7)
    draw_box("4D-DSCHM\nHyperchaotic Map", col2_x, 7.2, 2.2, 0.9, border_color='#ff9900')

    # Draw solid connections (Main Flow)
    draw_bent_arrow([(col1_x, 10.85), (col1_x, 10.25)])
    draw_bent_arrow([(col1_x, 9.55), (col1_x, 8.75)])
    draw_bent_arrow([(col1_x, 8.05), (col1_x, 7.35)])
    draw_bent_arrow([(col1_x, 6.65), (col1_x, 5.95)])
    draw_bent_arrow([(col1_x, 5.25), (col1_x, 4.55)])
    draw_bent_arrow([(col1_x, 3.85), (col1_x, 3.15)])
    draw_bent_arrow([(col1_x, 2.45), (col1_x, 1.75)])

    draw_bent_arrow([(col2_x, 10.85), (col2_x, 10.25)])
    draw_bent_arrow([(col2_x, 9.55), (col2_x, 8.95)])
    draw_bent_arrow([(col2_x, 8.25), (col2_x, 7.65)])

    # Draw dashed connections (Keystreams)
    # 1. 4D-DSCHM -> Scrambling
    draw_bent_arrow([(col2_x - 1.1, 7.4), (4.8, 7.4), (4.8, 8.4), (col1_x + 1.1, 8.4)], 
                    ls='--', color='black', label="xs, ys", label_pos=(4.6, 8.55), align='center')
                    
    # 2. 4D-DSCHM -> CA Iteration 1 (zs)
    draw_bent_arrow([(col2_x - 1.1, 7.2), (4.5, 7.2), (4.5, 7.0), (col1_x + 1.1, 7.0)], 
                    ls='--', color='black', label="zs", label_pos=(4.3, 7.15), align='center')
                    
    # 3. 4D-DSCHM -> CA Iteration 2 (ws)
    draw_bent_arrow([(col2_x - 1.1, 7.0), (4.2, 7.0), (4.2, 5.6), (col1_x + 1.1, 5.6)], 
                    ls='--', color='black', label="ws", label_pos=(4.0, 5.75), align='center')

    # 4. SHA-256 Digest -> Feedback Diffusion (IV parameters)
    draw_bent_arrow([(col2_x + 1.1, 9.9), (8.2, 9.9), (8.2, 4.2), (col1_x + 1.1, 4.2)], 
                    ls='--', color='black', label="IV1, IV2", label_pos=(4.8, 4.35), align='center')

    # Load and draw actual project images
    plain_img_path = "Images/lena_color_512.tif"
    if not os.path.exists(plain_img_path):
        plain_img_path = "Images/lena_gray_512.tif"
        
    cipher_img_path = "results/encrypted/enc_lenna.png"
    if not os.path.exists(cipher_img_path):
        cipher_img_path = "ciphertext.png"

    # Draw Plain Image Thumbnail
    if os.path.exists(plain_img_path):
        try:
            img_plain = Image.open(plain_img_path)
            ax.imshow(np.array(img_plain), extent=[0.4, 1.2, 10.8, 11.6], zorder=4)
            rect_border = patches.Rectangle((0.4, 10.8), 0.8, 0.8, linewidth=1.5, edgecolor='black', facecolor='none', zorder=5)
            ax.add_patch(rect_border)
            draw_bent_arrow([(1.2, 11.2), (col1_x - 1.1, 11.2)])
        except Exception as e:
            print(f"Error loading plain image: {e}")

    # Draw Cipher Image Thumbnail
    if os.path.exists(cipher_img_path):
        try:
            img_cipher = Image.open(cipher_img_path)
            ax.imshow(np.array(img_cipher), extent=[0.4, 1.2, 1.0, 1.8], zorder=4)
            rect_border = patches.Rectangle((0.4, 1.0), 0.8, 0.8, linewidth=1.5, edgecolor='black', facecolor='none', zorder=5)
            ax.add_patch(rect_border)
            draw_bent_arrow([(col1_x - 1.1, 1.4), (1.2, 1.4)])
        except Exception as e:
            print(f"Error loading cipher image: {e}")

    # Draw Legends Box (bottom right corner)
    legend_x = 8.8
    legend_y = 2.2
    legend_w = 3.0
    legend_h = 2.8
    rect_legend = patches.Rectangle((legend_x - legend_w/2, legend_y - legend_h/2), legend_w, legend_h,
                                    linewidth=1.5, edgecolor='#4d4d4d', facecolor='#fbfbfb', zorder=3)
    ax.add_patch(rect_legend)
    
    # Legend Text
    ax.text(legend_x, legend_y + 1.1, "LEGEND & TERMINOLOGY", ha='center', va='center', fontsize=9, fontweight='bold', color='black', zorder=4)
    ax.plot([legend_x - legend_w/2 + 0.2, legend_x + legend_w/2 - 0.2], [legend_y + 0.9, legend_y + 0.9], color='gray', lw=1, zorder=4)
    
    legend_text = (
        r"$\bf{xs, ys}$ : Row & column confusion maps" + "\n"
        r"$\bf{zs}$         : Space-varying CA rules grid" + "\n"
        r"$\bf{ws}$        : Cellular Automata key grid" + "\n"
        r"$\bf{IV1, IV2}$ : Initial diffusion vectors" + "\n"
        r"$\bf{CA}$       : Cellular Automata updates" + "\n"
        r"$\bf{DSCHM}$  : Sine-Cosine Hyperchaotic Map" + "\n"
        r"$\bf{RB}$-$\bf{SVRCA}$: Red-Black Space-Varying" + "\n"
        r"                Reversible CA"
    )
    ax.text(legend_x - 1.35, legend_y - 0.45, legend_text, ha='left', va='center', fontsize=8.5, color='black', linespacing=1.4, zorder=4)

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/graphical_abstract_ieee.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated results/plots/graphical_abstract_ieee.png with legend and placement updates")

if __name__ == "__main__":
    draw_flowchart()
