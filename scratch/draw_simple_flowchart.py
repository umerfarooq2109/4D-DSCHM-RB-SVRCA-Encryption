import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_simple_flowchart():
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 8)
    ax.axis('off')

    # Helper function to draw rectangles
    def draw_box(text, x, y, w, h, bg_color='white', border_color='black', text_size=10, bold=False):
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=1.5, edgecolor=border_color, facecolor=bg_color, zorder=3)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=text_size, fontweight=weight, color='black', zorder=4)

    # Helper function for straight arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, mutation_scale=15),
                    zorder=2)

    # Helper function for bent arrows
    def draw_bent_arrow(pts):
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            if i == len(pts) - 2:
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, mutation_scale=15),
                            zorder=2)
            else:
                ax.plot([x1, x2], [y1, y2], color='black', lw=1.5, zorder=2)

    # Column 1 (x=2)
    draw_box("Original Image", 2.0, 7.0, 2.0, 0.7, bg_color='#fff2cc', border_color='#d6b656', bold=True)
    draw_box("SHA-256\nKey Derivation", 2.0, 5.5, 2.0, 0.7)
    draw_box("4D-DSCHM\nChaos Generator", 2.0, 4.0, 2.2, 0.7)

    # Column 2 (x=6)
    draw_box("Row & Column\nScrambling", 6.0, 4.0, 2.0, 0.7)

    # Column 1.5 (x=4)
    draw_box("RB-SVRCA\nIteration 1", 4.0, 2.5, 2.0, 0.7)

    # Column 2.5 (x=8)
    draw_box("RB-SVRCA\nIteration 2", 8.0, 2.5, 2.0, 0.7)
    draw_box("Bidirectional\nFeedback Diffusion", 8.0, 1.0, 2.0, 0.7)
    draw_box("Cipher Image", 8.0, -0.5, 2.0, 0.7, bg_color='#f5f5f5', border_color='#7f7f7f', bold=True)

    # Connect the blocks
    # Original Image -> SHA-256
    draw_arrow(2.0, 6.65, 2.0, 5.85)
    # SHA-256 -> 4D-DSCHM
    draw_arrow(2.0, 5.15, 2.0, 4.35)
    # 4D-DSCHM -> Row & Column Scrambling
    draw_arrow(3.1, 4.0, 5.0, 4.0)
    # Row & Column Scrambling -> RB-SVRCA Iteration 1 (bent)
    draw_bent_arrow([(6.0, 3.65), (6.0, 3.25), (4.0, 3.25), (4.0, 2.85)])
    # RB-SVRCA Iteration 1 -> RB-SVRCA Iteration 2
    draw_arrow(5.0, 2.5, 7.0, 2.5)
    # RB-SVRCA Iteration 2 -> Bidirectional Feedback Diffusion
    draw_arrow(8.0, 2.15, 8.0, 1.35)
    # Bidirectional Feedback Diffusion -> Cipher Image
    draw_arrow(8.0, 0.65, 8.0, -0.15)

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/simple_flowchart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated results/plots/simple_flowchart.png")

if __name__ == "__main__":
    # Draw simple flowchart
    draw_simple_flowchart()
