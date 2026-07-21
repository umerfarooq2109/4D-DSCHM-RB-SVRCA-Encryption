import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

# Reconfigure stdout to support UTF-8
sys.stdout.reconfigure(encoding='utf-8')
# Add the project root to system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.cellular_automata as ca

# Scientific styling parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
})

def generate_color_histogram(image_path):
    print(f"Loading image from: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return
        
    try:
        # Load plaintext image
        pil_img = Image.open(image_path)
        if pil_img.mode in ('RGBA', 'LA') or (isinstance(pil_img.info.get('transparency'), (bytes, int))):
            pil_img = pil_img.convert('RGB')
        img_arr = np.array(pil_img, dtype=np.uint8)
        
        # Check if the image is indeed color (RGB)
        if len(img_arr.shape) != 3 or img_arr.shape[2] != 3:
            print("Error: The image is not a 3-channel color image.")
            return

        # Setup output folder and paths
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = "results/Histogram_Color"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_histogram.png")

        # Encrypt the color image
        print("Encrypting image...")
        key = b"IEEE_CA_Secure_Key_256_Bits_2026"
        r_enc = ca.encrypt_image(img_arr[:, :, 0], key)
        g_enc = ca.encrypt_image(img_arr[:, :, 1], key)
        b_enc = ca.encrypt_image(img_arr[:, :, 2], key)
        c_img = np.stack([r_enc, g_enc, b_enc], axis=2)
        
        # Set up matplotlib figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 9.5), dpi=300)
        plt.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.93, wspace=0.28, hspace=0.28)
        
        # Premium academic color palette
        colors = ['#d62728', '#2ca02c', '#1f77b4']  # Academic red, green, blue
        channel_names = ['Red', 'Green', 'Blue']
        
        # ==========================================
        # ROW 1: PLAIN IMAGE & HISTOGRAMS
        # ==========================================
        
        # Plain Image
        axes[0, 0].imshow(img_arr)
        axes[0, 0].set_title("Plain Image", fontweight='bold', pad=10)
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        axes[0, 0].set_box_aspect(1)
        
        # Plain R, G, B histograms
        for i in range(3):
            ax = axes[0, i+1]
            channel_data = img_arr[:, :, i].flatten()
            hist, bins = np.histogram(channel_data, bins=256, range=[0, 256])
            
            # Plot clean bar chart representing pixel distribution
            ax.bar(np.arange(256), hist, width=1.0, color=colors[i], edgecolor=colors[i], alpha=0.85)
            ax.set_title(f"Plain {channel_names[i]} Channel", fontweight='bold', pad=10)
            
            # Configure spines (remove top and right for clean modern academic look)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle=':', alpha=0.5, color='gray')
            ax.set_xlim([0, 255])
            ax.set_ylim([0, int(np.max(hist) * 1.05)])
            ax.set_ylabel("Pixel Frequency")
            ax.set_box_aspect(1)
            
        # ==========================================
        # ROW 2: CIPHER IMAGE & HISTOGRAMS
        # ==========================================
        
        # Cipher Image
        axes[1, 0].imshow(c_img)
        axes[1, 0].set_title("Cipher Image", fontweight='bold', pad=10)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        axes[1, 0].set_box_aspect(1)
        
        # Cipher R, G, B histograms
        for i in range(3):
            ax = axes[1, i+1]
            channel_data = c_img[:, :, i].flatten()
            hist, bins = np.histogram(channel_data, bins=256, range=[0, 256])
            
            # Plot clean bar chart representing pixel distribution
            ax.bar(np.arange(256), hist, width=1.0, color=colors[i], edgecolor=colors[i], alpha=0.85)
            ax.set_title(f"Cipher {channel_names[i]} Channel", fontweight='bold', pad=10)
            
            # Configure spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle=':', alpha=0.5, color='gray')
            ax.set_xlim([0, 255])
            
            # Standardize y-axis limit for ciphertext histogram to visually emphasize uniformity
            avg_height = channel_data.size / 256.0
            y_limit = int(avg_height * 1.35)  # around 1380 for 512x512
            ax.set_ylim([0, y_limit])
            ax.set_ylabel("Pixel Frequency")
            ax.set_box_aspect(1)

        # ==========================================
        # BOTTOM LABELS (COLUMN CAPTIONS)
        # ==========================================
        axes[1, 0].set_xlabel("(a) Plain & Cipher Images", fontweight='bold', labelpad=15)
        axes[1, 1].set_xlabel("(b) Red Channel Histograms", fontweight='bold', labelpad=15)
        axes[1, 2].set_xlabel("(c) Green Channel Histograms", fontweight='bold', labelpad=15)
        axes[1, 3].set_xlabel("(d) Blue Channel Histograms", fontweight='bold', labelpad=15)
        
        # Set x-labels for all histogram columns
        for row in range(2):
            for col in range(1, 4):
                axes[row, col].set_xlabel("Pixel Intensity")
        
        # Save figure in high resolution (600 DPI)
        plt.savefig(output_path, bbox_inches='tight', dpi=600)
        plt.close()
        
        print(f"Histogram analysis figure saved successfully to: {output_path} (600 DPI)")
        
    except Exception as e:
        print(f"Error performing histogram analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    image_path = "Images/pepper.tiff"  # Default to pepper.tiff since it exists in the workspace
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
    generate_color_histogram(image_path)

if __name__ == "__main__":
    main()
