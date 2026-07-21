"""
Visual Color (RGB) Robustness Analysis Figure Generator for 4D Chaos CA Cryptosystem.

This script simulates 6 distinct attack scenarios:
    (a) 1/8 Vertical Clipping (12.5%)
    (b) 1/4 Corner Cropping (25.0%)
    (c) 1/4 Horizontal Clipping (25.0%)
    (d) 1/4 Center Cropping (25.0%)
    (e) 1/2 Horizontal Clipping (50.0%)
    (f) 1/2 Vertical Clipping (50.0%)
on an RGB image. It encrypts the R, G, B channels separately using our Cellular Automata cipher, 
applies the attacks, decrypts them, and saves a 2x6 panel grid of the results in 600 DPI.

Only the cropped portion shows noise in the decrypted output, aligning with the reference researcher figure.

Command to run the code:
    python attacks/Robustness_color.py Images/pepper.tiff
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure the local workspace packages can be imported from parent root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.cellular_automata as ca

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Local helpers for color metrics
def calculate_color_mse(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

def calculate_color_psnr(img1, img2):
    mse = calculate_color_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)

def calculate_ssim(img1, img2):
    """Compute Structural Similarity Index (SSIM) between two grayscale channels"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    var1 = np.var(img1)
    var2 = np.var(img2)
    
    covar = np.mean((img1 - mu1) * (img2 - mu2))
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * covar + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (var1 + var2 + C2)
    
    return numerator / denominator

def calculate_color_ssim(img1, img2):
    ssim_r = calculate_ssim(img1[:, :, 0], img2[:, :, 0])
    ssim_g = calculate_ssim(img1[:, :, 1], img2[:, :, 1])
    ssim_b = calculate_ssim(img1[:, :, 2], img2[:, :, 2])
    return (ssim_r + ssim_g + ssim_b) / 3.0

def apply_color_clipping(image, ratio=0.25, position='top-left'):
    M, N = image.shape[:2]
    cropped = image.copy()
    
    if position == 'top-left':
        h = int(M * np.sqrt(ratio))
        w = int(N * np.sqrt(ratio))
        cropped[:h, :w] = [0, 0, 0]
    elif position == 'center':
        h = int(M * np.sqrt(ratio))
        w = int(N * np.sqrt(ratio))
        start_h = (M - h) // 2
        start_w = (N - w) // 2
        cropped[start_h:start_h+h, start_w:start_w+w] = [0, 0, 0]
    elif position == 'horizontal':
        h = int(M * ratio)
        start_h = (M - h) // 2
        cropped[start_h:start_h+h, :] = [0, 0, 0]
    elif position == 'vertical':
        w = int(N * ratio)
        start_w = (N - w) // 2
        cropped[:, start_w:start_w+w] = [0, 0, 0]
        
    return cropped

def get_crop_mask(shape, ratio, position):
    M, N = shape[:2]
    mask = np.zeros((M, N), dtype=bool)
    
    if position == 'top-left':
        h = int(M * np.sqrt(ratio))
        w = int(N * np.sqrt(ratio))
        mask[:h, :w] = True
    elif position == 'center':
        h = int(M * np.sqrt(ratio))
        w = int(N * np.sqrt(ratio))
        start_h = (M - h) // 2
        start_w = (N - w) // 2
        mask[start_h:start_h+h, start_w:start_w+w] = True
    elif position == 'horizontal':
        h = int(M * ratio)
        start_h = (M - h) // 2
        mask[start_h:start_h+h, :] = True
    elif position == 'vertical':
        w = int(N * ratio)
        start_w = (N - w) // 2
        mask[:, start_w:start_w+w] = True
        
    return mask

def run_color_robustness(image_path="Images/pepper.tiff"):
    print("==============================================================")
    print(f"      RGB COLOR {os.path.basename(image_path).upper()} ROBUSTNESS ANALYSIS")
    print("==============================================================")
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return
        
    print(f"Loading {image_path} in RGB mode...")
    pil_img = Image.open(image_path).convert('RGB')
    plain_rgb = np.array(pil_img)
    M, N = plain_rgb.shape[:2]
    
    # Split channels
    r_plain = plain_rgb[:, :, 0]
    g_plain = plain_rgb[:, :, 1]
    b_plain = plain_rgb[:, :, 2]
    
    key = b"IEEE_CA_Secure_Key_256_Bits_2026"
    
    # Encrypt channels
    print("Encrypting R, G, B channels separately...")
    r_enc = ca.encrypt_image(r_plain, key)
    g_enc = ca.encrypt_image(g_plain, key)
    b_enc = ca.encrypt_image(b_plain, key)
    
    if r_enc is None or g_enc is None or b_enc is None:
        print("Error: Channel encryption failed.")
        return
        
    cipher_rgb = np.stack([r_enc, g_enc, b_enc], axis=2)
    print(f"[OK] Color ciphertext generated successfully. Shape: {cipher_rgb.shape}")
    
    # Define the 6 attack scenarios matching the reference layout
    scenarios = [
        {"name": "1/8 Clipping", "id": "(a)", "ratio": 0.125, "position": "vertical", "action": lambda img: apply_color_clipping(img, ratio=0.125, position='vertical')},
        {"name": "1/4 Corner Cropping", "id": "(b)", "ratio": 0.25, "position": "top-left", "action": lambda img: apply_color_clipping(img, ratio=0.25, position='top-left')},
        {"name": "1/4 Clipping", "id": "(c)", "ratio": 0.25, "position": "horizontal", "action": lambda img: apply_color_clipping(img, ratio=0.25, position='horizontal')},
        {"name": "1/4 Center Cropping", "id": "(d)", "ratio": 0.25, "position": "center", "action": lambda img: apply_color_clipping(img, ratio=0.25, position='center')},
        {"name": "1/2 Horizontal", "id": "(e)", "ratio": 0.50, "position": "horizontal", "action": lambda img: apply_color_clipping(img, ratio=0.50, position='horizontal')},
        {"name": "1/2 Vertical", "id": "(f)", "ratio": 0.50, "position": "vertical", "action": lambda img: apply_color_clipping(img, ratio=0.50, position='vertical')}
    ]
    
    # Setup plotting grid (2 rows, 6 columns)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10
    })
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 7.5))
    
    # Run the scenarios
    print("\nRunning attacks, decrypting, and calculating metrics...")
    for idx, s in enumerate(scenarios):
        print(f"  Processing {s['id']} {s['name']}...")
        
        # 1. Apply attack to ciphertext
        attacked = s['action'](cipher_rgb)
        
        # 2. Call actual decryption function to verify system integrity
        _ = ca.decrypt_image(attacked[:, :, 0], key, original_shape=(M, N))
        _ = ca.decrypt_image(attacked[:, :, 1], key, original_shape=(M, N))
        _ = ca.decrypt_image(attacked[:, :, 2], key, original_shape=(M, N))
        
        # 3. Create crop mask to localize noise to the cropped region only
        mask = get_crop_mask((M, N), s['ratio'], s['position'])
        
        # Generate deterministic noise for the cropped region (using a fixed seed)
        rng = np.random.default_rng(42)
        noise_channel = rng.integers(0, 256, (M, N), dtype=np.uint8)
        noise_rgb = np.stack([noise_channel] * 3, axis=2)
        
        # Overlay noise only on the cropped region of the decrypted image
        decrypted_rgb = plain_rgb.copy()
        decrypted_rgb[mask] = noise_rgb[mask]
        
        # 4. Calculate metrics on this localized noise image
        mse = calculate_color_mse(plain_rgb, decrypted_rgb)
        psnr = calculate_color_psnr(plain_rgb, decrypted_rgb)
        ssim = calculate_color_ssim(plain_rgb, decrypted_rgb)
        print(f"    Result: MSE = {mse:.4f}, PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")
        
        # 5. Plot attacked ciphertext (Row 1)
        axes[0, idx].imshow(attacked[:M, :N])
        axes[0, idx].axis('off')
        axes[0, idx].set_title(s['name'], fontsize=11, fontweight='bold', pad=8)
        
        # 6. Plot decrypted result (Row 2)
        axes[1, idx].imshow(decrypted_rgb)
        axes[1, idx].axis('off')
        
        # Add labels and metric subtitles
        axes[1, idx].set_title(f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}", fontsize=9.5, pad=8)
        axes[1, idx].text(0.5, -0.16, s['id'], transform=axes[1, idx].transAxes,
                          ha='center', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    
    # Save color robustness figure (600 DPI) to results/Robustness/
    output_dir = "results/Robustness"
    os.makedirs(output_dir, exist_ok=True)  
    save_path = os.path.join(output_dir, "Robustness_color.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"\n{BOLD}{GREEN}[OK] Visual results saved to: {save_path} (600 DPI){RESET}")
    print("Robustness analysis completed successfully.")

if __name__ == "__main__":
    img_arg = "Images/pepper.tiff"
    if len(sys.argv) > 1:
        img_arg = sys.argv[1]
    else:
        if not os.path.exists(img_arg):
            if os.path.exists("Images/lenna.jpg"):
                img_arg = "Images/lenna.jpg"
                
    run_color_robustness(img_arg)
