"""
Visual Color (RGB) Noise Attack Figure Generator for 4D Chaos CA Cryptosystem.

Generates publication-quality figures showing the encrypted images corrupted
by varying levels of noise (1%, 2%, 3%, 5%, 8%, 10% intensity) and their corresponding
decrypted color results side-by-side, matching the style of standard journal publications.

Saved in results/Noise folder with the name nameoftheimage_noise.png at 600 DPI.
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Ensure the local workspace packages can be imported from parent root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.cellular_automata as ca

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

def add_color_sp_noise(image, density=0.02):
    noisy = image.copy()
    M, N = image.shape[:2]
    num_pixels = M * N
    
    # Use deterministic RNG for reproducible figures
    rng = np.random.default_rng(42)
    
    # Salt (white pixels [255, 255, 255])
    num_salt = int(np.ceil(density * num_pixels / 2))
    salt_r = rng.integers(0, M, num_salt)
    salt_c = rng.integers(0, N, num_salt)
    noisy[salt_r, salt_c] = [255, 255, 255]
    
    # Pepper (black pixels [0, 0, 0])
    num_pepper = int(np.ceil(density * num_pixels / 2))
    pepper_r = rng.integers(0, M, num_pepper)
    pepper_c = rng.integers(0, N, num_pepper)
    noisy[pepper_r, pepper_c] = [0, 0, 0]
    
    return noisy

def generate_noise_figure(image_path="Images/pepper.tiff"):
    """
    Apply color salt & pepper noise with varying densities, decrypt, and generate a 2x6 grid:
      - Row 1: Noise-corrupted color encrypted images
      - Row 2: Corresponding decrypted color images
    """
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    print(f"Loading {image_path} in RGB mode...")
    pil_img = Image.open(image_path).convert('RGB')
    plain_rgb = np.array(pil_img)
    M, N = plain_rgb.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Split channels
    r_plain = plain_rgb[:, :, 0]
    g_plain = plain_rgb[:, :, 1]
    b_plain = plain_rgb[:, :, 2]
    
    key = b"IEEE_CA_Secure_Key_256_Bits_2026"
    
    # Encrypt the color image
    print("Encrypting R, G, B channels separately...")
    r_enc = ca.encrypt_image(r_plain, key)
    g_enc = ca.encrypt_image(g_plain, key)
    b_enc = ca.encrypt_image(b_plain, key)
    
    if r_enc is None or g_enc is None or b_enc is None:
        print("Error: Image encryption failed.")
        return
        
    cipher_rgb = np.stack([r_enc, g_enc, b_enc], axis=2)

    # Define the 6 noise intensities: 1%, 2%, 3%, 5%, 8%, 10%
    scenarios = [
        {"name": "1% Noise Intensity", "density": 0.01, "label": "(a) 1% Intensity"},
        {"name": "2% Noise Intensity", "density": 0.02, "label": "(b) 2% Intensity"},
        {"name": "3% Noise Intensity", "density": 0.03, "label": "(c) 3% Intensity"},
        {"name": "5% Noise Intensity", "density": 0.05, "label": "(d) 5% Intensity"},
        {"name": "8% Noise Intensity", "density": 0.08, "label": "(e) 8% Intensity"},
        {"name": "10% Noise Intensity", "density": 0.10, "label": "(f) 10% Intensity"}
    ]

    # Run attacks and decryption
    attacked_images = []
    decrypted_images = []
    metrics = []

    print("Running noise corruption and decryption...")
    for s in scenarios:
        # Apply S&P noise to the color ciphertext
        attacked = add_color_sp_noise(cipher_rgb, density=s["density"])
        attacked_images.append(attacked)

        # Decrypt R, G, B channels separately
        r_dec = ca.decrypt_image(attacked[:, :, 0], key, original_shape=(M, N))
        g_dec = ca.decrypt_image(attacked[:, :, 1], key, original_shape=(M, N))
        b_dec = ca.decrypt_image(attacked[:, :, 2], key, original_shape=(M, N))

        if r_dec is not None and g_dec is not None and b_dec is not None:
            decrypted_rgb = np.stack([r_dec, g_dec, b_dec], axis=2)
            mse = calculate_color_mse(plain_rgb, decrypted_rgb)
            psnr = calculate_color_psnr(plain_rgb, decrypted_rgb)
            ssim = calculate_color_ssim(plain_rgb, decrypted_rgb)
            metrics.append((mse, psnr, ssim))
            decrypted_images.append(decrypted_rgb)
            print(f"  {s['name']} -> MSE: {mse:.4f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        else:
            metrics.append((0.0, 0.0, 0.0))
            decrypted_images.append(np.zeros_like(plain_rgb))

    # Setup the plot with tight vertical space (6 columns)
    fig, axes = plt.subplots(2, 6, figsize=(18, 7.5))
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10
    })

    for i in range(6):
        # 1. Plot attacked encrypted image
        axes[0, i].imshow(attacked_images[i][:M, :N])
        axes[0, i].axis('off')
        axes[0, i].set_title(scenarios[i]["name"], fontsize=11, fontweight='bold', pad=8)

        # 2. Plot decrypted result
        axes[1, i].imshow(decrypted_images[i])
        axes[1, i].axis('off')
        
        # Add MSE, PSNR, and SSIM info underneath decrypted image in the white space
        mse, psnr, ssim = metrics[i]
        metrics_text = f"MSE: {mse:.2f}\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}"
        axes[1, i].text(0.5, -0.16, metrics_text, transform=axes[1, i].transAxes,
                        ha='center', va='top', fontsize=9.5)
        
        # Label each column at the bottom
        axes[1, i].text(0.5, -0.40, scenarios[i]["label"], transform=axes[1, i].transAxes,
                        ha='center', va='top', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.16, hspace=0.15)
    
    # Save the figure to results/Noise/ with high resolution (600 DPI)
    output_dir = "results/Noise"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{base_name}_noise.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"\nSuccessfully generated and saved color noise robustness figure to: {save_path} (600 DPI)")

if __name__ == "__main__":
    img_arg = "Images/pepper.tiff" 
    if len(sys.argv) > 1:
        img_arg = sys.argv[1]
    else:
        if not os.path.exists(img_arg):
            if os.path.exists("Images/lenna.jpg"):
                img_arg = "Images/lenna.jpg"
    generate_noise_figure(img_arg)
