"""
PSNR, MSE, and SSIM Quantitative Robustness Evaluation for 4D Chaos CA Cryptosystem.

This script evaluates the resistance of our 4D Trigonometric Hyperchaotic Map & Cellular Automata Cryptosystem 
against transmission attacks (cropping and noise). It encrypts color images channel-by-channel (R, G, B), 
applies vertical/horizontal clipping, corner/center cropping, and salt-and-pepper noise, decrypts the attacked ciphers, 
and calculates standard evaluation metrics (MSE, PSNR, and SSIM) relative to the original plaintext.

Command to run the code for a single image:
    python attacks/psnr_mse_ssim.py Images/lenna.jpg
"""

import os
import sys
import numpy as np
from PIL import Image

# Reconfigure stdout to support UTF-8 (for drawing clean tables)
if sys.platform != 'win32' or hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add the project root to system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.cellular_automata as ca

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91"
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

# Local helpers for color attacks
def add_color_sp_noise(image, density=0.02):
    noisy = image.copy()
    M, N = image.shape[:2]
    num_pixels = M * N
    
    # Use fixed seed for deterministic reproducibility in research
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

def run_metrics_evaluation(image_path):
    print(f"\n{BOLD}{CYAN}Evaluating Quantitative Robustness for: {image_path}{RESET}")
    if not os.path.exists(image_path):
        print(f"{RED}Error: File {image_path} does not exist.{RESET}")
        return None
        
    try:
        pil_img = Image.open(image_path).convert('RGB')
        plain_rgb = np.array(pil_img)
        M, N = plain_rgb.shape[:2]
        
        # Split channels
        r_plain = plain_rgb[:, :, 0]
        g_plain = plain_rgb[:, :, 1]
        b_plain = plain_rgb[:, :, 2]
        
        key = b"IEEE_CA_Secure_Key_256_Bits_2026"
        
        # Encrypt channels
        print("Encrypting R, G, B channels...")
        r_enc = ca.encrypt_image(r_plain, key)
        g_enc = ca.encrypt_image(g_plain, key)
        b_enc = ca.encrypt_image(b_plain, key)
        
        cipher_rgb = np.stack([r_enc, g_enc, b_enc], axis=2)
        
        # Define attack scenarios
        scenarios = [
            {"name": "1/8 Vertical Clipping (12.5%)", "action": lambda img: apply_color_clipping(img, ratio=0.125, position='vertical')},
            {"name": "1/4 Corner Cropping (25.0%)", "action": lambda img: apply_color_clipping(img, ratio=0.25, position='top-left')},
            {"name": "1/4 Horizontal Clipping (25.0%)", "action": lambda img: apply_color_clipping(img, ratio=0.25, position='horizontal')},
            {"name": "1/4 Center Cropping (25.0%) [Scenario D]", "action": lambda img: apply_color_clipping(img, ratio=0.25, position='center')},
            {"name": "1/2 Horizontal Clipping (50.0%)", "action": lambda img: apply_color_clipping(img, ratio=0.50, position='horizontal')},
            {"name": "1/2 Vertical Clipping (50.0%)", "action": lambda img: apply_color_clipping(img, ratio=0.50, position='vertical')},
            {"name": "Salt & Pepper Noise (2%)", "action": lambda img: add_color_sp_noise(img, density=0.02)},
            {"name": "Salt & Pepper Noise (10%)", "action": lambda img: add_color_sp_noise(img, density=0.10)},
            {"name": "Salt & Pepper Noise (20%)", "action": lambda img: add_color_sp_noise(img, density=0.20)}
        ]
        
        results = []
        print("\nSimulating attacks and decrypting to calculate MSE, PSNR, and SSIM...")
        
        for s in scenarios:
            # 1. Apply attack
            attacked = s['action'](cipher_rgb)
            
            # 2. Decrypt
            r_dec = ca.decrypt_image(attacked[:, :, 0], key, original_shape=(M, N))
            g_dec = ca.decrypt_image(attacked[:, :, 1], key, original_shape=(M, N))
            b_dec = ca.decrypt_image(attacked[:, :, 2], key, original_shape=(M, N))
            
            decrypted_rgb = np.stack([r_dec, g_dec, b_dec], axis=2)
            
            # 3. Calculate metrics
            mse = calculate_color_mse(plain_rgb, decrypted_rgb)
            psnr = calculate_color_psnr(plain_rgb, decrypted_rgb)
            ssim = calculate_color_ssim(plain_rgb, decrypted_rgb)
            
            results.append({
                'name': s['name'],
                'mse': mse,
                'psnr': psnr,
                'ssim': ssim
            })
            print(f"  [OK] {s['name']:<40} | MSE: {mse:<9.4f} | PSNR: {psnr:<6.2f} dB | SSIM: {ssim:<6.4f}")
            
        # Print table
        print(f"\n{BOLD}{'Attack Scenario':<42} {'MSE':<12} {'PSNR (dB)':<12} {'SSIM':<12}{RESET}")
        print("-" * 82)
        for r in results:
            print(f"{r['name']:<42} {r['mse']:<12.4f} {r['psnr']:<12.2f} {r['ssim']:<12.4f}")
        print("-" * 82)
    except Exception as e:
        print(f"{RED}Error evaluating metrics: {e}{RESET}")
        import traceback
        traceback.print_exc()

def main():
    print(f"{BOLD}========================================================================{RESET}")
    print(f"{BOLD}             PSNR, MSE, and SSIM Robustness Evaluation                 {RESET}")
    print(f"{BOLD}========================================================================{RESET}")
    
    # Get image from arguments or default
    image_path = "Images/pepper.tiff"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        if not os.path.exists(image_path):
            if os.path.exists("Images/lenna.jpg"):
                image_path = "Images/lenna.jpg"
            else:
                image_path = "test_image.png"
                
    run_metrics_evaluation(image_path)

if __name__ == "__main__":
    main()
