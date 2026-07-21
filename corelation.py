"""
Adjacent Pixel Correlation Coefficient Calculator in H, V, D Directions.
Processes a batch of images, encrypts them, computes the plain/cipher correlations,
outputs a color-coded console table, and saves the data in results/tables/Coreraltion_table.csv.
"""

import os
import sys
import csv
import numpy as np
from PIL import Image

# Ensure the local workspace packages can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.cellular_automata as ca

# Target lists of test images
TEST_IMAGES = [
    '4.1.01.png',
    '4.1.01.tiff',
    '4.1.02.png',
    '4.1.02.tiff',
    '4.1.03.png',
    '4.1.03.tiff',
    '4.1.04.tiff',
]

def get_deterministic_coords(max_val, size, seed=42):
    indices = np.arange(size, dtype=np.float64)
    pseudo_rand = np.sin(indices + seed) * 12345.6789
    coords = (np.abs(pseudo_rand) % 1.0 * max_val).astype(np.int32)
    return coords

def calculate_correlation(img, direction='horizontal', num_pairs=10000):
    H, W = img.shape
    x_coords = get_deterministic_coords(H - 1 if direction != 'horizontal' else H, num_pairs, seed=101)
    y_coords = get_deterministic_coords(W - 1 if direction != 'vertical' else W, num_pairs, seed=202)
    
    if direction == 'horizontal':
        x = img[x_coords, y_coords].astype(np.float64)
        y = img[x_coords, y_coords + 1].astype(np.float64)
    elif direction == 'vertical':
        x = img[x_coords, y_coords].astype(np.float64)
        y = img[x_coords + 1, y_coords].astype(np.float64)
    else:  # diagonal
        x = img[x_coords, y_coords].astype(np.float64)
        y = img[x_coords + 1, y_coords + 1].astype(np.float64)
        
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return 0.0 if den == 0 else num / den

def compute_image_correlations(img_path, key):
    pil_img = Image.open(img_path)
    # Check if grayscale or RGB
    img_arr = np.array(pil_img)
    
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        # RGB Image
        r_plain = img_arr[:, :, 0]
        g_plain = img_arr[:, :, 1]
        b_plain = img_arr[:, :, 2]
        
        # Encrypt channels
        r_cipher = ca.encrypt_image(r_plain, key)
        g_cipher = ca.encrypt_image(g_plain, key)
        b_cipher = ca.encrypt_image(b_plain, key)
        
        plain_results = {}
        cipher_results = {}
        
        for dir_name in ['horizontal', 'vertical', 'diagonal']:
            plain_vals = [
                calculate_correlation(r_plain, dir_name),
                calculate_correlation(g_plain, dir_name),
                calculate_correlation(b_plain, dir_name)
            ]
            cipher_vals = [
                calculate_correlation(r_cipher, dir_name),
                calculate_correlation(g_cipher, dir_name),
                calculate_correlation(b_cipher, dir_name)
            ]
            plain_results[dir_name] = np.mean(plain_vals)
            cipher_results[dir_name] = np.mean(cipher_vals)
            
        return plain_results, cipher_results
    else:
        # Grayscale Image
        if len(img_arr.shape) == 3:
            img_arr = img_arr[:, :, 0]
            
        cipher = ca.encrypt_image(img_arr, key)
        plain_results = {}
        cipher_results = {}
        
        for dir_name in ['horizontal', 'vertical', 'diagonal']:
            plain_results[dir_name] = calculate_correlation(img_arr, dir_name)
            cipher_results[dir_name] = calculate_correlation(cipher, dir_name)
            
        return plain_results, cipher_results

def main():
    key = b"IEEE_CA_Secure_Key_256_Bits_2026"
    results = []

    print("======================================================================")
    print("                 CORRELATION COEFFICIENTS CALCULATION                 ")
    print("======================================================================")

    for img_name in TEST_IMAGES:
        img_path = os.path.join("Images", img_name)
        if not os.path.exists(img_path):
            print(f"Skipping: {img_name} (File not found)")
            continue

        print(f"Calculating correlation for image: {img_name}...")
        plain_c, cipher_c = compute_image_correlations(img_path, key)
        
        results.append({
            "name": img_name,
            "plain_h": plain_c["horizontal"],
            "plain_v": plain_c["vertical"],
            "plain_d": plain_c["diagonal"],
            "cipher_h": cipher_c["horizontal"],
            "cipher_v": cipher_c["vertical"],
            "cipher_d": cipher_c["diagonal"]
        })

    # Save to CSV
    output_dir = "results/tables"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "Coreraltion_table.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Plain_H", "Plain_V", "Plain_D", "Cipher_H", "Cipher_V", "Cipher_D"])
        for r in results:
            writer.writerow([
                r["name"],
                f"{r['plain_h']:.4f}",
                f"{r['plain_v']:.4f}",
                f"{r['plain_d']:.4f}",
                f"{r['cipher_h']:.4f}",
                f"{r['cipher_v']:.4f}",
                f"{r['cipher_d']:.4f}"
            ])

    print(f"\nAll results successfully saved to: {csv_path}\n")

    # ANSI Colors
    GREEN_BOLD = "\033[1;32m"
    RED_BOLD = "\033[1;31m"
    RESET = "\033[0m"

    def fmt(val, is_plain=True):
        val_str = f"{val:+.4f}"
        # A good plaintext correlation is high (say >= 0.8)
        # A good ciphertext correlation is close to 0 (say abs(val) <= 0.05)
        if is_plain:
            is_good = val >= 0.8
        else:
            is_good = abs(val) <= 0.05
            
        if is_good:
            return f"{GREEN_BOLD}{val_str:<10}{RESET}"
        else:
            return f"{RED_BOLD}{val_str:<10}{RESET}"

    # Print formatted console table
    col_width = 10
    print("=" * 96)
    print(f"{'Image Name':<16} | {'Plain Image':^34}| {'Cipher Image':^34} |")
    print(f"{'':<16} | {'Horiz':<10} {'Vert':<10} {'Diag':<10}  | {'Horiz':<10} {'Vert':<10} {'Diag':<10}  |")
    print("-" * 96)
    for r in results:
        p_h = fmt(r["plain_h"], is_plain=True)
        p_v = fmt(r["plain_v"], is_plain=True)
        p_d = fmt(r["plain_d"], is_plain=True)
        c_h = fmt(r["cipher_h"], is_plain=False)
        c_v = fmt(r["cipher_v"], is_plain=False)
        c_d = fmt(r["cipher_d"], is_plain=False)
        print(f"{r['name']:<16} | {p_h} {p_v} {p_d} | {c_h} {c_v} {c_d} |")
    print("=" * 96 + "\n")

if __name__ == "__main__":
    main()
