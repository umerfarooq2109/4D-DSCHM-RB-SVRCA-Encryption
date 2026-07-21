"""
Chi-Square Goodness-of-Fit Test for 100 Encrypted Images (Lena & Pepper/Vegetables).
Calculates the mean Chi-Square statistic and count of values exceeding 295.25
across 100 keys for R, G, and B channels, then outputs the comparative LaTeX table.
"""

import os
import sys
import numpy as np
import hashlib
from PIL import Image

# Ensure the local workspace packages can be imported from parent root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.cellular_automata as ca

def calculate_chi_square(channel):
    H, W = channel.shape
    observed = np.bincount(channel.flatten(), minlength=256)
    expected = (H * W) / 256.0
    chi_sq = np.sum((observed - expected)**2 / expected)
    return chi_sq

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    images_dir = os.path.join(project_dir, "Images")
    
    lena_path = None
    veg_path = None
    for f in os.listdir(images_dir):
        if "lena" in f.lower():
            lena_path = os.path.join(images_dir, f)
            break
    for f in os.listdir(images_dir):
        if "pepper" in f.lower() or "4.2.07" in f.lower():
            veg_path = os.path.join(images_dir, f)
            break

    if not lena_path or not veg_path:
        print(f"Error: Could not locate Lena (found: {lena_path}) or Vegetables (found: {veg_path}) images.")
        return

    images = {
        "Lena": lena_path,
        "Vegetables": veg_path
    }

    results = {}
    
    # We run 100 evaluations
    num_evaluations = 100
    threshold = 295.25

    for category, img_path in images.items():
        if not os.path.exists(img_path):
            print(f"Error: {img_path} not found.")
            return

        print(f"Processing {category} ({img_path})...")
        pil_img = Image.open(img_path).convert('RGB')
        img_arr = np.array(pil_img)
        
        r_chan = img_arr[:, :, 0]
        g_chan = img_arr[:, :, 1]
        b_chan = img_arr[:, :, 2]

        chi_squares = []

        for i in range(1, num_evaluations + 1):
            # Generate deterministic key for evaluation
            key_bytes = hashlib.sha256(f"key_evaluation_{i}".encode()).digest()
            
            # Encrypt R, G, B channels
            r_enc = ca.encrypt_image(r_chan, key_bytes)
            g_enc = ca.encrypt_image(g_chan, key_bytes)
            b_enc = ca.encrypt_image(b_chan, key_bytes)
            
            # Compute chi-square for each channel
            chi_squares.append(calculate_chi_square(r_enc))
            chi_squares.append(calculate_chi_square(g_enc))
            chi_squares.append(calculate_chi_square(b_enc))
            
            if i % 10 == 0:
                print(f"  Completed {i}/{num_evaluations} runs...")

        mean_chi = np.mean(chi_squares)
        
        # We have 300 total channel evaluations. Count failures, normalize to out of 100 (per image run basis)
        failures = sum(1 for c in chi_squares if c > threshold)
        normalized_failures = failures / 3.0
        
        results[category] = {
            "mean": mean_chi,
            "failures": normalized_failures
        }

    # Reference values from comparison papers (Lena & Vegetables/Pepper)
    # Ref [11] Lena: Mean=250.37, Count=2; Vegetables: Mean=253.77, Count=3
    # Ref [14] Lena: Mean=252.47, Count=2; Vegetables: Mean=255.76, Count=4
    # Ref [32] Lena: Mean=257.92, Count=8; Vegetables: Mean=253.80, Count=5
    # Ref [42] Lena: Mean=254.42, Count=3; Vegetables: Mean=256.00, Count=2
    
    ref_data = {
        "Ref. [11]": {"Lena": (250.37, 2), "Vegetables": (253.77, 3)},
        "Ref. [14]": {"Lena": (252.47, 2), "Vegetables": (255.76, 4)},
        "Ref. [32]": {"Lena": (257.92, 8), "Vegetables": (253.80, 5)},
        "Ref. [42]": {"Lena": (254.42, 3), "Vegetables": (256.00, 2)}
    }

    # Calculate Proposed averages
    prop_lena_mean = results["Lena"]["mean"]
    prop_lena_fail = results["Lena"]["failures"]
    prop_veg_mean = results["Vegetables"]["mean"]
    prop_veg_fail = results["Vegetables"]["failures"]
    
    prop_avg_mean = (prop_lena_mean + prop_veg_mean) / 2.0
    prop_avg_fail = (prop_lena_fail + prop_veg_fail) / 2.0

    # Color escape sequences
    GREEN = "\033[1;32m"
    RED = "\033[1;31m"
    RESET = "\033[0m"

    def format_mean(val, width=18):
        val_str = f"{val:.2f}"
        padded = f"{val_str:<{width}}"
        # For mean chi-square, closer to 255 is better, < 256 is excellent
        if val < 256.0:
            return f"{GREEN}{padded}{RESET}"
        else:
            return f"{RED}{padded}{RESET}"

    def format_fail(val, width=20):
        val_str = f"{val:.2f}"
        padded = f"{val_str:<{width}}"
        # For failures, <= 5 is statistically expected at alpha=0.05
        if val <= 5.0:
            return f"{GREEN}{padded}{RESET}"
        else:
            return f"{RED}{padded}{RESET}"

    print("\n" + "="*60)
    print("                PROPOSED METHOD RESULTS ONLY")
    print("="*60)
    print(f"{'Image Category':<20} {'Mean Chi-Square':<18} {'Failures (>295.25)':<20}")
    print("-"*60)
    print(f"{'Lena':<20} {format_mean(prop_lena_mean)} {format_fail(prop_lena_fail)}")
    print(f"{'Vegetables':<20} {format_mean(prop_veg_mean)} {format_fail(prop_veg_fail)}")
    print(f"{'Average':<20} {format_mean(prop_avg_mean)} {format_fail(prop_avg_fail)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
