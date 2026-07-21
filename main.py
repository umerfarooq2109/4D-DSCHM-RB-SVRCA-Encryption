import os
import cv2
import time
import numpy as np
import sys

# Target lists of test images
TEST_IMAGES = [

    '4.1.01.png',
    '4.1.01.tiff',
    '4.1.02.png',
    '4.1.02.tiff',
    '4.1.03.png',
    '4.1.03.tiff',
    '4.1.04.tiff',
    '4.1.05.tiff',
    '4.1.06.tiff',
    '4.1.07.tiff',
    '4.1.08.tiff',
    '4.2.01.tiff',
    '4.2.03.tiff',
    '4.2.05.tiff',
    '4.2.06.tiff',
    '4.2.07.tiff',
    '5.1.09.tiff',
    '5.1.10.tiff',
    '5.1.11.png',
    '5.1.11.tiff',
    '5.1.12.tiff',
    '5.1.13.tiff',
    '5.1.14.tiff',
    '5.2.08.tiff',
    '5.2.09.tiff',
    '5.2.10.tiff',
    '5.3.01.tiff',
    '5.3.02.tiff',
    '7.1.01.tiff',
    '7.1.02.tiff',
    '7.1.03.tiff',
    '7.1.04.tiff',
    '7.1.05.tiff',
    '7.1.06.tiff',
    '7.1.07.tiff',
    '7.1.08.tiff',
    '7.1.09.tiff',
    '7.1.10.tiff',
    '7.2.01.tiff',
    'Baboon.png',
    'Black.png',
    'White.png',
    'aeroplane.png',
    'aeroplane.tiff',
    'b.jpg',
    'black1.png',
    'boat.512.tiff',
    'cameraman.jpg',
    'girl.tiff',
    'gray21.512.tiff',
    'house.tiff',
    'jet.tiff',
    'lenna.jpg',
    'pepper.tiff',
        '1024.png',
    'ruler.512.tiff',
]

# Core Cryptosystem
from src.utils import create_directories, generate_synthetic_image, calculate_mse
import src.cellular_automata as ca

def calculate_entropy(img):
    """Calculates Shannon information entropy."""
    pixel_counts = np.bincount(img.flatten(), minlength=256)
    probabilities = pixel_counts / len(img.flatten())
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_npcr_uaci(img, key):
    """Calculates NPCR and UACI sensitivity metrics by mutating 1 pixel."""
    H, W = img.shape
    
    # Modify 1 pixel at center
    img_mod = img.copy()
    cx, cy = H // 2, W // 2
    img_mod[cx, cy] = (int(img[cx, cy]) + 1) % 256
    
    # Encrypt both
    c1 = ca.encrypt_image(img, key)
    c2 = ca.encrypt_image(img_mod, key)
    
    # Compute NPCR
    diff = (c1 != c2).astype(np.float64)
    npcr = np.mean(diff) * 100.0
    
    # Compute UACI
    abs_diff = np.abs(c1.astype(np.float64) - c2.astype(np.float64))
    uaci = np.mean(abs_diff / 255.0) * 100.0
    
    return npcr, uaci

def format_npcr(val):
    val_str = f"{val:.4f}%"
    padded = f"{val_str:<12}"
    if val >= 99.6094:
        return f"\033[1;32m{padded}\033[0m"  # Bold Green
    else:
        return f"\033[1;31m{padded}\033[0m"  # Bold Red

def format_uaci(val):
    val_str = f"{val:.4f}%"
    padded = f"{val_str:<12}"
    if 33.15 <= val <= 33.75:
        return f"\033[1;32m{padded}\033[0m"  # Bold Green
    else:
        return f"\033[1;31m{padded}\033[0m"  # Bold Red

def format_entropy(val):
    val_str = f"{val:.6f}"
    padded = f"{val_str:<12}"
    if val >= 7.990000:
        return f"\033[1;32m{padded}\033[0m"  # Bold Green
    else:
        return f"\033[1;31m{padded}\033[0m"  # Bold Red

def format_mse(val):
    val_str = f"{val:.4f}"
    padded = f"{val_str:<10}"
    if val == 0.0:
        return f"\033[1;32m{padded}\033[0m"  # Bold Green
    else:
        return f"\033[1;31m{padded}\033[0m"  # Bold Red

def format_psnr(val):
    if np.isinf(val) or val == float('inf'):
        val_str = "inf"
    else:
        val_str = f"{val:.2f} dB"
    padded = f"{val_str:<12}"
    if np.isinf(val) or val == float('inf'):
        return f"\033[1;32m{padded}\033[0m"  # Bold Green
    else:
        return f"\033[1;31m{padded}\033[0m"  # Bold Red

def format_time(val):
    val_str = f"{val:.3f}s"
    padded = f"{val_str:<9}"
    return f"\033[1m{padded}\033[0m"  # Bold white/default

def main():
    # Setup directories
    create_directories()
    
    # Parse inputs to find targets
    targets = []
    if len(sys.argv) > 1:
        targets.append(sys.argv[1])
    else:
        # Load from TEST_IMAGES list
        for img_name in TEST_IMAGES:
            targets.append(os.path.join("Images", img_name))
            
    # Fallback to test_image.png if list is empty
    if not targets:
        targets.append("test_image.png")
        
    print("+------------------------------------------------------------------+")
    print("|       4D trigonometric HYPERCHAOTIC MAP EXPERIMENT SUITE         |")
    print("+------------------------------------------------------------------+")
    print(f"\nFound {len(targets)} test images to process.\n")
    
    key = b"IEEE_CA_Secure_Key_256_Bits_2026"
    results_rows = []
    
    for idx, input_path in enumerate(targets):
        filename = os.path.basename(input_path)
        print("======================================================================")
        print(f"  [{idx+1}/{len(targets)}] {filename}")
        print("======================================================================")
        
        # Load image
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            if input_path == "test_image.png":
                img = generate_synthetic_image()
                cv2.imwrite(input_path, img)
            else:
                print(f"  \033[1;31m[ERROR] Could not read image at '{input_path}'. Skipping.\033[0m\n")
                continue
                
        print(f"  Loaded successfully. Shape: {img.shape[0]}x{img.shape[1]}")
        
        # Run encryption and decryption & measure latency
        print("  Running core encryption & decryption...")
        t0 = time.time()
        cipher = ca.encrypt_image(img, key)
        decrypted = ca.decrypt_image(cipher, key, original_shape=img.shape)
        elapsed = time.time() - t0
        
        # Save ciphertext and decrypted images with enc_ and dec_ prefixes
        base_name = os.path.splitext(filename)[0]
        enc_filename = f"enc_{base_name}.png"
        dec_filename = f"dec_{base_name}.png"
        
        cv2.imwrite(os.path.join("results/encrypted", enc_filename), cipher)
        cv2.imwrite("ciphertext.png", cipher) # Keep root copy for attack script compatibility
        cv2.imwrite(os.path.join("results/decrypted", dec_filename), decrypted)
        
        # Calculate performance and security values
        print("  Evaluating NPCR & UACI Plaintext Sensitivity...")
        npcr, uaci = calculate_npcr_uaci(img, key)
        
        print("  Calculating Shannon Entropy & Quality MSE...")
        entropy = calculate_entropy(cipher)
        mse = calculate_mse(img, decrypted)
        if mse == 0.0:
            psnr = float('inf')
        else:
            psnr = 10.0 * np.log10((255.0**2) / mse)
            
        status_str = "\033[1;32m[OK]\033[0m" if mse == 0.0 else "\033[1;31m[FAIL]\033[0m"
        print(f"  Lossless Decryption: {status_str} (MSE = {mse:.4f}, PSNR = {psnr if psnr == float('inf') else f'{psnr:.2f} dB'})")
        print(f"  Metrics: NPCR = {npcr:.2f}%, UACI = {uaci:.3f}%, Entropy = {entropy:.4f}, Time = {elapsed:.2f}s\n")
        
        results_rows.append({
            'filename': filename,
            'size': f"{img.shape[0]} X {img.shape[1]}",
            'npcr': npcr,
            'uaci': uaci,
            'entropy': entropy,
            'mse': mse,
            'psnr': psnr,
            'time': elapsed
        })
        
    if not results_rows:
        print("Error: No images were successfully processed.")
        return
        
    # Output clean summary results table
    print("\n===========================================================================================")
    print("                                      SUMMARY RESULTS")
    print("===========================================================================================")
    print(f"{'Image':<15} {'Size':<15} {'NPCR(%)':<12} {'UACI(%)':<12} {'Entropy':<12} {'MSE':<6} {'PSNR':<12} {'Time':<9}")
    print("-------------------------------------------------------------------------------------------")
    for row in results_rows:
        print(f"{row['filename']:<15} "
              f"{row['size']:<15} "
              f"{format_npcr(row['npcr'])}"
              f"{format_uaci(row['uaci'])}"
              f"{format_entropy(row['entropy'])}"
              f"{format_mse(row['mse'])}"
              f"{format_psnr(row['psnr'])}"
              f"{format_time(row['time'])}")
    print("===========================================================================================\n")

    # Save to CSV and JSON inside results/tables/
    os.makedirs("results/tables", exist_ok=True)
    
    import csv
    csv_path = "results/tables/results.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Size', 'NPCR(%)', 'UACI(%)', 'Entropy', 'MSE', 'PSNR', 'Time'])
        for row in results_rows:
            writer.writerow([
                row['filename'],
                row['size'],
                f"{row['npcr']:.6f}",
                f"{row['uaci']:.6f}",
                f"{row['entropy']:.6f}",
                f"{row['mse']:.6f}",
                "inf" if np.isinf(row['psnr']) else f"{row['psnr']:.6f}",
                f"{row['time']:.3f}s"
            ])
            
    import json
    json_path = "results/tables/results.json"
    json_data = []
    for r in results_rows:
        json_data.append({
            'filename': r['filename'],
            'size': r['size'],
            'npcr': r['npcr'],
            'uaci': r['uaci'],
            'entropy': r['entropy'],
            'mse': r['mse'],
            'psnr': "inf" if np.isinf(r['psnr']) else r['psnr'],
            'time': f"{r['time']:.3f}s"
        })
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)
        
    # Yellow warning message line with green folder and red filenames (using safe ASCII symbols)
    print(f"\033[1;33m[!] All Results saved in folder \033[1;32mresults/tables\033[1;33m with file name \033[1;31mresults.csv\033[1;33m and \033[1;31mresults.json\033[0m\n")

if __name__ == "__main__":
    main()
