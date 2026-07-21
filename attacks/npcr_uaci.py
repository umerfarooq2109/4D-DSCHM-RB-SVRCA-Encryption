"""
NPCR (Number of Pixels Change Rate) and UACI (Unified Average Changing Intensity) Pixel-Change Sensitivity Test.

This script tests the differential attack resistance of our 4D Trigonometric Hyperchaotic Map & Cellular Automata Cryptosystem.
It mutates exactly one pixel at five distinct coordinates (top-left, top-right, center, bottom-left, bottom-right), 
encrypts both the original and mutated images using the same cryptographic key, and computes the NPCR and UACI metrics.

Command to run the code for a single image:
    python attacks/npcr_uaci.py Images/lenna.jpg
"""

import os
import sys
import numpy as np
from PIL import Image

# Reconfigure stdout to support UTF-8 (for drawing clean tables and symbols)
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
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def compute_npcr(c1, c2):
    diff = (c1 != c2)
    return (np.sum(diff) / diff.size) * 100.0

def compute_uaci(c1, c2):
    diff = np.abs(c1.astype(float) - c2.astype(float))
    return (np.sum(diff) / (255.0 * c1.size)) * 100.0

def run_npcr_uaci_test(image_path):
    print(f"\n{BOLD}{CYAN}Analyzing Image: {image_path}{RESET}")
    if not os.path.exists(image_path):
        print(f"{RED}Error: File {image_path} does not exist.{RESET}")
        return None
        
    try:
        # Load and convert image to grayscale
        pil_img = Image.open(image_path)
        img_arr = np.array(pil_img, dtype=np.uint8)
        
        if len(img_arr.shape) == 3:
            gray_img = np.array(pil_img.convert('L'), dtype=np.uint8)
        else:
            gray_img = img_arr
            
        M, N = gray_img.shape
        print(f"Image dimensions: {M}x{N}")
        
        key = b"IEEE_CA_Secure_Key_256_Bits_2026"
        
        # Encrypt the original image
        print("Encrypting original image...")
        c1 = ca.encrypt_image(gray_img, key)
        
        # Define 5 test positions (1-indexed for presentation, 0-indexed for python)
        # Positions: (1, 1), (1, N), (M//2+1, N//2+1), (M, 1), (M, N)
        positions = [
            (0, 0, "(1, 1)"),
            (0, N - 1, f"(1, {N})"),
            (M // 2, N // 2, f"({M // 2 + 1}, {N // 2 + 1})"),
            (M - 1, 0, f"({M}, 1)"),
            (M - 1, N - 1, f"({M}, {N})")
        ]
        
        results = []
        
        print("Evaluating NPCR and UACI for modified pixels...")
        for r, c, label in positions:
            orig_pixel = gray_img[r, c]
            # Perturb pixel by +1 (or -1 if pixel is 255)
            changed_pixel = orig_pixel + 1 if orig_pixel < 255 else orig_pixel - 1
            
            # Create perturbed image copy
            gray_img_mod = gray_img.copy()
            gray_img_mod[r, c] = changed_pixel
            
            # Encrypt modified image with key
            c2 = ca.encrypt_image(gray_img_mod, key)
            
            # Compute NPCR & UACI
            npcr_val = compute_npcr(c1, c2)
            uaci_val = compute_uaci(c1, c2)
            
            results.append({
                'position': label,
                'orig': int(orig_pixel),
                'changed': int(changed_pixel),
                'npcr': npcr_val,
                'uaci': uaci_val
            })
            
        # Calculate averages
        avg_npcr = np.mean([r['npcr'] for r in results])
        avg_uaci = np.mean([r['uaci'] for r in results])
        
        # Print table
        print(f"\n{BOLD}{'Position':<16} {'Original Pixel':<16} {'Changed Pixel':<16} {'NPCR (%)':<12} {'UACI (%)':<12}{RESET}")
        print("-" * 76)
        for r in results:
            print(f"{r['position']:<16} {r['orig']:<16} {r['changed']:<16} {r['npcr']:<12.4f} {r['uaci']:<12.4f}")
        print("-" * 76)
        
        # Determine colors for NPCR and UACI averages
        # NPCR threshold: 99.60%, UACI threshold: 33.15% - 33.75%
        npcr_color = GREEN if avg_npcr >= 99.6 else RED
        uaci_color = GREEN if (33.15 <= avg_uaci <= 33.75) else RED
        
        print(f"{BOLD}{'Avg.':<16} {'':<16} {'':<16} "
              f"{npcr_color}{avg_npcr:<12.4f}{RESET}{BOLD}"
              f"{uaci_color}{avg_uaci:<12.4f}{RESET}")
        print("-" * 76)
        
        # Overall status
        npcr_ok = avg_npcr >= 99.6
        uaci_ok = (33.15 <= avg_uaci <= 33.75)
        status_str = f"{GREEN}PASS{RESET}" if (npcr_ok and uaci_ok) else f"{RED}FAIL{RESET}"
        print(f"NPCR Status: {'OK' if npcr_ok else 'LOW'} | UACI Status: {'OK' if uaci_ok else 'LOW'} | Status: {status_str}\n")
        
        return {
            'image': image_path,
            'results': results,
            'avg_npcr': avg_npcr,
            'avg_uaci': avg_uaci,
            'passed': npcr_ok and uaci_ok
        }
        
    except Exception as e:
        print(f"{RED}Error analyzing image {image_path}: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print(f"{BOLD}========================================================================{RESET}")
    print(f"{BOLD}             NPCR and UACI Pixel-Change Sensitivity Test               {RESET}")
    print(f"{BOLD}========================================================================{RESET}")
    
    # If paths are provided via CLI arguments, use them, otherwise use defaults
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        # Default images in workspace
        image_paths = [
            "Images/pepper.tiff"
        ]
        
    for path in image_paths:
        run_npcr_uaci_test(path)

if __name__ == "__main__":
    main()
