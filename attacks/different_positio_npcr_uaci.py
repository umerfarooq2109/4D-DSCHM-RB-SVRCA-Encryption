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

# ANSI escape codes for coloring text
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def compute_npcr(c1, c2):
    diff = (c1 != c2)
    return (np.sum(diff) / diff.size) * 100.0

def compute_uaci(c1, c2):
    diff = np.abs(c1.astype(float) - c2.astype(float))
    return (np.sum(diff) / (255.0 * c1.size)) * 100.0

def run_different_position_test(image_path):
    if not os.path.exists(image_path):
        print(f"{RED}Error: File {image_path} does not exist.{RESET}")
        return
        
    try:
        # Load and convert image to grayscale
        pil_img = Image.open(image_path)
        img_arr = np.array(pil_img, dtype=np.uint8)
        
        # If it is color, convert it to grayscale for standard analysis
        if len(img_arr.shape) == 3:
            gray_img = np.array(pil_img.convert('L'), dtype=np.uint8)
        else:
            gray_img = img_arr
            
        H, W = gray_img.shape
        
        key = b"IEEE_CA_Secure_Key_256_Bits_2026"
        
        # Encrypt the original image
        c1 = ca.encrypt_image(gray_img, key)
        
        # Define the 5 positions as shown in the paper (1-indexed for presentation, 0-indexed for python)
        positions = [
            (0, 0, "(1,1)"),
            (0, W - 1, f"(1,{W})"),
            (H // 2, W // 2, f"({H//2},{W//2})"),
            (H - 1, 0, f"({H},1)"),
            (H - 1, W - 1, f"({H},{W})")
        ]
        
        results = []
        
        for r, c, label in positions:
            orig_pixel = gray_img[r, c]
            # Perturb pixel by +1 (or -1 if pixel is 255)
            changed_pixel = orig_pixel + 1 if orig_pixel < 255 else orig_pixel - 1
            
            # Create perturbed image copy
            gray_img_mod = gray_img.copy()
            gray_img_mod[r, c] = changed_pixel
            
            # Encrypt modified image
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
        
        
        # Calculate dynamic statistical critical values based on image size H x W (N = H * W)
        N = H * W
        
        # 1. NPCR Critical Value at alpha = 0.05 (One-tailed test, Z = 1.6449)
        # Formula: (255 - Z_alpha * sqrt(255/N)) / 256 * 100
        npcr_crit = (255 - 1.6449 * np.sqrt(255.0 / N)) / 256.0 * 100.0
        
        # 2. UACI Critical Interval at alpha = 0.05 (Two-tailed test, Z = 1.96)
        # Standard deviation of UACI is sqrt(2.67402 / N) * 100
        uaci_mean = 33.4635
        uaci_std = np.sqrt(2.67402 / N) * 100.0
        uaci_lower = uaci_mean - 1.96 * uaci_std
        uaci_upper = uaci_mean + 1.96 * uaci_std
        
        # Display Table Header
        print(f"\n{BOLD}{BLUE}TABLE. NPCR and UACI when pixel value changed in different position for {os.path.basename(image_path)} ({H} \u00d7 {W}).{RESET}")
        print(f"{BLUE}{'='*85}{RESET}")
        print(f"{BOLD}{BLUE}{'Position':<15} {'Original Pixel':<18} {'Changed Pixel':<18} {'NPCR (%)':<16} {'UACI (%)':<16}{RESET}")
        print(f"{BLUE}{'-'*85}{RESET}")
        
        for r in results:
            npcr_color = GREEN if r['npcr'] >= npcr_crit else RED
            uaci_color = GREEN if (uaci_lower <= r['uaci'] <= uaci_upper) else RED
            
            print(f"{r['position']:<15} {r['orig']:<18} {r['changed']:<18} "
                  f"{npcr_color}{r['npcr']:<16.4f}{RESET}"
                  f"{uaci_color}{r['uaci']:<16.4f}{RESET}")
                  
        print(f"{BLUE}{'-'*85}{RESET}")
        
        avg_npcr_color = GREEN if avg_npcr >= npcr_crit else RED
        avg_uaci_color = GREEN if (uaci_lower <= avg_uaci <= uaci_upper) else RED
        
        print(f"{BOLD}{'Avg.':<15} {'-':<18} {'-':<18} "
              f"{avg_npcr_color}{avg_npcr:<16.4f}{RESET}{BOLD}"
              f"{avg_uaci_color}{avg_uaci:<16.4f}{RESET}")
        print(f"{BLUE}{'='*85}{RESET}\n")
        
    except Exception as e:
        print(f"{RED}Error running test: {e}{RESET}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default to Lena if it exists, otherwise Pepper
        candidates = [
            "Images/lena_gray_512.tif",
            "Images/lena_color_512.tif",
            "Images/pepper.tiff",
            "test_image.png"
        ]
        image_path = None
        for cand in candidates:
            if os.path.exists(cand):
                image_path = cand
                break
        if image_path is None:
            print(f"{RED}No candidate image found in Images/ directory!{RESET}")
            return
            
    run_different_position_test(image_path)

if __name__ == "__main__":
    main()
