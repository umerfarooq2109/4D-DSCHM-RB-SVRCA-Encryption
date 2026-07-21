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

def compute_chi_square(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    expected = img.size / 256.0
    chi_val = np.sum((hist - expected) ** 2 / expected)
    return chi_val

def compute_gvd(plain, cipher):
    plain = plain.astype(float)
    cipher = cipher.astype(float)
    
    def average_diff(img):
        # Calculate squared neighbor differences
        diff_up = (img[1:-1, 1:-1] - img[0:-2, 1:-1]) ** 2
        diff_down = (img[1:-1, 1:-1] - img[2:, 1:-1]) ** 2
        diff_left = (img[1:-1, 1:-1] - img[1:-1, 0:-2]) ** 2
        diff_right = (img[1:-1, 1:-1] - img[1:-1, 2:]) ** 2
        GN = (diff_up + diff_down + diff_left + diff_right) / 4.0
        return np.mean(GN)
        
    an_plain = average_diff(plain)
    an_cipher = average_diff(cipher)
    
    gvd_val = (an_cipher - an_plain) / (an_cipher + an_plain)
    return gvd_val

def compute_psnr(plain, cipher):
    mse = np.mean((plain.astype(float) - cipher.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    psnr_val = 10.0 * np.log10((255.0 ** 2) / mse)
    return psnr_val

def compute_entropy(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    prob = hist / img.size
    prob = prob[prob > 0]
    entropy_val = -np.sum(prob * np.log2(prob))
    return entropy_val

def analyze_image(name, path):
    if not os.path.exists(path):
        return None
        
    try:
        # Load and convert image to grayscale
        pil_img = Image.open(path)
        img_arr = np.array(pil_img, dtype=np.uint8)
        
        if len(img_arr.shape) == 3:
            gray_img = np.array(pil_img.convert('L'), dtype=np.uint8)
        else:
            gray_img = img_arr
            
        key = b"IEEE_CA_Secure_Key_256_Bits_2026"
        
        # Encrypt image
        cipher_img = ca.encrypt_image(gray_img, key)
        
        # Compute metrics
        chi2 = compute_chi_square(cipher_img)
        gvd = compute_gvd(gray_img, cipher_img)
        psnr = compute_psnr(gray_img, cipher_img)
        entropy = compute_entropy(cipher_img)
        
        return {
            'name': name,
            'chi2': chi2,
            'gvd': gvd,
            'psnr': psnr,
            'entropy': entropy
        }
    except Exception as e:
        print(f"{RED}Error analyzing {name}: {e}{RESET}")
        return None

def main():
    # Define candidates in the Images directory
    candidates = [
        ("Lenna", "Images/lena_gray_512.tif"),
        ("Cameraman", "Images/cameraman.tif"),
        ("Baboon", "Images/Baboon.png"),
        ("Pepper", "Images/4.2.07.tiff"), # pepper
    ]
    
    results = []
    used_names = set()
    
    # Process up to 4 images to match the standard paper size
    for name, path in candidates:
        if len(results) >= 4:
            break
        if os.path.exists(path) and name not in used_names:
            res = analyze_image(name, path)
            if res:
                results.append(res)
                used_names.add(name)
                
    if not results:
        print(f"{RED}No candidate images found in Images/ directory!{RESET}")
        return
        
    # Calculate Averages
    avg_chi2 = np.mean([r['chi2'] for r in results])
    avg_gvd = np.mean([r['gvd'] for r in results])
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_entropy = np.mean([r['entropy'] for r in results])
    
    # Display Table Header
    print(f"\n{BOLD}{BLUE}TABLE. Quantitative Encryption Security Metrics (Proposed Scheme){RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{'Image':<18} {'\u03c7\u00b2 (Chi-Square)':<18} {'GVD':<15} {'PSNR (dB)':<15} {'Entropy':<12}{RESET}")
    print(f"{BLUE}{'-'*80}{RESET}")
    
    # Printing results
    # Thresholds for good results:
    # Chi-Square: <= 293.2478
    # GVD: >= 0.90
    # PSNR: <= 10.0
    # Entropy: >= 7.9970
    for r in results:
        chi_color = GREEN if r['chi2'] <= 293.2478 else RED
        gvd_color = GREEN if r['gvd'] >= 0.90 else RED
        psnr_color = GREEN if r['psnr'] <= 10.0 else RED
        ent_color = GREEN if r['entropy'] >= 7.9970 else RED
        
        print(f"{r['name']:<18} "
              f"{chi_color}{r['chi2']:<18.4f}{RESET}"
              f"{gvd_color}{r['gvd']:<15.5f}{RESET}"
              f"{psnr_color}{r['psnr']:<15.4f}{RESET}"
              f"{ent_color}{r['entropy']:<12.4f}{RESET}")
              
    print(f"{BLUE}{'-'*80}{RESET}")
    
    avg_chi_color = GREEN if avg_chi2 <= 293.2478 else RED
    avg_gvd_color = GREEN if avg_gvd >= 0.90 else RED
    avg_psnr_color = GREEN if avg_psnr <= 10.0 else RED
    avg_ent_color = GREEN if avg_entropy >= 7.9970 else RED
    
    print(f"{BOLD}{'Avg.':<18} "
          f"{avg_chi_color}{avg_chi2:<18.4f}{RESET}{BOLD}"
          f"{avg_gvd_color}{avg_gvd:<15.5f}{RESET}{BOLD}"
          f"{avg_psnr_color}{avg_psnr:<15.4f}{RESET}{BOLD}"
          f"{avg_ent_color}{avg_entropy:<12.4f}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

if __name__ == "__main__":
    main()
