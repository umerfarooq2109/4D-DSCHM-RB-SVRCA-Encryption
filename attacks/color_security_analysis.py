import os
import sys
import numpy as np
import hashlib
from PIL import Image
from scipy.stats import chi2

# Reconfigure stdout to support UTF-8 (for drawing clean tables and symbols)
sys.stdout.reconfigure(encoding='utf-8')
# Add the project root to system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.cellular_automata as ca

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Critical value at alpha = 0.05, df = 255
CRITICAL_VALUE = 293.2478

def calculate_entropy(channel):
    """Calculate Information Entropy of a single 2D channel"""
    hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])
    prob = hist / channel.size
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def calculate_chi_square(channel):
    """Calculate Chi-Square statistic and P-value for a single 2D channel"""
    hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])
    E = channel.size / 256.0
    chi_stat = np.sum((hist - E) ** 2 / E)
    p_val = 1.0 - chi2.cdf(chi_stat, 255)
    return chi_stat, p_val

def compute_npcr(c1, c2):
    diff = (c1 != c2)
    return (np.sum(diff) / diff.size) * 100.0

def compute_uaci(c1, c2):
    diff = np.abs(c1.astype(float) - c2.astype(float))
    return (np.sum(diff) / (255.0 * c1.size)) * 100.0

def run_analysis(image_path):
    print(f"\n{BOLD}{CYAN}Running Quantitative Security Analysis for: {image_path}{RESET}")
    if not os.path.exists(image_path):
        print(f"{RED}Error: File {image_path} does not exist.{RESET}")
        return
        
    try:
        # 1. Load original image
        pil_img = Image.open(image_path)
        if pil_img.mode in ('RGBA', 'LA') or (isinstance(pil_img.info.get('transparency'), (bytes, int))):
            pil_img = pil_img.convert('RGB')
        img_arr = np.array(pil_img, dtype=np.uint8)
        
        if len(img_arr.shape) != 3 or img_arr.shape[2] != 3:
            print(f"{RED}Error: Image must be a 3-channel color image.{RESET}")
            return
            
        M, N, C = img_arr.shape
        
        # 2. Encrypt original image channel-by-channel
        print("Encrypting original image...")
        c_orig = np.zeros_like(img_arr)
        for i in range(3):
            chan = img_arr[:, :, i]
            K = hashlib.sha256(chan.tobytes()).digest()
            c_chan = ca.encrypt_image(chan, K)
            c_orig[:, :, i] = c_chan[:M, :N]
            
        print("Computing Entropy, Chi-Square, NPCR, and UACI per channel...")
        
        channels = ['Red Channel', 'Green Channel', 'Blue Channel']
        results = []
        
        for i, name in enumerate(channels):
            plain_chan = img_arr[:, :, i]
            cipher_chan = c_orig[:, :, i]
            
            # (a) Entropy
            p_ent = calculate_entropy(plain_chan)
            c_ent = calculate_entropy(cipher_chan)
            
            # (b) Chi-Square
            p_chi, p_pval = calculate_chi_square(plain_chan)
            c_chi, c_pval = calculate_chi_square(cipher_chan)
            
            # (c) NPCR and UACI under single-pixel change
            # We perturb pixel (0, 0) in the plaintext channel by +1 (or -1 if it is 255)
            orig_pixel = plain_chan[0, 0]
            changed_pixel = orig_pixel + 1 if orig_pixel < 255 else orig_pixel - 1
            
            plain_chan_mod = plain_chan.copy()
            plain_chan_mod[0, 0] = changed_pixel
            
            # Derive new key from modified channel hash
            K_mod = hashlib.sha256(plain_chan_mod.tobytes()).digest()
            
            # Encrypt modified channel
            c_chan_mod = ca.encrypt_image(plain_chan_mod, K_mod)
            c_chan_mod_cropped = c_chan_mod[:M, :N]
            
            npcr_val = compute_npcr(cipher_chan, c_chan_mod_cropped)
            uaci_val = compute_uaci(cipher_chan, c_chan_mod_cropped)
            
            results.append({
                'channel': name,
                'plain_entropy': p_ent,
                'cipher_entropy': c_ent,
                'plain_chi': p_chi,
                'cipher_chi': c_chi,
                'cipher_pval': c_pval,
                'npcr': npcr_val,
                'uaci': uaci_val
            })
            
        # ==========================================
        # PRINT RESULTS TO CONSOLE
        # ==========================================
        print(f"\n{BOLD}Statistical Metrics (Entropy and Chi-Square):{RESET}")
        print(f"{'Channel':<15} {'Plain Entropy':<15} {'Cipher Entropy':<15} {'Plain Chi-Square':<18} {'Cipher Chi-Square':<18} {'P-Value':<12}")
        print("-" * 97)
        for r in results:
            print(f"{r['channel']:<15} "
                  f"{r['plain_entropy']:<15.6f} "
                  f"{r['cipher_entropy']:<15.6f} "
                  f"{r['plain_chi']:<18.4f} "
                  f"{r['cipher_chi']:<18.4f} "
                  f"{r['cipher_pval']:<12.4f}")
        print("-" * 97)
        
        print(f"\n{BOLD}Differential Metrics (NPCR and UACI under 1-pixel change):{RESET}")
        print(f"{'Channel':<15} {'NPCR (%)':<15} {'UACI (%)':<15} {'Ideal NPCR':<15} {'Ideal UACI':<15}")
        print("-" * 79)
        for r in results:
            print(f"{r['channel']:<15} "
                  f"{r['npcr']:<15.4f} "
                  f"{r['uaci']:<15.4f} "
                  f"{99.6094:<15.4f} "
                  f"{33.4635:<15.4f}")
        print("-" * 79)
        
    except Exception as e:
        print(f"{RED}Error running security analysis: {e}{RESET}")
        import traceback
        traceback.print_exc()

def main():
    image_path = "Images/pepper.tiff"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    run_analysis(image_path)

if __name__ == "__main__":
    main()
