import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import src.cellular_automata as ca

def calculate_npcr_uaci(img, key):
    """
    Computes NPCR and UACI by mutating exactly 1 pixel in the original image.
    """
    H, W = img.shape
    
    # 1. Modify exactly 1 pixel in the center
    img_mod = img.copy()
    cx, cy = H // 2, W // 2
    img_mod[cx, cy] = (int(img[cx, cy]) + 1) % 256
    
    # 2. Encrypt both using the same key
    c1 = ca.encrypt_image(img, key)
    c2 = ca.encrypt_image(img_mod, key)
    
    # 3. Calculate NPCR (Number of Pixels Change Rate)
    # D(i, j) = 1 if c1[i, j] != c2[i, j] else 0
    diff = (c1 != c2).astype(np.float64)
    npcr = np.mean(diff) * 100.0
    
    # 4. Calculate UACI (Unified Average Changing Intensity)
    abs_diff = np.abs(c1.astype(np.float64) - c2.astype(np.float64))
    uaci = np.mean(abs_diff / 255.0) * 100.0
    
    return npcr, uaci

def main():
    print("====================================================")
    print("Running Differential Attack Sensitivity Analysis...")
    print("====================================================")
    
    key = b"IEEE_CA_Secure_Key_256_Bits_2026"
    img = cv2.imread("test_image.png", cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: test_image.png not found. Run main.py first.")
        return
        
    print(f"Plaintext Image Shape: {img.shape[0]}x{img.shape[1]}")
    print("Mutating 1 pixel and executing twin encryptions...")
    
    npcr, uaci = calculate_npcr_uaci(img, key)
    
    print("\nDifferential Security Metrics:")
    print(f"  NPCR: {npcr:.6f}%  (Ideal: >99.6094%)")
    print(f"  UACI: {uaci:.6f}%  (Ideal: ~33.4635%)")
    
    # Determine security status
    if npcr > 99.6 and abs(uaci - 33.46) < 1.0:
        print("\nStatus: PASS (Algorithm shows high sensitivity to plaintext modifications)")
    else:
        print("\nStatus: WARNING (Sub-optimal differential resistance detected)")
    print("====================================================")

if __name__ == "__main__":
    main()
