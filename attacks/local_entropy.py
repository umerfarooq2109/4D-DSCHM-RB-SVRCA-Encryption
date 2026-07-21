import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.cellular_automata as ca

def calculate_local_shannon_entropy(image, k=30, TB=1936):
    """
    Computes Local Shannon Entropy (LSE) based on:
    - k: number of random non-overlapping blocks
    - TB: number of pixels per block (TB = 1936 corresponds to a 44x44 block)
    """
    H, W = image.shape
    block_side = int(np.sqrt(TB))
    
    # We will choose k non-overlapping blocks randomly but using a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    
    entropies = []
    selected_blocks = []
    
    # Try to find k non-overlapping blocks
    attempts = 0
    max_attempts = 1000
    while len(selected_blocks) < k and attempts < max_attempts:
        attempts += 1
        # Random top-left corner
        r = rng.integers(0, H - block_side)
        c = rng.integers(0, W - block_side)
        
        # Check overlap
        overlap = False
        for (sr, sc) in selected_blocks:
            if not (r + block_side <= sr or sr + block_side <= r or
                    c + block_side <= sc or sc + block_side <= c):
                overlap = True
                break
                
        if not overlap:
            selected_blocks.append((r, c))
            # Extract block
            block = image[r:r+block_side, c:c+block_side]
            
            # Compute standard Shannon entropy for this block
            pixel_counts = np.bincount(block.flatten(), minlength=256)
            probs = pixel_counts / block.size
            probs = probs[probs > 0]
            block_entropy = -np.sum(probs * np.log2(probs))
            entropies.append(block_entropy)
            
    mean_entropy = np.mean(entropies)
    return mean_entropy, entropies

def main():
    image_path = "Images/pepper.tiff"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return
        
    pil_img = Image.open(image_path).convert('RGB')
    plain_rgb = np.array(pil_img)
    M, N = plain_rgb.shape[:2]
    
    key = b"IEEE_CA_Secure_Key_256_Bits_2026"
    
    # Encrypt channels
    print("Encrypting R, G, B channels of the image...")
    r_enc = ca.encrypt_image(plain_rgb[:, :, 0], key)
    g_enc = ca.encrypt_image(plain_rgb[:, :, 1], key)
    b_enc = ca.encrypt_image(plain_rgb[:, :, 2], key)
    
    # LSE Critical interval for k=30, TB=1936, alpha=0.001
    h_left = 7.901515698
    h_right = 7.903422936
    
    print("\n--- LOCAL SHANNON ENTROPY (LSE) TEST RESULTS ---")
    print(f"Parameters: k = 30, TB = 1936 (44x44 blocks), alpha = 0.001")
    print(f"Critical Interval: [{h_left:.9f}, {h_right:.9f}]\n")
    
    for name, channel in [("Red Channel  ", r_enc), ("Green Channel", g_enc), ("Blue Channel ", b_enc)]:
        mean_lse, all_lse = calculate_local_shannon_entropy(channel, k=30, TB=1936)
        passed = h_left <= mean_lse <= h_right
        status = "PASS" if passed else "FAIL"
        print(f"{name} LSE: {mean_lse:.9f} | Status: {status}")

if __name__ == "__main__":
    main()
