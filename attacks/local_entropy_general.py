import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.cellular_automata as ca

def calculate_local_shannon_entropy(image, k=30, TB=1936, seed=42):
    """
    Computes Local Shannon Entropy (LSE) based on:
    - k: number of random non-overlapping blocks
    - TB: number of pixels per block (TB = 1936 corresponds to a 44x44 block)
    """
    H, W = image.shape
    block_side = int(np.sqrt(TB))
    
    # Deterministic RNG for consistency and comparison
    rng = np.random.default_rng(seed)
    
    entropies = []
    selected_blocks = []
    
    attempts = 0
    max_attempts = 2000
    while len(selected_blocks) < k and attempts < max_attempts:
        attempts += 1
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
            block = image[r:r+block_side, c:c+block_side]
            
            pixel_counts = np.bincount(block.flatten(), minlength=256)
            probs = pixel_counts / block.size
            probs = probs[probs > 0]
            block_entropy = -np.sum(probs * np.log2(probs))
            entropies.append(block_entropy)
            
    if len(entropies) < k:
        # Fallback if hard to pack tightly in smaller images (e.g. 256x256)
        # We allow slight overlap or select blocks using grid partitioning
        entropies = []
        for i in range(k):
            grid_r = (i * block_side) % (H - block_side)
            grid_c = ((i * block_side) // (H - block_side) * block_side) % (W - block_side)
            block = image[grid_r:grid_r+block_side, grid_c:grid_c+block_side]
            pixel_counts = np.bincount(block.flatten(), minlength=256)
            probs = pixel_counts / block.size
            probs = probs[probs > 0]
            entropies.append(-np.sum(probs * np.log2(probs)))
            
    mean_entropy = np.mean(entropies)
    return mean_entropy

def main():
    # List of files corresponding to standard benchmark images in the folder
    test_files = [
        "5.1.09.tiff", "5.1.10.tiff", "5.1.11.tiff", "5.1.12.tiff", "5.1.13.tiff", "5.1.14.tiff",
        "5.2.08.tiff", "5.2.09.tiff", "5.2.10.tiff",
        "5.3.01.tiff", "5.3.02.tiff",
        "7.1.01.tiff", "7.1.02.tiff", "7.1.03.tiff", "7.1.04.tiff", "7.1.05.tiff", "7.1.06.tiff", "7.1.07.tiff", "7.1.08.tiff", "7.1.09.tiff", "7.1.10.tiff",
        "7.2.01.tiff", "boat.512.tiff", "gray21.512.tiff", "ruler.512.tiff", "pepper.tiff", "lenna.jpg", "cameraman.jpg"
    ]
    
    img_dir = "Images"
    key = b"IEEE_CA_Secure_Key_256_Bits_2026"
    
    h_left = 7.901515698
    h_right = 7.903422936
    
    results = []
    
    print("Running general Local Shannon Entropy (LSE) test on benchmark suite...")
    
    for filename in test_files:
        path = os.path.join(img_dir, filename)
        if not os.path.exists(path):
            continue
            
        # Read as grayscale
        pil_img = Image.open(path).convert('L')
        gray_img = np.array(pil_img)
        
        # Encrypt the grayscale image
        cipher = ca.encrypt_image(gray_img, key)
        
        # Calculate LSE
        mean_lse = calculate_local_shannon_entropy(cipher, k=30, TB=1936)
        passed = h_left <= mean_lse <= h_right
        
        results.append({
            'name': filename.replace('.tiff', '').replace('.jpg', '').replace('.png', ''),
            'lse': mean_lse,
            'passed': passed
        })
        
    # Calculate statistics
    lse_values = [r['lse'] for r in results]
    mean_lse_all = np.mean(lse_values)
    pass_count = sum([1 for r in results if r['passed']])
    total_count = len(results)
    
    # Print the table
    print("\n" + "="*50)
    print(" LOCAL SHANNON ENTROPY (LSE) GENERAL ANALYSIS")
    print(f" Parameters: k = 30, TB = 1936, alpha = 0.001")
    print(f" Critical Interval: [{h_left:.9f}, {h_right:.9f}]")
    print("="*50)
    print(f"{'Image Name':<15} | {'Mean LSE':<12} | {'Status':<6}")
    print("-"*50)
    
    for r in results:
        status_str = "PASS" if r['passed'] else "FAIL"
        bold_prefix = "\033[1m" if r['passed'] else ""
        bold_suffix = "\033[0m" if r['passed'] else ""
        print(f"{r['name']:<15} | {bold_prefix}{r['lse']:.9f}{bold_suffix} | {status_str}")
        
    print("-"*50)
    print(f"Mean LSE:      {mean_lse_all:.9f}")
    print(f"Pass Rate:     {pass_count}/{total_count}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
