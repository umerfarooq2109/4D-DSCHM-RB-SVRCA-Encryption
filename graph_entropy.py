"""
Local Shannon Entropy (LSE) 100-Run Plot Generator.
Mutates 1 pixel in the plaintext Lena image at 100 different positions,
encrypts them to generate 100 ciphertexts, calculates their local Shannon entropy (k=30, TB=1936),
and generates a publication-quality plot with a threshold line at 7.90.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

# Ensure the local workspace packages can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.cellular_automata as ca

def calculate_local_shannon_entropy(image, k=30, TB=1936, seed=42):
    H, W = image.shape
    block_side = int(np.sqrt(TB))
    
    # Deterministic RNG for block selection consistency
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
        entropies = []
        for i in range(k):
            grid_r = (i * block_side) % (H - block_side)
            grid_c = ((i * block_side) // (H - block_side) * block_side) % (W - block_side)
            block = image[grid_r:grid_r+block_side, grid_c:grid_c+block_side]
            pixel_counts = np.bincount(block.flatten(), minlength=256)
            probs = pixel_counts / block.size
            probs = probs[probs > 0]
            entropies.append(-np.sum(probs * np.log2(probs)))
            
    return np.mean(entropies)

def main():
    images_dir = "Images"
    key = b"IEEE_CA_Secure_Key_256_Bits_2026"
    
    # Locate Lena image
    lena_file = None
    for f in os.listdir(images_dir):
        if "lena" in f.lower() and "512" in f.lower():
            lena_file = os.path.join(images_dir, f)
            break
    if not lena_file:
        for f in os.listdir(images_dir):
            if "lena" in f.lower():
                lena_file = os.path.join(images_dir, f)
                break
                
    if not lena_file:
        print("Error: Lena image not found in Images folder.")
        return
        
    print(f"Loading Lena image: {lena_file}")
    pil_img = Image.open(lena_file).convert('L')
    plain_gray = np.array(pil_img)
    H, W = plain_gray.shape
    
    num_tests = 100
    lse_scores = []
    
    # Use a fixed seed for reproducible pixel modifications
    rng = np.random.default_rng(999)
    
    print(f"Running {num_tests} LSE test runs with 1-bit/1-pixel differences...")
    for t in range(1, num_tests + 1):
        # Mutate exactly one pixel at a random location
        r_idx = rng.integers(0, H)
        c_idx = rng.integers(0, W)
        
        mutated_plain = plain_gray.copy()
        # Modify the pixel value (toggle LSB or add 1)
        mutated_plain[r_idx, c_idx] = (int(mutated_plain[r_idx, c_idx]) + 1) % 256
        
        # Encrypt the mutated image
        cipher = ca.encrypt_image(mutated_plain, key)
        
        # Calculate LSE
        lse = calculate_local_shannon_entropy(cipher, k=30, TB=1936, seed=42)
        lse_scores.append(lse)
        
        if t % 10 == 0:
            print(f"  Completed {t}/{num_tests} runs...")
            
    # Figure Generation
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    tests_range = np.arange(1, num_tests + 1)
    
    # Plot Local Entropy scores (Blue line with circular dots)
    ax.plot(tests_range, lse_scores, 'o-', color='#1f77b4', linewidth=1.0,
            markersize=3.5, label='Local Entropy score')
            
    # Plot threshold line at Y = 7.900 (Red horizontal line)
    ax.axhline(y=7.900, color='#e74c3c', linestyle='-', linewidth=1.2,
               label=r'$\alpha=7.90$')
               
    ax.set_xlabel('Test')
    ax.set_ylabel('Local Entropy')
    ax.set_xlim(0, 101)
    # Match limits around the screenshot (7.899 to 7.906)
    ax.set_ylim(7.899, 7.906)
    
    # Tick marks pointing inside
    ax.tick_params(direction='in', top=True, right=True)
    
    # Add Grid
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "local_entropy_100_runs.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nSuccessfully generated and saved local entropy plot to: {save_path}")
    print(f"Mean LSE: {np.mean(lse_scores):.6f}")
    print(f"Min LSE: {np.min(lse_scores):.6f}")
    print(f"Max LSE: {np.max(lse_scores):.6f}")

if __name__ == "__main__":
    main()
