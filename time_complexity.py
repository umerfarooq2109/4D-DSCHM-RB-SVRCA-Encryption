"""
Time Complexity & Performance Profiling of 4D Trigonometric Hyperchaotic Cellular Automata Cryptosystem.

Profiles both the encryption and decryption pipelines step-by-step,
dynamically highlighting the absolute slowest step in RED and the total times in GREEN.
It also processes all batch test images, displaying their size and time, and outputs
the overall average execution time in Bold Yellow.
"""

import os
import sys
import time
import cv2
import numpy as np
import hashlib

# Initialize virtual terminal processing for ANSI escape sequences on Windows
if sys.platform == 'win32':
    os.system('')

# Ensure the local workspace packages can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.cellular_automata as ca
from src.utils import generate_synthetic_image

# Hardcoded target lists of test images from main.py for standalone compatibility
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

def profile_encryption_decryption(image_path, key_bytes):
    # ANSI escape colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    
    enc_times = {}
    dec_times = {}
    
    # Load Image
    t_load = time.perf_counter()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    enc_times['Step 0: Load Image'] = time.perf_counter() - t_load
    
    H_orig, W_orig = img.shape
    
    # ---------------- ENCRYPTION PROFILE ----------------
    
    # Step 1: Padding & Shape Setup
    t0 = time.perf_counter()
    pad_h = H_orig % 2
    pad_w = W_orig % 2
    if pad_h or pad_w:
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
    else:
        padded_img = img.copy()
    H, W = padded_img.shape
    N = H * W
    enc_times['Step 1: Padding & Shape Setup'] = time.perf_counter() - t0
    
    # Step 2: Chaotic Sequences Generation
    t0 = time.perf_counter()
    img_hash = hashlib.sha256(padded_img.tobytes()).digest()
    combined_key = bytes(a ^ b for a, b in zip(key_bytes, img_hash))
    x0, y0, z0, w0 = ca.key_to_initial_states(combined_key)
    xs, ys, zs, ws = ca.generate_chaotic_sequences(H, W, x0, y0, z0, w0)
    enc_times['Step 2: Chaotic Sequences Generation'] = time.perf_counter() - t0
    
    # Step 3: Position Scrambling
    t0 = time.perf_counter()
    row_idx = np.argsort(xs[:H])
    col_idx = np.argsort(ys[:W])
    scrambled = padded_img[row_idx, :][:, col_idx]
    enc_times['Step 3: Position Scrambling'] = time.perf_counter() - t0
    
    # Step 4: CA Rules & Keys Grid Setup
    t0 = time.perf_counter()
    rules = ((zs * 10**14).astype(np.int64) % 8).reshape(H, W)
    key_grid = ((ws * 255).astype(np.uint8)).reshape(H, W)
    enc_times['Step 4: CA Rules & Keys Grid Setup'] = time.perf_counter() - t0
    
    # Step 5: Reversible Cellular Automata Iterations
    t0 = time.perf_counter()
    grid_y, grid_x = np.indices((H, W))
    red_mask = (grid_y + grid_x) % 2 == 0
    black_mask = ~red_mask
    state = scrambled.copy()
    for _ in range(2):
        g_red = ca.compute_ca_transition(state, key_grid, rules)
        state[red_mask] = (state[red_mask].astype(np.int16) + g_red[red_mask].astype(np.int16)) % 256
        g_black = ca.compute_ca_transition(state, key_grid, rules)
        state[black_mask] = (state[black_mask].astype(np.int16) + g_black[black_mask].astype(np.int16)) % 256
    enc_times['Step 5: Reversible Cellular Automata Iterations'] = time.perf_counter() - t0
    
    # Step 6: Forward Feedback Diffusion
    t0 = time.perf_counter()
    state_flat = state.flatten()
    K_flat = key_grid.flatten()
    C1 = np.zeros_like(state_flat)
    IV1 = int(key_bytes[0]) ^ 157
    prev = IV1
    for i in range(N):
        c = (int(state_flat[i]) + prev) % 256 ^ int(K_flat[i])
        C1[i] = c
        prev = c
    enc_times['Step 6: Forward Feedback Diffusion'] = time.perf_counter() - t0
    
    # Step 7: Backward Feedback Diffusion
    t0 = time.perf_counter()
    C2 = np.zeros_like(C1)
    IV2 = int(key_bytes[1]) ^ 223
    prev2 = IV2
    for i in range(N - 1, -1, -1):
        c2 = (int(C1[i]) + prev2) % 256 ^ int(K_flat[i])
        C2[i] = c2
        prev2 = c2
    cipher = C2.reshape(H, W).astype(np.uint8)
    enc_times['Step 7: Backward Feedback Diffusion'] = time.perf_counter() - t0
    
    # ---------------- DECRYPTION PROFILE ----------------
    
    # Step 1: Regenerate Chaotic Sequences
    t0 = time.perf_counter()
    x0_dec, y0_dec, z0_dec, w0_dec = x0, y0, z0, w0
    xs_dec, ys_dec, zs_dec, ws_dec = ca.generate_chaotic_sequences(H, W, x0_dec, y0_dec, z0_dec, w0_dec)
    dec_times['Step 1: Regenerate Chaotic Sequences'] = time.perf_counter() - t0
    
    # Step 2: CA Rules & Keys Grid Setup
    t0 = time.perf_counter()
    rules_dec = ((zs_dec * 10**14).astype(np.int64) % 8).reshape(H, W)
    key_grid_dec = ((ws_dec * 255).astype(np.uint8)).reshape(H, W)
    dec_times['Step 2: CA Rules & Keys Grid Setup'] = time.perf_counter() - t0
    
    # Step 3: Inverse Backward Feedback Diffusion
    t0 = time.perf_counter()
    C2_flat = cipher.flatten().astype(np.int32)
    K_flat_dec = key_grid_dec.flatten().astype(np.int32)
    IV2_dec = int(key_bytes[1]) ^ 223
    prev_C2 = np.concatenate((C2_flat[1:], [IV2_dec]))
    C1_dec = ((C2_flat ^ K_flat_dec) - prev_C2) % 256
    dec_times['Step 3: Inverse Backward Feedback Diffusion'] = time.perf_counter() - t0
    
    # Step 4: Inverse Forward Feedback Diffusion
    t0 = time.perf_counter()
    IV1_dec = int(key_bytes[0]) ^ 157
    prev_C1 = np.concatenate(([IV1_dec], C1_dec[:-1]))
    S_flat = ((C1_dec ^ K_flat_dec) - prev_C1) % 256
    state_dec = S_flat.reshape(H, W).astype(np.uint8)
    dec_times['Step 4: Inverse Forward Feedback Diffusion'] = time.perf_counter() - t0
    
    # Step 5: Inverse Reversible CA Iterations
    t0 = time.perf_counter()
    for _ in range(2):
        g_black = ca.compute_ca_transition(state_dec, key_grid_dec, rules_dec)
        state_dec[black_mask] = (state_dec[black_mask].astype(np.int16) - g_black[black_mask].astype(np.int16)) % 256
        g_red = ca.compute_ca_transition(state_dec, key_grid_dec, rules_dec)
        state_dec[red_mask] = (state_dec[red_mask].astype(np.int16) - g_red[red_mask].astype(np.int16)) % 256
    dec_times['Step 5: Inverse Reversible CA Iterations'] = time.perf_counter() - t0
    
    # Step 6: Inverse Scrambling & Crop
    t0 = time.perf_counter()
    row_idx_dec = np.argsort(xs_dec[:H])
    col_idx_dec = np.argsort(ys_dec[:W])
    inv_row_idx = np.zeros_like(row_idx_dec)
    inv_row_idx[row_idx_dec] = np.arange(H)
    inv_col_idx = np.zeros_like(col_idx_dec)
    inv_col_idx[col_idx_dec] = np.arange(W)
    decrypted_img = state_dec[:, inv_col_idx][inv_row_idx, :]
    decrypted_img = decrypted_img[:H_orig, :W_orig]
    dec_times['Step 6: Inverse Scrambling & Crop'] = time.perf_counter() - t0
    
    print("\n" + "="*70)
    print("      4D SINE-COSINE CA CRYPTOSYSTEM DETAILED PROFILE")
    print("="*70)
    print(f"Target Image : {image_path} ({H_orig}x{W_orig})")
    print("-"*70)
    
    # Find slowest stages
    max_enc_step = max(enc_times, key=enc_times.get)
    max_dec_step = max(dec_times, key=dec_times.get)
    
    # Print Encryption stages
    print(f"\033[1;32m [ENCRYPTION STAGES]\033[0m")
    for step, duration in enc_times.items():
        if step == max_enc_step:
            print(f"{RED}{step:<45} : {duration:.4f}s{RESET}")
        else:
            print(f"{step:<45} : {duration:.4f}s")
            
    print("-"*70)
    
    # Print Decryption stages
    print(f"\033[1;32m [DECRYPTION STAGES]\033[0m")
    for step, duration in dec_times.items():
        if step == max_dec_step:
            print(f"{RED}{step:<45} : {duration:.4f}s{RESET}")
        else:
            print(f"{step:<45} : {duration:.4f}s")
            
    print("-"*70)
    
    total_enc_time = sum(enc_times.values())
    total_dec_time = sum(dec_times.values())
    combined_total_time = total_enc_time + total_dec_time
    
    print(f"{GREEN}{'Total Encryption Time':<45} : {total_enc_time:.4f}s{RESET}")
    print(f"{GREEN}{'Total Decryption Time':<45} : {total_dec_time:.4f}s{RESET}")
    print(f"{GREEN}{'Total Execution Time (Enc + Dec)':<45} : {combined_total_time:.4f}s{RESET}")
    print("="*70 + "\n")
    
    return combined_total_time, f"{H_orig} X {W_orig}"

def main():
    key = b"IEEE_CA_Secure_Key_256_Bits_2026"
    
    # If a specific image is passed as command line argument
    if len(sys.argv) > 1:
        target_img = sys.argv[1]
        if not os.path.exists(target_img):
            print(f"Error: {target_img} not found. Please verify the file path.")
            sys.exit(1)
        profile_encryption_decryption(target_img, key)
        return  # Stop execution here, don't run the batch sweep!
        
    # Determine the default image to profile step-by-step
    default_img = "Images/pepper.tiff"
    if not os.path.exists(default_img):
        # Fallback to lenna.jpg or generate synthetic image
        if os.path.exists("Images/lenna.jpg"):
            default_img = "Images/lenna.jpg"
        else:
            default_img = "test_image.png"
            if not os.path.exists(default_img):
                img = generate_synthetic_image()
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(default_img, img)

    # Perform step-by-step detailed profiling for default image
    profile_encryption_decryption(default_img, key)
    
    # Sweep through all batch test images in the list
    print("=================================================================================")
    print("  BATCH PERFORMANCE PROFILE")
    print("=================================================================================")
    print(f"{'Image':<20} {'Size':<25} {'Time':<12}")
    print("---------------------------------------------------------------------------------")
    
    total_batch_time = 0.0
    processed_count = 0
    
    for img_name in TEST_IMAGES:
        img_path = os.path.join("Images", img_name)
        if not os.path.exists(img_path):
            continue
            
        t_start = time.perf_counter()
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Run core encryption and decryption
        cipher = ca.encrypt_image(img, key)
        decrypted = ca.decrypt_image(cipher, key, original_shape=img.shape)
        t_duration = time.perf_counter() - t_start
        
        size_str = f"{img.shape[0]} X {img.shape[1]}"
        print(f"{img_name:<20} {size_str:<15} {t_duration:.3f}s")
        
        total_batch_time += t_duration
        processed_count += 1
        
    print("==========================================================================================")
    
    if processed_count > 0:
        avg_time = total_batch_time / processed_count
        # Yellow and Bold color average line
        print(f"\033[1;33mAverage time is = {avg_time:.3f}s\033[0m\n")
    else:
        print("No batch images were found for performance profiling.\n")

if __name__ == "__main__":
    main()
