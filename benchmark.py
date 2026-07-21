import cv2
import numpy as np
import time
import encryptor

def calculate_entropy(img):
    """Calculates the Shannon information entropy of the image."""
    pixel_counts = np.bincount(img.flatten(), minlength=256)
    probabilities = pixel_counts / len(img.flatten())
    # Filter out zero probabilities for log2
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_correlation(img, direction='horizontal', num_pairs=10000):
    """Calculates Pearson correlation between 10,000 adjacent pixel pairs."""
    H, W = img.shape
    np.random.seed(42)  # Set seed for consistency
    
    if direction == 'horizontal':
        x_coords = np.random.randint(0, H, num_pairs)
        y_coords = np.random.randint(0, W - 1, num_pairs)
        x = img[x_coords, y_coords].astype(np.float64)
        y = img[x_coords, y_coords + 1].astype(np.float64)
    elif direction == 'vertical':
        x_coords = np.random.randint(0, H - 1, num_pairs)
        y_coords = np.random.randint(0, W, num_pairs)
        x = img[x_coords, y_coords].astype(np.float64)
        y = img[x_coords + 1, y_coords].astype(np.float64)
    elif direction == 'diagonal':
        x_coords = np.random.randint(0, H - 1, num_pairs)
        y_coords = np.random.randint(0, W - 1, num_pairs)
        x = img[x_coords, y_coords].astype(np.float64)
        y = img[x_coords + 1, y_coords + 1].astype(np.float64)
        
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    
    if den == 0:
        return 0.0
    return num / den

def calculate_npcr_uaci(img, key):
    """Calculates NPCR and UACI by modifying exactly 1 pixel in the plaintext."""
    H, W = img.shape
    
    # Modify 1 pixel at center
    img_mod = img.copy()
    cx, cy = H // 2, W // 2
    img_mod[cx, cy] = (img[cx, cy] + 1) % 256
    
    # Encrypt both
    c1 = encryptor.encrypt_image(img, key)
    c2 = encryptor.encrypt_image(img_mod, key)
    
    # Compute NPCR
    diff = (c1 != c2).astype(np.float64)
    npcr = np.mean(diff) * 100.0
    
    # Compute UACI
    abs_diff = np.abs(c1.astype(np.float64) - c2.astype(np.float64))
    uaci = np.mean(abs_diff / 255.0) * 100.0
    
    return npcr, uaci

def main():
    print("====================================================")
    print("RB-SVRCA Cryptosystem IEEE Benchmark Suite")
    print("====================================================")
    
    key = b"IEEE_CA_Secure_Key_256_Bits_2026"
    
    # Load test image
    img = cv2.imread("test_image.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: test_image.png not found. Run main.py first.")
        return
        
    cipher = cv2.imread("ciphertext.png", cv2.IMREAD_GRAYSCALE)
    if cipher is None:
        print("Error: ciphertext.png not found. Run main.py first.")
        return
        
    print(f"Benchmark Image Size: {img.shape[0]}x{img.shape[1]}")
    
    # 1. Entropy Analysis
    entropy_plain = calculate_entropy(img)
    entropy_cipher = calculate_entropy(cipher)
    print("\n1. Information Entropy:")
    print(f"   Plaintext Entropy:  {entropy_plain:.6f}")
    print(f"   Ciphertext Entropy: {entropy_cipher:.6f} (Ideal: 8.000000)")
    
    # 2. Correlation Coefficients
    print("\n2. Correlation Coefficients (Adjacent Pixels):")
    for direction in ['horizontal', 'vertical', 'diagonal']:
        r_plain = calculate_correlation(img, direction)
        r_cipher = calculate_correlation(cipher, direction)
        print(f"   {direction.capitalize()}:")
        print(f"     Plaintext:  {r_plain:+.6f}")
        print(f"     Ciphertext: {r_cipher:+.6f} (Ideal: ~0.000000)")
        
    # 3. Differential Sensitivity (NPCR and UACI)
    print("\n3. Differential Attack Resistance:")
    print("   Encrypting pixel-modified image pair...")
    npcr, uaci = calculate_npcr_uaci(img, key)
    print(f"   NPCR: {npcr:.6f}% (Ideal: >99.6094%)")
    print(f"   UACI: {uaci:.6f}% (Ideal: ~33.4635%)")
    
    # 4. Latency Analysis (10 runs)
    print("\n4. Performance Benchmark:")
    print("   Running encryption 10 times to average latency...")
    enc_times = []
    dec_times = []
    for _ in range(10):
        t0 = time.time()
        c = encryptor.encrypt_image(img, key)
        enc_times.append((time.time() - t0) * 1000)
        
        t0 = time.time()
        d = encryptor.decrypt_image(c, key)
        dec_times.append((time.time() - t0) * 1000)
        
    avg_enc = np.mean(enc_times)
    avg_dec = np.mean(dec_times)
    print(f"   Average Encryption Speed: {avg_enc:.2f} ms")
    print(f"   Average Decryption Speed: {avg_dec:.2f} ms")
    print("====================================================")

if __name__ == "__main__":
    main()
