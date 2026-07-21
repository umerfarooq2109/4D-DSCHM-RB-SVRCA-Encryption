import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import src.cellular_automata as ca
from src.utils import calculate_mse, calculate_ssim

def generate_pseudo_random_image(H, W, seed):
    """
    Generates a deterministic pseudo-random image of size HxW using a vectorized
    sine wave sequence, avoiding numpy.random dll load errors.
    """
    indices = np.arange(H * W, dtype=np.float64)
    # Sine multipliers create high frequency pseudo-random numbers
    vals = np.sin(indices * 0.123456789 + seed) * 10000.0
    img = (np.abs(vals) % 256.0).astype(np.uint8).reshape(H, W)
    return img

def run_lossless_test():
    """
    Tests the RB-SVRCA pipeline on 5 random noise images of size 256x256
    to verify MSE = 0.0 and SSIM = 1.0 (lossless recovery).
    """
    print("Running automated lossless pipeline test...")
    key = b"Test_Key_For_Automated_Pipeline_01"
    
    for i in range(5):
        # Generate pseudo-random image using deterministic function
        img = generate_pseudo_random_image(256, 256, seed=1234.56 * (i + 1))
        
        # Run Encryption and Decryption
        cipher = ca.encrypt_image(img, key)
        decrypted = ca.decrypt_image(cipher, key, original_shape=img.shape)
        
        # Calculate verification metrics
        mse = calculate_mse(img, decrypted)
        ssim = calculate_ssim(img, decrypted)
        
        print(f"  Test Case #{i+1}: MSE = {mse:.6f}, SSIM = {ssim:.6f}")
        assert mse == 0.0, f"Test Case #{i+1} failed: MSE is not 0 (Got: {mse})"
        assert ssim == 1.0, f"Test Case #{i+1} failed: SSIM is not 1.0 (Got: {ssim})"
        
    print("All functional lossless test cases PASSED successfully.")
    return True

if __name__ == "__main__":
    run_lossless_test()
