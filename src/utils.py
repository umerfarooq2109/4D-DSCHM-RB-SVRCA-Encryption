import cv2
import numpy as np
import os

def create_directories():
    """Ensures results directory structure is present."""
    paths = [
        "results/encrypted",
        "results/decrypted",
        "results/plots"
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)

def generate_synthetic_image():
    """Generates a synthetic grayscale image with gradients and geometry for testing."""
    img = np.zeros((256, 256), dtype=np.uint8)
    
    # Diagonal gradient
    for i in range(256):
        for j in range(256):
            img[i, j] = (i + j) // 2
            
    # Draw shapes
    cv2.circle(img, (128, 128), 55, 255, -1)
    cv2.rectangle(img, (35, 35), (85, 85), 0, -1)
    cv2.putText(img, "IEEE CA", (45, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    
    return img

def calculate_mse(img1, img2):
    """Calculates Mean Square Error between two images."""
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return np.mean(diff ** 2)

def calculate_ssim(img1, img2):
    """
    Calculates Structural Similarity Index (SSIM) manually using Gaussian filters,
    eliminating any dependency on the scikit-image library.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Local means
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Local variances and covariances
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)
