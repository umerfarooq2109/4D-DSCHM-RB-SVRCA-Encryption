import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.cellular_automata as ca

def compute_npcr(c1, c2):
    diff = (c1 != c2)
    return (np.sum(diff) / diff.size) * 100.0

def compute_uaci(c1, c2):
    diff = np.abs(c1.astype(float) - c2.astype(float))
    return (np.sum(diff) / (255.0 * c1.size)) * 100.0

def run_differential_general(image_path):
    if not os.path.exists(image_path):
        return None
        
    pil_img = Image.open(image_path).convert('L')
    gray_img = np.array(pil_img)
    M, N = gray_img.shape
    
    key = b"IEEE_CA_Secure_Key_256_Bits_2026"
    
    # Original encryption
    c1 = ca.encrypt_image(gray_img, key)
    
    # Mutate center pixel
    r, c = M // 2, N // 2
    orig_pixel = gray_img[r, c]
    changed_pixel = orig_pixel + 1 if orig_pixel < 255 else orig_pixel - 1
    
    gray_img_mod = gray_img.copy()
    gray_img_mod[r, c] = changed_pixel
    
    # Mutated encryption
    c2 = ca.encrypt_image(gray_img_mod, key)
    
    # Calculate NPCR and UACI
    npcr_val = compute_npcr(c1, c2)
    uaci_val = compute_uaci(c1, c2)
    return npcr_val, uaci_val

def main():
    test_files = [
        "5.1.09.tiff", "5.1.10.tiff", "5.1.11.tiff", "5.1.12.tiff", "5.1.13.tiff", "5.1.14.tiff",
        "5.2.08.tiff", "5.2.09.tiff", "5.2.10.tiff",
        "7.1.01.tiff", "7.1.02.tiff", "7.1.03.tiff", "7.1.04.tiff", "7.1.05.tiff", "7.1.06.tiff", "7.1.07.tiff", "7.1.08.tiff", "7.1.09.tiff", "7.1.10.tiff",
        "boat.512.tiff", "gray21.512.tiff", "ruler.512.tiff",
        "5.3.01.tiff", "5.3.02.tiff",
        "7.2.01.tiff"
    ]
    
    img_dir = "Images"
    results = []
    
    print("Calculating general NPCR and UACI results across benchmark images...")
    
    for filename in test_files:
        path = os.path.join(img_dir, filename)
        metrics = run_differential_general(path)
        if metrics is not None:
            npcr_val, uaci_val = metrics
            results.append({
                'name': filename.replace('.tiff', ''),
                'npcr': npcr_val,
                'uaci': uaci_val
            })
            print(f"  {filename:<15} -> NPCR: {npcr_val:.4f}%, UACI: {uaci_val:.4f}%")
            
    print("\nCalculations finished! Here are the summary values:")
    for r in results:
        print(f"'{r['name']}': {{'npcr': {r['npcr']:.6f}, 'uaci': {r['uaci']:.6f}}},")

if __name__ == "__main__":
    main()
