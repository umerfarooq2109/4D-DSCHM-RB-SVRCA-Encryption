import os
import sys
import time
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.cellular_automata as ca

key = b"IEEE_CA_Secure_Key_256_Bits_2026"
sizes = [64, 128, 256, 512, 1024]

print("=========================================")
speed_test_header = "    Proposed Cryptosystem Grayscale Speed Test"
print(speed_test_header)
print("=========================================")
for s in sizes:
    # Create random 8-bit gray level image of size s x s
    img = np.random.randint(0, 256, (s, s), dtype=np.uint8)
    
    # Measure encryption time
    t0 = time.perf_counter()
    _ = ca.encrypt_image(img, key)
    t_enc = time.perf_counter() - t0
    
    print(f" {s:<4} x {s:<4}   ->   {t_enc:.4f} seconds")
print("=========================================")
