import numpy as np
import hashlib

# System constants for 4D-DSCHM (guaranteeing hyperchaotic behavior)
A, B, C, D, E, F, G, H_param = 1.412, 0.354, 1.287, 0.219, 1.543, 0.412, 1.109, 0.331

def key_to_initial_states(key_bytes):
    """
    Hashes the 256-bit key using SHA-256 to extract four initial states in [0, 1).
    """
    h = hashlib.sha256(key_bytes).digest()
    chunks = [h[i:i+8] for i in range(0, 32, 8)]
    states = []
    for chunk in chunks:
        val = int.from_bytes(chunk, byteorder='big')
        # Map to range [0.1, 0.9] to avoid edge cases near exactly 0 or 1
        states.append(0.1 + (val % 10**16) / 10**16 * 0.8)
    return states

def generate_chaotic_sequences(H, W, x0, y0, z0, w0):
    """
    Generates four pseudo-random sequences of length H * W using the 4D-DSCHM equations.
    """
    N = H * W
    xs = np.zeros(N)
    ys = np.zeros(N)
    zs = np.zeros(N)
    ws = np.zeros(N)
    
    x, y, z, w = x0, y0, z0, w0
    
    # Warm-up phase of 1000 steps to remove initial transients
    for _ in range(1000):
        x = (np.sin(A * y) + B * np.cos(x) + w) % 1.0
        y = (np.sin(C * x) + D * np.cos(y)) % 1.0
        z = (np.sin(E * z) + F * np.cos(w)) % 1.0
        w = (np.sin(G * w) + H_param * np.cos(z)) % 1.0
        
    for i in range(N):
        x = (np.sin(A * y) + B * np.cos(x) + w) % 1.0
        y = (np.sin(C * x) + D * np.cos(y)) % 1.0
        z = (np.sin(E * z) + F * np.cos(w)) % 1.0
        w = (np.sin(G * w) + H_param * np.cos(z)) % 1.0
        xs[i] = x
        ys[i] = y
        zs[i] = z
        ws[i] = w
        
    return xs, ys, zs, ws
