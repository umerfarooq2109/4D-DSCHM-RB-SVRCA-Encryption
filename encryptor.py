import numpy as np
import hashlib

def key_to_initial_states(key_bytes):
    """
    Hashes the 256-bit key to extract four high-entropy initial states in [0, 1) and parameters.
    """
    h = hashlib.sha256(key_bytes).digest()
    chunks = [h[i:i+8] for i in range(0, 32, 8)]
    states = []
    for chunk in chunks:
        val = int.from_bytes(chunk, byteorder='big')
        states.append((val % 10**16) / 10**16)
    return states

def generate_chaotic_sequences(H, W, x0, y0, z0, w0):
    """
    Generates four pseudo-random sequences of length H * W using the 4D-DSCHM.
    """
    N = H * W
    xs = np.zeros(N)
    ys = np.zeros(N)
    zs = np.zeros(N)
    ws = np.zeros(N)
    
    # Chaos control parameters
    a, b, c, d, e, f, g, h = 1.412, 0.354, 1.287, 0.219, 1.543, 0.412, 1.109, 0.331
    
    x, y, z, w = x0, y0, z0, w0
    
    # Warm-up to eliminate transients
    for _ in range(1000):
        x = (np.sin(a * y) + b * np.cos(x) + w) % 1.0
        y = (np.sin(c * x) + d * np.cos(y)) % 1.0
        z = (np.sin(e * z) + f * np.cos(w)) % 1.0
        w = (np.sin(g * w) + h * np.cos(z)) % 1.0
        
    for i in range(N):
        x = (np.sin(a * y) + b * np.cos(x) + w) % 1.0
        y = (np.sin(c * x) + d * np.cos(y)) % 1.0
        z = (np.sin(e * z) + f * np.cos(w)) % 1.0
        w = (np.sin(g * w) + h * np.cos(z)) % 1.0
        xs[i] = x
        ys[i] = y
        zs[i] = z
        ws[i] = w
        
    return xs, ys, zs, ws

def compute_ca_transition(img, key, rules):
    """
    Vectorized computation of the 8 space-varying Cellular Automata rules.
    """
    # Cast to int32 to prevent numpy overflow during addition/subtraction/modulo
    left = np.roll(img, 1, axis=1).astype(np.int32)
    right = np.roll(img, -1, axis=1).astype(np.int32)
    top = np.roll(img, 1, axis=0).astype(np.int32)
    bottom = np.roll(img, -1, axis=0).astype(np.int32)
    self_state = img.astype(np.int32)
    key_int = key.astype(np.int32)
    
    # 8 distinct space-varying CA rules (modulo 256)
    out0 = ((left + right + top + bottom + self_state) ^ key_int) % 256
    out1 = ((left ^ right ^ top ^ bottom) + key_int) % 256
    out2 = (((left + right) ^ (top + bottom)) + key_int) % 256
    out3 = ((left + right - top - bottom) ^ key_int) % 256
    out4 = ((left * 3 + right * 7 + top * 5 + bottom * 2) ^ key_int) % 256
    out5 = ((left ^ self_state ^ right) + key_int) % 256
    out6 = ((top ^ self_state ^ bottom) + key_int) % 256
    out7 = (((left + top) ^ (right + bottom)) ^ key_int) % 256
    
    # Select transition rule dynamically per cell
    g = np.zeros_like(img, dtype=np.uint8)
    g = np.where(rules == 0, out0,
        np.where(rules == 1, out1,
        np.where(rules == 2, out2,
        np.where(rules == 3, out3,
        np.where(rules == 4, out4,
        np.where(rules == 5, out5,
        np.where(rules == 6, out6, out7)))))))
    return g.astype(np.uint8)


def encrypt_image(img, key_bytes):
    """
    Encrypts a grayscale image using position permutation and RB-SVRCA.
    """
    H, W = img.shape
    
    # 1. Generate chaotic sequences
    x0, y0, z0, w0 = key_to_initial_states(key_bytes)
    xs, ys, zs, ws = generate_chaotic_sequences(H, W, x0, y0, z0, w0)
    
    # 2. Row and Column Scrambling
    row_idx = np.argsort(xs[:H])
    col_idx = np.argsort(ys[:W])
    scrambled = img[row_idx, :][:, col_idx]
    
    # 3. Cellular Automata grid setup
    rules = ((zs * 10**14).astype(np.int64) % 8).reshape(H, W)
    key_grid = ((ws * 255).astype(np.uint8)).reshape(H, W)
    
    # 4. Red-Black Reversible CA iterations
    # Create Red/Black Checkerboard Masks
    grid_y, grid_x = np.indices((H, W))
    red_mask = (grid_y + grid_x) % 2 == 0
    black_mask = ~red_mask
    
    # Iterate Cellular Automata (2 rounds of Red-Black update for diffusion)
    state = scrambled.copy()
    for _ in range(2):
        # Update Red cells (reading from Black neighbors)
        g_red = compute_ca_transition(state, key_grid, rules)
        state[red_mask] = (state[red_mask] + g_red[red_mask]) % 256
        
        # Update Black cells (reading from updated Red neighbors)
        g_black = compute_ca_transition(state, key_grid, rules)
        state[black_mask] = (state[black_mask] + g_black[black_mask]) % 256
        
    return state

def decrypt_image(cipher_img, key_bytes):
    """
    Decrypts the cipher image using inverse RB-SVRCA and reverse permutation.
    """
    H, W = cipher_img.shape
    
    # 1. Regenerate chaotic sequences
    x0, y0, z0, w0 = key_to_initial_states(key_bytes)
    xs, ys, zs, ws = generate_chaotic_sequences(H, W, x0, y0, z0, w0)
    
    # 2. CA Grid Setup
    rules = ((zs * 10**14).astype(np.int64) % 8).reshape(H, W)
    key_grid = ((ws * 255).astype(np.uint8)).reshape(H, W)
    
    grid_y, grid_x = np.indices((H, W))
    red_mask = (grid_y + grid_x) % 2 == 0
    black_mask = ~red_mask
    
    # 3. Inverse Red-Black Reversible CA iterations (run backwards)
    state = cipher_img.copy()
    for _ in range(2):
        # Reverse Black cells first
        g_black = compute_ca_transition(state, key_grid, rules)
        state[black_mask] = (state[black_mask] - g_black[black_mask]) % 256
        
        # Reverse Red cells second
        g_red = compute_ca_transition(state, key_grid, rules)
        state[red_mask] = (state[red_mask] - g_red[red_mask]) % 256
        
    # 4. Reverse Permutation
    row_idx = np.argsort(xs[:H])
    col_idx = np.argsort(ys[:W])
    
    inv_row_idx = np.zeros_like(row_idx)
    inv_row_idx[row_idx] = np.arange(H)
    inv_col_idx = np.zeros_like(col_idx)
    inv_col_idx[col_idx] = np.arange(W)
    
    decrypted_img = state[:, inv_col_idx][inv_row_idx, :]
    return decrypted_img
