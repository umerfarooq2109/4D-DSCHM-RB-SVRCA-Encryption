import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

def calculate_approximate_entropy(u, m=2, r=None):
    """Calculates the Approximate Entropy (ApEn) of a time series u."""
    u = np.array(u)
    N = len(u)
    if r is None:
        r = 0.2 * np.std(u)
        
    def _phi(m_val):
        N_m = N - m_val + 1
        # Create vectors of length m_val: shape (N_m, m_val)
        x = np.array([u[i : i + m_val] for i in range(N_m)])
        
        # Calculate C_i(r):
        # We broadcast subtract to find max absolute difference between all pairs
        # x[:, None, :] shape: (N_m, 1, m_val)
        # x[None, :, :] shape: (1, N_m, m_val)
        # diff shape: (N_m, N_m, m_val)
        diff = np.abs(x[:, None, :] - x[None, :, :])
        max_diff = np.max(diff, axis=-1)  # shape (N_m, N_m)
        
        # Count elements <= r
        counts = np.sum(max_diff <= r, axis=-1) / N_m
        return np.mean(np.log(counts + 1e-18))
        
    return _phi(m) - _phi(m + 1)

def main():
    print("====================================================")
    print("Calculating Approximate Entropy (ApEn) Analysis...")
    print("====================================================")
    
    # 4D-DSCHM Parameters
    a = 1.412
    b, c, d, e, f, g, h = 0.354, 1.287, 0.219, 1.543, 0.412, 1.109, 0.331
    
    N = 1000  # N=1000 is ideal for ApEn speed and accuracy
    x, y, z, w = 0.1, 0.2, 0.3, 0.4
    
    # Warm-up (transient discard)
    for _ in range(1000):
        x = (np.sin(a * y) + b * np.cos(x) + w) % 1.0
        y = (np.sin(c * x) + d * np.cos(y)) % 1.0
        z = (np.sin(e * z) + f * np.cos(w)) % 1.0
        w = (np.sin(g * w) + h * np.cos(z)) % 1.0
        
    x_pts = []
    for _ in range(N):
        x = (np.sin(a * y) + b * np.cos(x) + w) % 1.0
        y = (np.sin(c * x) + d * np.cos(y)) % 1.0
        z = (np.sin(e * z) + f * np.cos(w)) % 1.0
        w = (np.sin(g * w) + h * np.cos(z)) % 1.0
        x_pts.append(x)
        
    x_pts = np.array(x_pts)
    
    # Compute ApEn for m=2
    apen = calculate_approximate_entropy(x_pts, m=2)
    
    print(f"Approximate Entropy (ApEn, m=2, r=0.2*std): {apen:.6f}")
    
    # Typical thresholds: ApEn > 0.5 indicates high unpredictability for chaotic maps
    if apen > 0.5:
        print("Status: PASS (High entropy confirms complex unpredictability)")
    else:
        print("Status: WARNING (Sub-optimal entropy detected)")
    print("====================================================")

if __name__ == "__main__":
    main()
