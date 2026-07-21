import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

def compute_zero_one_k(x_series, c):
    """Computes the K parameter of the 0-1 test for a given constant c."""
    N = len(x_series)
    j_arr = np.arange(1, N + 1)
    
    # Phase coordinates
    p = np.cumsum(x_series * np.cos(j_arr * c))
    q = np.cumsum(x_series * np.sin(j_arr * c))
    
    # Define MSD calculation limit
    N_cut = N // 10
    n_arr = np.arange(1, N_cut + 1)
    M = np.zeros(N_cut)
    
    # Vectorized mean square displacement (MSD)
    # D(n) = (1/N) * sum_{j=1}^{N-n} [ (p(j+n)-p(j))^2 + (q(j+n)-q(j))^2 ]
    for n in range(1, N_cut + 1):
        diff_p = p[n:] - p[:-n]
        diff_q = q[n:] - q[:-n]
        d_n = np.mean(diff_p**2 + diff_q**2)
        # Corrected MSD formulation to remove drift
        cov_factor = (np.mean(x_series)**2) * (1.0 - np.cos(n * c)) / (1.0 - np.cos(c))
        M[n - 1] = d_n - cov_factor
        
    # Calculate Pearson correlation coefficient K = corr(n, M)
    mean_n = np.mean(n_arr)
    mean_M = np.mean(M)
    num = np.sum((n_arr - mean_n) * (M - mean_M))
    den = np.sqrt(np.sum((n_arr - mean_n)**2) * np.sum((M - mean_M)**2))
    
    return 0.0 if den == 0 else num / den

def main():
    print("====================================================")
    print("Running Gottwald-Melbourne 0-1 Test for Chaos...")
    print("====================================================")
    
    # 4D-DSCHM Parameters
    a = 1.412
    b, c_param, d, e, f, g, h = 0.354, 1.287, 0.219, 1.543, 0.412, 1.109, 0.331
    
    N = 3000
    x, y, z, w = 0.1, 0.2, 0.3, 0.4
    
    # Warm-up (transient discard)
    for _ in range(1000):
        x = (np.sin(a * y) + b * np.cos(x) + w) % 1.0
        y = (np.sin(c_param * x) + d * np.cos(y)) % 1.0
        z = (np.sin(e * z) + f * np.cos(w)) % 1.0
        w = (np.sin(g * w) + h * np.cos(z)) % 1.0
        
    x_series = []
    for _ in range(N):
        x = (np.sin(a * y) + b * np.cos(x) + w) % 1.0
        y = (np.sin(c_param * x) + d * np.cos(y)) % 1.0
        z = (np.sin(e * z) + f * np.cos(w)) % 1.0
        w = (np.sin(g * w) + h * np.cos(z)) % 1.0
        x_series.append(x)
        
    x_series = np.array(x_series)
    
    # Evaluate over multiple random c-values to prevent resonances
    # Choose 10 random values of c in (pi/5, 4*pi/5)
    np.random.seed(42)  # Deterministic seed for testing consistency
    c_vals = np.linspace(np.pi / 5.0 + 0.1, 4.0 * np.pi / 5.0 - 0.1, 10)
    
    k_results = []
    for c in c_vals:
        k = compute_zero_one_k(x_series, c)
        k_results.append(k)
        
    median_k = np.median(k_results)
    print(f"Computed K values for 10 distinct frequencies:")
    for idx, k in enumerate(k_results):
        print(f"  c{idx+1}: {c_vals[idx]:.4f} -> K = {k:.6f}")
        
    print(f"\nMedian Chaos parameter K: {median_k:.6f}")
    if median_k > 0.90:
        print("Status: PASS (K close to 1.0 confirms deterministic chaos)")
    else:
        print("Status: FAIL (K close to 0.0 indicates periodic/non-chaotic behavior)")
    print("====================================================")

if __name__ == "__main__":
    main()
