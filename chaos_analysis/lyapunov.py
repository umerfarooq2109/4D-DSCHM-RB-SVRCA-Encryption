import numpy as np
import matplotlib.pyplot as plt

def get_jacobian(x, y, z, w, a, b, c, d, e, f, g, h_param):
    """
    Computes the 4x4 Jacobian matrix for the 4D-DSCHM.
    """
    J = np.zeros((4, 4))
    
    # Row 1 partial derivatives
    J[0, 0] = -b * np.sin(x)
    J[0, 1] = a * np.cos(a * y)
    J[0, 2] = 0.0
    J[0, 3] = 1.0
    
    # Row 2 partial derivatives
    J[1, 0] = c * np.cos(c * x)
    J[1, 1] = -d * np.sin(y)
    J[1, 2] = 0.0
    J[1, 3] = 0.0
    
    # Row 3 partial derivatives
    J[2, 0] = 0.0
    J[2, 1] = 0.0
    J[2, 2] = e * np.cos(e * z)
    J[2, 3] = -f * np.sin(w)
    
    # Row 4 partial derivatives
    J[3, 0] = 0.0
    J[3, 1] = 0.0
    J[3, 2] = -h_param * np.sin(z)
    J[3, 3] = g * np.cos(g * w)
    
    return J

def calculate_lyapunov_exponents(iterations=5000, start_a=0.5, end_a=3.0, steps=100):
    """
    Computes the four Lyapunov Exponents for varying values of parameter alpha (A).
    """
    a_values = np.linspace(start_a, end_a, steps)
    
    # System parameters
    b, c, d, e, f, g, h_param = 0.354, 1.287, 0.219, 1.543, 0.412, 1.109, 0.331
    
    le_results = np.zeros((steps, 4))
    
    for idx, a in enumerate(a_values):
        # Initial states
        x, y, z, w = 0.1, 0.2, 0.3, 0.4
        
        # Identity matrix for orthogonalization
        Q = np.eye(4)
        sums = np.zeros(4)
        
        # Warm-up phase
        for _ in range(500):
            x = (np.sin(a * y) + b * np.cos(x) + w) % 1.0
            y = (np.sin(c * x) + d * np.cos(y)) % 1.0
            z = (np.sin(e * z) + f * np.cos(w)) % 1.0
            w = (np.sin(g * w) + h_param * np.cos(z)) % 1.0
            
        # Calculation phase
        for _ in range(iterations):
            # 1. Iterate Map
            x = (np.sin(a * y) + b * np.cos(x) + w) % 1.0
            y = (np.sin(c * x) + d * np.cos(y)) % 1.0
            z = (np.sin(e * z) + f * np.cos(w)) % 1.0
            w = (np.sin(g * w) + h_param * np.cos(z)) % 1.0
            
            # 2. Get Jacobian
            J = get_jacobian(x, y, z, w, a, b, c, d, e, f, g, h_param)
            
            # 3. QR Orthogonalization to prevent exponent explosion
            M = np.dot(J, Q)
            Q, R = np.linalg.qr(M)
            
            # 4. Accumulate eigenvalues log
            diag_val = np.abs(np.diag(R))
            # Safe log calculation to avoid divide by zero warnings
            diag_val[diag_val < 1e-12] = 1e-12
            sums += np.log(diag_val)
            
        le_results[idx, :] = sums / iterations
        
    return a_values, le_results

def plot_lyapunov_exponents(a_values, le_results, output_path="results/plots/lyapunov_exponents.png"):
    """
    Plots and saves the Lyapunov exponents curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(a_values, le_results[:, 0], label=r"$\lambda_1$ (LE1)", color='red')
    plt.plot(a_values, le_results[:, 1], label=r"$\lambda_2$ (LE2)", color='blue')
    plt.plot(a_values, le_results[:, 2], label=r"$\lambda_3$ (LE3)", color='green')
    plt.plot(a_values, le_results[:, 3], label=r"$\lambda_4$ (LE4)", color='purple')
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel(r"Control Parameter $\alpha$")
    plt.ylabel("Lyapunov Exponents")
    plt.title("Lyapunov Exponents Spectrum of 4D-DSCHM")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    print("====================================================")
    print("Computing Lyapunov Exponents for 4D-DSCHM...")
    print("====================================================")
    
    os.makedirs("results/plots", exist_ok=True)
    
    # Compute and plot
    a_vals, le_results = calculate_lyapunov_exponents(iterations=1500, steps=80)
    plot_lyapunov_exponents(a_vals, le_results)
    print("LE analysis complete. Plot saved to results/plots/lyapunov_exponents.png")
    
    # Standard alpha parameter check
    idx = np.argmin(np.abs(a_vals - 1.412))
    le_vals = le_results[idx]
    
    def color_val(val):
        if val > 0:
            return f"\033[92m{val:+.6f}\033[0m"  # Green
        else:
            return f"\033[91m{val:+.6f}\033[0m"  # Red

    print(f"\nLyapunov Exponents at alpha = 1.412:")
    print(f"  LE1: {color_val(le_vals[0])}")
    print(f"  LE2: {color_val(le_vals[1])}")
    print(f"  LE3: {color_val(le_vals[2])}")
    print(f"  LE4: {color_val(le_vals[3])}")
    
    pos_count = np.sum(le_vals > 0.0)
    print(f"\nNumber of Positive Lyapunov Exponents: {pos_count}")
    if pos_count >= 2:
        print("Status: \033[92mHyperchaos Confirmed (>= 2 positive exponents)\033[0m")
    else:
        print("Status: \033[91mNo Hyperchaos\033[0m")
    print("====================================================")
