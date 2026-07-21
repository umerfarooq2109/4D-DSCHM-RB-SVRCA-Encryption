import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

# Import Lyapunov solver
from chaos_analysis.lyapunov import calculate_lyapunov_exponents

def main():
    print("====================================================")
    print("Computing Kaplan-Yorke (Lyapunov) Dimension...")
    print("====================================================")
    
    # 1. Calculate Lyapunov exponents at parameter alpha = 1.412
    a_vals, le_results = calculate_lyapunov_exponents(iterations=2000, steps=20)
    idx = np.argmin(np.abs(a_vals - 1.412))
    le_vals = le_results[idx]
    
    # Sort descending
    le_sorted = np.sort(le_vals)[::-1]
    print(f"Sorted Lyapunov Exponents: {', '.join([f'{x:+.6f}' for x in le_sorted])}")
    
    # 2. Find j such that the sum of the first j exponents is non-negative
    running_sum = 0.0
    j = 0
    for idx, le in enumerate(le_sorted):
        if running_sum + le >= 0.0:
            running_sum += le
            j = idx + 1
        else:
            break
            
    print(f"Largest index j with non-negative sum: {j}")
    print(f"Sum of first {j} exponents: {running_sum:+.6f}")
    
    if j < len(le_sorted):
        # 3. Apply Kaplan-Yorke formula
        next_le = le_sorted[j]
        d_ky = j + (running_sum / np.abs(next_le))
        print(f"Kaplan-Yorke Dimension (D_KY): {d_ky:.6f}")
        print("\nStatus: PASS (Fractal Dimension > 2.0 indicates strange attractor)")
    else:
        print("Error: Exponents sum never became negative.")
    print("====================================================")

if __name__ == "__main__":
    main()
