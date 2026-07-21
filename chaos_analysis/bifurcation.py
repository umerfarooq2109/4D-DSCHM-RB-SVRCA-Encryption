import numpy as np
import matplotlib.pyplot as plt

def calculate_bifurcation(start_a=0.5, end_a=3.0, steps=400, transient=1000, keep=200):
    """
    Computes the bifurcation points for parameter alpha (A).
    """
    a_values = np.linspace(start_a, end_a, steps)
    
    # Constants
    b, c, d, e, f, g, h_param = 0.354, 1.287, 0.219, 1.543, 0.412, 1.109, 0.331
    
    a_out = []
    x_out = []
    
    for a in a_values:
        x, y, z, w = 0.1, 0.2, 0.3, 0.4
        
        # Discard transient phase
        for _ in range(transient):
            x = (np.sin(a * y) + B * np.cos(x) + w) % 1.0
            y = (np.sin(C * x) + D * np.cos(y)) % 1.0
            z = (np.sin(E * z) + F * np.cos(w)) % 1.0
            w = (np.sin(G * w) + H_param * np.cos(z)) % 1.0
            
        # Keep next points
        for _ in range(keep):
            x = (np.sin(a * y) + B * np.cos(x) + w) % 1.0
            y = (np.sin(C * x) + D * np.cos(y)) % 1.0
            z = (np.sin(E * z) + F * np.cos(w)) % 1.0
            w = (np.sin(G * w) + H_param * np.cos(z)) % 1.0
            
            a_out.append(a)
            x_out.append(x)
            
    return a_out, x_out

# Define B, C, D, E, F, G, H_param as global inside module or inside functions
# Let's declare them as module-level constants to make it clean:
B, C, D, E, F, G, H_param = 0.354, 1.287, 0.219, 1.543, 0.412, 1.109, 0.331

def plot_bifurcation(a_out, x_out, output_path="results/plots/bifurcation_diagram.png"):
    """
    Plots and saves the bifurcation diagram.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(a_out, x_out, s=0.1, color='navy', alpha=0.5)
    plt.xlabel(r"Control Parameter $\alpha$")
    plt.ylabel(r"State Variable $x$")
    plt.title("Bifurcation Diagram of the 4D-DSCHM")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(min(a_out), max(a_out))
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    print("====================================================")
    print("Computing Bifurcation Diagram for 4D-DSCHM...")
    print("====================================================")
    
    os.makedirs("results/plots", exist_ok=True)
    
    # Compute and plot
    a_out, x_out = calculate_bifurcation(steps=250, transient=500, keep=100)
    plot_bifurcation(a_out, x_out)
    print("Bifurcation analysis complete. Plot saved to results/plots/bifurcation_diagram.png")
    print("====================================================")
