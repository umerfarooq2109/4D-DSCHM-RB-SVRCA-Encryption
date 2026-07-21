import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("====================================================")
    print("Generating Strange Attractor Phase Portraits...")
    print("====================================================")
    
    # 4D-DSCHM Parameters
    a = 1.412
    b, c, d, e, f, g, h = 0.354, 1.287, 0.219, 1.543, 0.412, 1.109, 0.331
    
    # Generate trajectory points
    N = 10000
    x, y, z, w = 0.1, 0.2, 0.3, 0.4
    
    # Discard transients
    for _ in range(1000):
        x = (np.sin(a * y) + b * np.cos(x) + w) % 1.0
        y = (np.sin(c * x) + d * np.cos(y)) % 1.0
        z = (np.sin(e * z) + f * np.cos(w)) % 1.0
        w = (np.sin(g * w) + h * np.cos(z)) % 1.0
        
    x_pts, y_pts, z_pts, w_pts = [], [], [], []
    for _ in range(N):
        x = (np.sin(a * y) + b * np.cos(x) + w) % 1.0
        y = (np.sin(c * x) + d * np.cos(y)) % 1.0
        z = (np.sin(e * z) + f * np.cos(w)) % 1.0
        w = (np.sin(g * w) + h * np.cos(z)) % 1.0
        
        x_pts.append(x)
        y_pts.append(y)
        z_pts.append(z)
        w_pts.append(w)
        
    x_pts = np.array(x_pts)
    y_pts = np.array(y_pts)
    z_pts = np.array(z_pts)
    w_pts = np.array(w_pts)
    
    # Ensure plots folder exists
    os.makedirs("results/plots", exist_ok=True)
    
    # Plot 1: 3D strange attractor (x, y, z)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_pts, y_pts, z_pts, c=w_pts, cmap='viridis', s=0.5, alpha=0.6)
    ax.set_xlabel('State Variable X')
    ax.set_ylabel('State Variable Y')
    ax.set_zlabel('State Variable Z')
    ax.set_title('3D Phase Portrait strange attractor of the 4D-DSCHM')
    plt.tight_layout()
    plt.savefig("results/plots/strange_attractor_3d.png", dpi=300)
    plt.close()
    
    # Plot 2: 2D projection (x, y)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_pts, y_pts, s=0.3, color='navy', alpha=0.5)
    plt.xlabel('State Variable X')
    plt.ylabel('State Variable Y')
    plt.title('2D Projection (X vs Y) strange attractor of the 4D-DSCHM')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/plots/strange_attractor_2d_xy.png", dpi=300)
    plt.close()
    
    print("Strange Attractor phase portrait plots generated and saved:")
    print("  - results/plots/strange_attractor_3d.png")
    print("  - results/plots/strange_attractor_2d_xy.png")
    print("====================================================")

if __name__ == "__main__":
    main()
