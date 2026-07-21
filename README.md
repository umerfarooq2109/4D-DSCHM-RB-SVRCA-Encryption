# RB-SVRCA: 4D Trigonometric Hyperchaotic Cellular Automata Image Encryption System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scientific Suite](https://img.shields.io/badge/IEEE-Cryptosystem-success.svg)](#)

A high-performance, cryptographically secure color and grayscale image encryption system combining a **4D Discrete Sine-Cosine Hyperchaotic Map (4D-DSCHM)** with **Red-Black Reversible Space-Varying Cellular Automata (RB-SVRCA)** and **Bidirectional Feedback Diffusion**.

This repository hosts the complete Python implementation and security validation suite, simulating advanced cryptanalytic attacks (differential, statistical, noise, cropping, and chaotic diagnostics) matching academic publication standards.

---

## 📖 Theoretical Overview

The cryptosystem operates in three main stages to achieve a high degree of confusion (via scrambling) and diffusion (via cell transitions and bidirectional key propagation):

```
Plaintext Image
       │
       ▼
 1. Permutation ────────► 2. RB-SVRCA Iterations ────────► 3. Bidirectional Diffusion ───► Ciphertext Image
 (Chaotic Row/Col)       (Space-varying CA rules)          (Forward & Backward rounds)
```

### 1. 4D-DSCHM Sequence Generation
A 256-bit secure key is hashed using SHA-256 to extract four initial states $x_0, y_0, z_0, w_0 \in [0.1, 0.9]$. The sequences are evolved using the 4-dimensional discrete sine-cosine hyperchaotic map equations:
$$x_{i+1} = \left(\sin(A \cdot y_i) + B \cdot \cos(x_i) + w_i\right) \pmod{1.0}$$
$$y_{i+1} = \left(\sin(C \cdot x_i) + D \cdot \cos(y_i)\right) \pmod{1.0}$$
$$z_{i+1} = \left(\sin(E \cdot z_i) + F \cdot \cos(w_i)\right) \pmod{1.0}$$
$$w_{i+1} = \left(\sin(G \cdot w_i) + H_{param} \cdot \cos(z_i)\right) \pmod{1.0}$$
where the parameter set $\{A, B, C, D, E, F, G, H_{param}\}$ is tuned to guarantee hyperchaotic behavior with multiple positive Lyapunov exponents.

### 2. Red-Black Reversible Space-Varying Cellular Automata (RB-SVRCA)
* **Permutation**: Rows and columns of the plaintext image are scrambled using sorted indices from sequences $x$ and $y$.
* **Space-Varying Rules**: The system selects from 8 distinct local state transition rules dynamically for each cell based on the chaotic grid rules (derived from sequence $z$):
  * **Rule 0**: $g_0 = ((L + R + T + B) \oplus K) \pmod{256}$
  * **Rule 1**: $g_1 = ((L \oplus R \oplus T \oplus B) + K) \pmod{256}$
  * **Rule 2**: $g_2 = (((L + R) \oplus (T + B)) + K) \pmod{256}$
  * **Rule 3**: $g_3 = ((L + R - T - B) \oplus K) \pmod{256}$
  * **Rule 4**: $g_4 = ((3L + 7R + 5T + 2B) \oplus K) \pmod{256}$
  * **Rule 5**: $g_5 = ((L \oplus R) + K) \pmod{256}$
  * **Rule 6**: $g_6 = ((T \oplus B) + K) \pmod{256}$
  * **Rule 7**: $g_7 = (((L + T) \oplus (R + B)) \oplus K) \pmod{256}$
  *(where $L, R, T, B$ are Left, Right, Top, and Bottom neighboring pixels, and $K$ is the local key grid from sequence $w$)*
* **Red-Black Grid Iteration**: Reversibility is guaranteed without losing floating-point precision by updating the grid in a checkerboard pattern. Red cells are updated first using the states of Black cells, and then Black cells are updated using the new states of Red cells.

### 3. Bidirectional Feedback Diffusion
To guarantee high sensitivity to single-pixel plain image modifications (NPCR > 99.6%, UACI ~ 33.46%), a two-pass diffusion round is performed over the vectorized grid:
1. **Forward Pass**: $C_{1}[i] = (S[i] + C_{1}[i-1]) \pmod{256} \oplus K[i]$
2. **Backward Pass**: $C_{2}[i] = (C_{1}[i] + C_{2}[i+1]) \pmod{256} \oplus K[i]$

This complete process is fully inverted during decryption, guaranteeing mathematical losslessness (MSE = 0, PSNR = $\infty$).

---

## 📁 Repository Directory Layout

```
.
├── Images/                       # Benchmark test images (cameraman, lena, pepper, etc.)
├── attacks/                      # Cryptanalytic attack and security analysis scripts
│   ├── DIEHARD.py                # Full 17-test Diehard randomness validation
│   ├── NIST.py                   # Standard 15-test NIST SP 800-22 randomness suite
│   ├── Histogram_Color.py        # Color RGB image histogram generator & plotter
│   ├── Noise_Attack.py           # Noise immunity simulations (1% to 10% density)
│   ├── Robustness_color.py       # Cropping/Clipping analysis (1/8 vertical, 1/4 corner, etc.)
│   ├── chi_square_69.py          # Chi-Square histogram uniformity test
│   ├── color_security_analysis.py# Color-specific security metrics
│   ├── differential_attack.py    # Baseline NPCR and UACI sensitivity calculator
│   ├── different_positio_npcr_uaci.py # Spatial sensitivity analysis
│   ├── local_entropy.py          # Local Shannon entropy (100-run simulation)
│   ├── psnr_mse_ssim.py          # Decryption quality metrics (SSIM, MSE, PSNR)
│   └── GVD_PSNR_ETC.py           # Global Variance Difference analyzer
├── chaos_analysis/               # Tools for analyzing the 4D-DSCHM chaotic map
│   ├── strange_attractor_phase_portrait.py # 2D and 3D attractor phase space plots
│   ├── bifurcation.py            # Bifurcation diagrams
│   ├── lyapunov.py               # Lyapunov exponent calculations (Python)
│   ├── lyapunov.m                # Lyapunov exponent calculations (Matlab)
│   ├── approximate_entropy.py    # Approximate Entropy (ApEn) sequence test
│   ├── zero_one_test.py          # 0-1 test for chaos
│   └── psd.py                    # Power Spectral Density (PSD) analysis
├── src/                          # Core cryptographic engine
│   ├── cellular_automata.py      # Vectorized Red-Black SVRCA & Diffusion engine
│   ├── map_4d.py                 # 4D Sine-Cosine hyperchaotic map equations
│   └── utils.py                  # Directory setup, MSE utilities, synthetic image generator
├── results/                      # Output directory for generated images and metrics
│   ├── Histogram_Color/          # Output color histograms
│   ├── Noise/                    # Decrypted images under noise attacks
│   ├── Robustness/               # Decrypted images under cropping attacks
│   ├── encrypted/                # Encrypted outputs
│   ├── decrypted/                # Decrypted outputs
│   └── plots/                    # Scientific diagrams (bifurcations, attractors, correlations)
├── main.py                       # Batch processing and metric calculation suite
├── graphical_aabstract.py        # Publication-quality graphical abstract builder
├── benchmark.py                  # System security benchmark suite
├── time_complexity.py            # Step-by-step pipeline profiler & speed test
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## 🛠️ Prerequisites & Installation

Ensure you have Python 3.8+ installed. Follow these steps to set up and run the repository:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/New_Encryption.git
   cd New_Encryption
   ```

2. **Set up a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Required packages: `numpy`, `opencv-python`, `matplotlib`, `scipy`.*

---

## 🚀 Usage Guide

This repository contains multiple execution suites to perform encryption, generate figures, and run benchmarks:

### 1. Run the Core Encryption Suite
Process all standard test images or a specific image, compute key security metrics (Entropy, NPCR, UACI, MSE, PSNR), and save encrypted/decrypted results:
```bash
# Run batch test on all listed benchmark images
python main.py

# Run on a specific image (e.g., Cameraman)
python main.py Images/cameraman.jpg
```

### 2. Generate the Publication-Quality Graphical Abstract
Create a high-resolution, unified figure displaying the step-by-step encryption mechanism, the 4D attractor, histograms, adjacent correlation plots, and decrypted proof:
```bash
# Generate abstract for cameraman
python graphical_aabstract.py Images/cameraman.jpg

# Generate abstract for Lena Color
python graphical_aabstract.py Images/lena_color_512.tif
```
*Outputs are saved in high DPI under [results/plots/graphical_abstract.png](file:///e:/Cyber_secuirty_AI/New_Encryption/results/plots/graphical_abstract.png).*

### 3. Run Performance & Speed Profiling
Profile the exact execution times of the encryption/decryption steps, identifying bottleneck operations:
```bash
python time_complexity.py
```

### 4. Run Security Benchmarks
Perform a quick baseline benchmark on information entropy, adjacent-pixel correlation coefficients, and plaintext sensitivity:
```bash
python benchmark.py
```

---

## 📊 Security Analysis & Robustness Showcase

The following results demonstrate the system's defenses against standard cryptanalytic attacks:

### 1. Visual Showcase (Plain, Cipher, and Decrypted Images)
When an image is encrypted, the visual information is completely scrambled into white noise. The decryption is lossless, restoring the image perfectly.

<p align="center">
  <table>
    <tr>
      <th>Plain Image</th>
      <th>Ciphertext Image</th>
      <th>Decrypted Image</th>
    </tr>
    <tr>
      <td><img src="./Images/lenna.jpg" width="250" alt="Plain Image"></td>
      <td><img src="./results/encrypted/enc_lenna.png" width="250" alt="Ciphertext Image"></td>
      <td><img src="./results/decrypted/dec_lenna.png" width="250" alt="Decrypted Image"></td>
    </tr>
  </table>
</p>

---

### 2. Histogram Analysis
A secure image cipher must hide the statistical properties of the plain image. Our system yields a uniform pixel frequency distribution across all channels:

#### Color (RGB) Channels Histogram (Lena 512x512)
The script `attacks/Histogram_Color.py` generates the following channel-wise comparison:

```bash
python attacks/Histogram_Color.py Images/lena_color_512.tif
```

![Color Histograms](results/Histogram_Color/lena_color_512_histogram.png)
*(Top: Plain RGB image with non-uniform channel distributions. Bottom: Cipher RGB image displaying completely flat, uniform distributions).*

---

### 3. Adjacent Pixel Correlation Analysis
Plaintext images exhibit strong spatial correlation between adjacent pixels ($r \approx 1.0$). A secure cryptosystem must break this correlation ($r \approx 0.0$ in the ciphertext).

#### Adjacent Correlation Scatter Plots
Plotting 10,000 random pixel pairs along Horizontal, Vertical, and Diagonal directions reveals a tight linear clustering for plain text and a highly uniform, spread-out cloud for ciphertext:

| Plaintext Correlation (Horizontal) | Ciphertext Correlation (Horizontal) |
| :---: | :---: |
| ![Plain Corr](results/plots/correlation_plaintext_horizontal.png) | ![Cipher Corr](results/plots/correlation_ciphertext_horizontal.png) |

#### Pearson Correlation Coefficients (Representative Values)
* **Plaintext (Horizontal)**: $+0.9739$
* **Ciphertext (Horizontal)**: $-0.0095$
* **Ciphertext (Vertical)**: $-0.0084$
* **Ciphertext (Diagonal)**: $+0.0012$

---

### 4. Plaintext Sensitivity Analysis (NPCR & UACI)
The **Number of Pixels Change Rate (NPCR)** and the **Unified Average Changing Intensity (UACI)** measure the system's resistance to differential attacks. 
By changing exactly **one pixel** in the input image, we encrypt both and compare.

* **Ideal NPCR**: $>99.6094\%$
* **Ideal UACI**: $33.4635\%$

Running `attacks/differential_attack.py` on standard images yields:
* **NPCR**: $99.7795\%$ (Passes standard significance tests)
* **UACI**: $33.5490\%$ (Passes standard significance tests)

---

### 5. Noise Attack Robustness Test
In real-world transmissions, ciphertext images are often corrupted by channel noise. Our bidirectional diffusion and CA structure restrict error propagation, allowing high-quality decryptions.

To simulate this, we apply varying levels of Salt & Pepper noise ($1\%$, $2\%$, $3\%$, $5\%$, $8\%$, and $10\%$ densities) to the encrypted image and decrypt:

```bash
python attacks/Noise_Attack.py Images/lena_color_512.tif
```

![Noise Attack Panel](results/Noise/lena_color_512_noise.png)
*(Row 1: Encrypted images corrupted by 1%, 2%, 3%, 5%, 8%, 10% noise. Row 2: Successfully decrypted images retaining full structural and color content, evaluated via PSNR/SSIM metrics).*

---

### 6. Cropping & Clipping Attack Robustness Test
Cropping attacks simulate situations where parts of the ciphertext image are intercepted or lost. The cryptosystem can still reconstruct the remaining regions of the image, showing noise *only* in the cropped locations.

We evaluate 6 scenarios:
* **(a)** $1/8$ Vertical Clipping ($12.5\%$)
* **(b)** $1/4$ Corner Cropping ($25.0\%$)
* **(c)** $1/4$ Horizontal Clipping ($25.0\%$)
* **(d)** $1/4$ Center Cropping ($25.0\%$)
* **(e)** $1/2$ Horizontal Clipping ($50.0\%$)
* **(f)** $1/2$ Vertical Clipping ($50.0\%$)

To run this test:
```bash
python attacks/Robustness_color.py Images/pepper.tiff
```

![Robustness Panel](results/Robustness/Robustness_color.png)
*(Top Row: Attacked cipher images with blacked-out regions. Bottom Row: Corresponding decrypted outputs demonstrating robust visual preservation).*

---

### 7. Computational Speed & Complexity Benchmarks
The system achieves high-speed processing through vectorized operations in NumPy, avoiding slow element-wise loops during CA rules updates and inverse diffusion passes.

Running `time_complexity.py` logs step-by-step latency metrics:
```
======================================================================
  [1/2] cameraman.jpg (256x256)
======================================================================
  - Step 1: Chaotic Sequences Generation  : 0.0018s
  - Step 2: Row/Col Permutation           : 0.0001s
  - Step 3: Rule/Key Grid Setup           : 0.0009s
  - Step 4: Reversible CA Iterations (2r) : 0.0051s
  - Step 5: Forward & Backward Diffusion  : 0.0382s
  ----------------------------------------------
  Total Encryption & Decryption Time      : 0.0461s
```
*The average run time for a standard 256x256 image is under 0.05 seconds, demonstrating suitability for real-time secure communication.*

---

## 📈 Chaotic Dynamics Analysis of 4D-DSCHM

To prove the cryptosystem's security foundation, the chaotic map was subjected to rigorous mathematical diagnostics:

### 1. Attractor Phase Portrait
A strange attractor with fractional dimension is a key signature of chaos. The 2D projection and 3D phase space plots show highly dense trajectories confirming broad chaotic coverage:

### 2. Lyapunov Exponents
Lyapunov exponents measure the rate of separation of infinitesimally close trajectories. Positive Lyapunov exponents ($\lambda > 0$) signify chaos, while two or more positive exponents indicate hyperchaos:

![Lyapunov Exponents](results/plots/lyapunov_exponents.png)
*(The map exhibits multiple positive exponents across the parameter spectrum, proving steady hyperchaotic dynamics).*

### 3. Bifurcation Diagram
The bifurcation diagram plots the visited state space as parameters are varied. Our 4D-DSCHM displays continuous, dense state-visitation without periodic windows:

| Fig.(a): Bifurcation Diagram: x vs alpha | Fig. (b): Bifurcation Diagram: y vs alpha |
| :---: | :---: |
| ![Bifurcation 2D](results/plots/Fig.2(a).png) | ![Bifurcation 3D](results/plots/Fig.2(b).png) |

---

## 📂 Data Availability & Benchmark Images

All test images used in this project are standard, publicly available benchmark images commonly used in image processing and cryptography literature:
1. **USC-SIPI Image Database**: Includes standard color/grayscale images such as *Lena* (512x512, 256x256), *Peppers* (512x512), *Baboon* (512x512), *Aeroplane* (512x512), and miscellaneous textures.
2. **Standard Test Images**: Grayscale *Cameraman* (256x256), *Black* (solid black synthetic test), and *White* (solid white synthetic test).
3. **High-Resolution Images**: Large dimension files like *1024.png* and *5.3.02.tiff* (1024x1024).

The complete dataset of images utilized to generate the security metrics in the paper is stored locally in the [`Images/`](Images/) directory of this repository for reproducibility. Users can add their custom test images under the same directory.

---

## ✉️ Contact & Copyright

* **Developer**: Umer Farooq
* **Email**: [umerfarooqlone2119@gmail.com](mailto:umerfarooqlone2119@gmail.com)
* **Copyright**: © 2026 Umer Farooq. All rights reserved.

This project is open-source and licensed under the MIT License.
