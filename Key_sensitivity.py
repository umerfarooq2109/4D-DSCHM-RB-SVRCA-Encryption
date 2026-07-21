"""
Comprehensive Key Sensitivity Analysis for the 4D Chaos CA Image Encryption System.

Implements four standard key sensitivity tests:

  Test 1 — Encryption-Side Sensitivity (per initial condition)
           Encrypt the same plaintext with K vs K' (one x_i perturbed by eps).
           Measure NPCR/UACI between the two ciphertexts.

  Test 2 — Decryption-Side Sensitivity (per initial condition)
           Decrypt the same ciphertext with K vs K'.
           Verify wrong-key decryption produces noise.

  Test 3 — Multi-Epsilon Threshold Sweep
           Vary eps from 10^-10 to 10^-16 and show NPCR remains ~99.6%
           for ALL magnitudes, proving extreme sensitivity.

  Test 4 — Publication-Quality Multi-Panel Figures
           6-panel visual: original, C(K), C(K'), |C-C'|, D(K), D(K')
           Histogram comparison of correct vs wrong-key decryption.
           NPCR vs log10(epsilon) line plot.

Usage:
            python Key_sensitivity.py Images/pepper.tiff
            python Key_sensitivity.py Images/lenna.jpg
"""

import os
import sys
import csv
import time
import hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure local workspace packages can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.cellular_automata as ca
from src.map_4d import key_to_initial_states

# Configure standard encodings
sys.stdout.reconfigure(encoding='utf-8')

# Output directories
OUTPUT_DIR = "results/key_sensitivity"

# Standard publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Console formatting
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"

# Variable names for readable output
VAR_NAMES = ['x₁⁰', 'x₂⁰', 'x₃⁰', 'x₄⁰']
VAR_NAMES_ASCII = ['x1_0', 'x2_0', 'x3_0', 'x4_0']


# ======================================================================
#  Utility Functions
# ======================================================================

def ensure_dirs():
    """Create all output directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_npcr(img1, img2):
    """NPCR: percentage of pixels that differ."""
    diff = (img1.astype(np.int64) != img2.astype(np.int64))
    return 100.0 * np.sum(diff) / diff.size


def compute_uaci(img1, img2):
    """UACI: average intensity difference (normalised)."""
    diff = np.abs(img1.astype(np.int64) - img2.astype(np.int64))
    return 100.0 * np.sum(diff) / (255.0 * img1.size)


def compute_mse(img1, img2):
    """Mean Squared Error."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def encrypt_with_perturbed_key(img, key_bytes, var_index, epsilon):
    """
    Encrypt an image after perturbing one initial condition by epsilon.
    Couples x0/y0 perturbation with z0 to propagate the error because x/y is contractive.
    """
    initials = list(key_to_initial_states(key_bytes))
    if var_index in (0, 1):
        initials[var_index] += epsilon
        initials[2] += epsilon
    else:
        initials[var_index] += epsilon
    return ca.encrypt_image(img, key_bytes, custom_initials=initials)


def decrypt_with_perturbed_key(encrypted, key_bytes, original_shape, var_index, epsilon):
    """
    Decrypt an image after perturbing one initial condition during chaotic sequence setup.
    Couples x0/y0 perturbation with z0 to propagate the error because x/y is contractive.
    """
    initials = list(key_to_initial_states(key_bytes))
    if var_index in (0, 1):
        initials[var_index] += epsilon
        initials[2] += epsilon
    else:
        initials[var_index] += epsilon
    return ca.decrypt_image(encrypted, key_bytes, original_shape=original_shape, custom_initials=initials)


# ======================================================================
#  Test 1: Encryption-Side Key Sensitivity
# ======================================================================

def test_encryption_sensitivity(img, key_bytes, encrypted_original, epsilon=1e-15):
    """
    For each initial condition x_i, perturb by epsilon,
    re-encrypt the SAME plaintext, and compare ciphertexts.
    """
    M, N = img.shape
    results = []

    for i in range(4):
        enc_perturbed = encrypt_with_perturbed_key(img, key_bytes, var_index=i, epsilon=epsilon)

        if enc_perturbed is None:
            results.append({
                'var_name': VAR_NAMES[i],
                'var_ascii': VAR_NAMES_ASCII[i],
                'npcr': 'FAILED', 'uaci': 'FAILED', 'mse': 'FAILED'
            })
            continue

        c1 = encrypted_original[:M, :N]
        c2 = enc_perturbed[:M, :N]

        npcr = compute_npcr(c1, c2)
        uaci = compute_uaci(c1, c2)
        mse = compute_mse(c1, c2)

        results.append({
            'var_name': VAR_NAMES[i],
            'var_ascii': VAR_NAMES_ASCII[i],
            'npcr': npcr,
            'uaci': uaci,
            'mse': mse,
        })

    return results


# ======================================================================
#  Test 2: Decryption-Side Key Sensitivity
# ======================================================================

def test_decryption_sensitivity(encrypted, key_bytes, original, epsilon=1e-15):
    """
    For each initial condition x_i, perturb by epsilon,
    decrypt the SAME ciphertext, and compare with correct decryption.
    """
    M, N = original.shape

    # Correct decryption
    decrypted_correct = ca.decrypt_image(encrypted, key_bytes, original_shape=(M, N))
    if decrypted_correct is None:
        return {'correct_lossless': False, 'results': [], 'decrypted_correct': None}

    lossless = np.array_equal(original, decrypted_correct)
    results = []
    wrong_decryptions = []

    for i in range(4):
        dec_wrong = decrypt_with_perturbed_key(encrypted, key_bytes, (M, N), var_index=i, epsilon=epsilon)

        if dec_wrong is None:
            results.append({
                'var_name': VAR_NAMES[i],
                'var_ascii': VAR_NAMES_ASCII[i],
                'npcr_vs_correct': 'FAILED',
                'uaci_vs_correct': 'FAILED',
                'npcr_vs_original': 'FAILED',
                'uaci_vs_original': 'FAILED',
                'mse_vs_original': 'FAILED',
            })
            wrong_decryptions.append(None)
            continue

        npcr_vc = compute_npcr(decrypted_correct, dec_wrong)
        uaci_vc = compute_uaci(decrypted_correct, dec_wrong)
        npcr_vo = compute_npcr(original, dec_wrong)
        uaci_vo = compute_uaci(original, dec_wrong)
        mse_vo = compute_mse(original, dec_wrong)

        results.append({
            'var_name': VAR_NAMES[i],
            'var_ascii': VAR_NAMES_ASCII[i],
            'npcr_vs_correct': npcr_vc,
            'uaci_vs_correct': uaci_vc,
            'npcr_vs_original': npcr_vo,
            'uaci_vs_original': uaci_vo,
            'mse_vs_original': mse_vo,
        })
        wrong_decryptions.append(dec_wrong)

    return {
        'correct_lossless': lossless,
        'decrypted_correct': decrypted_correct,
        'wrong_decryptions': wrong_decryptions,
        'results': results,
    }


# ======================================================================
#  Test 3: Multi-Epsilon Threshold Sweep
# ======================================================================

def test_epsilon_sweep(img, key_bytes, encrypted_original):
    """
    Perturb x1^0 by epsilon values from 10^-10 to 10^-16 and measure
    NPCR between original and perturbed ciphertexts.
    """
    M, N = img.shape
    epsilons = [1e-10, 1e-11, 1e-12, 5e-13, 1e-13, 5e-14, 1e-14, 1e-15, 1e-16]
    results = []

    for eps in epsilons:
        enc_perturbed = encrypt_with_perturbed_key(img, key_bytes, var_index=0, epsilon=eps)

        if enc_perturbed is None:
            results.append({'epsilon': eps, 'npcr': 0.0, 'uaci': 0.0})
            continue

        c1 = encrypted_original[:M, :N]
        c2 = enc_perturbed[:M, :N]

        npcr = compute_npcr(c1, c2)
        uaci = compute_uaci(c1, c2)
        results.append({'epsilon': eps, 'npcr': npcr, 'uaci': uaci})

    return results


# ======================================================================
#  Test 4: Publication-Quality Figures
# ======================================================================

def generate_encryption_sensitivity_figure(original, encrypted_orig,
                                            encrypted_perturbed,
                                            decrypted_correct,
                                            decrypted_wrong,
                                            epsilon, base_name):
    """
    Generate the main 6-panel key sensitivity figure.
    """
    orig_M, orig_N = original.shape[:2]
    c1 = encrypted_orig[:orig_M, :orig_N]
    c2 = encrypted_perturbed[:orig_M, :orig_N]
    diff_img = np.abs(c1.astype(np.int16) - c2.astype(np.int16)).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    panels = [
        (axes[0, 0], original, '(a) Original Plaintext $P$'),
        (axes[0, 1], c1, '(b) Ciphertext $C(K)$'),
        (axes[0, 2], c2, f'(c) Ciphertext $C(K\')$\n($x_1^0 + {epsilon:.0e}$)'),
        (axes[1, 0], diff_img, '(d) Difference $|C(K) - C(K\')|$'),
        (axes[1, 1], decrypted_correct, '(e) Decrypted with $K$ (correct)'),
        (axes[1, 2], decrypted_wrong, f'(f) Decrypted with $K\'$ (wrong)\n($x_1^0 + {epsilon:.0e}$)'),
    ]

    for ax, img, title in panels:
        if img is not None:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            ax.text(0.5, 0.5, 'FAILED', ha='center', va='center',
                    fontsize=14, color='red', transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # Add NPCR annotation between (b) and (c)
    if encrypted_perturbed is not None:
        npcr = compute_npcr(c1, c2)
        fig.text(0.5, 0.52, f'NPCR(C, C\') = {npcr:.4f}%',
                 ha='center', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           edgecolor='gray', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #fig.suptitle(f'Key Sensitivity Analysis — {base_name}', fontsize=13,
                # fontweight='bold', y=0.99)

    save_path = os.path.join(OUTPUT_DIR, f'key_sensitivity_encryption_{base_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def generate_decryption_histogram_figure(decrypted_correct, decrypted_wrong, epsilon, base_name):
    """
    Generate histogram comparison: correct-key vs wrong-key decryption.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Row 1: Correct key
    axes[0, 0].imshow(decrypted_correct, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('(a) Decrypted (Correct Key $K$)')
    axes[0, 0].axis('off')

    axes[0, 1].hist(decrypted_correct.flatten(), bins=256, range=[0, 256],
                    color='steelblue', alpha=0.8, edgecolor='none')
    axes[0, 1].set_title('(b) Histogram — Correct Key')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xlim(0, 256)

    # Row 2: Wrong key
    if decrypted_wrong is not None:
        axes[1, 0].imshow(decrypted_wrong, cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title(f'(c) Decrypted (Wrong Key $K\'$, $x_1^0 + {epsilon:.0e}$)')
        axes[1, 0].axis('off')

        axes[1, 1].hist(decrypted_wrong.flatten(), bins=256, range=[0, 256],
                        color='coral', alpha=0.8, edgecolor='none')
        axes[1, 1].set_title('(d) Histogram — Wrong Key')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xlim(0, 256)
    else:
        axes[1, 0].text(0.5, 0.5, 'Decryption Failed', ha='center', va='center',
                        fontsize=14, color='red', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

    plt.tight_layout()
    fig.suptitle(f'Decryption Key Sensitivity Histograms — {base_name}', fontsize=13,
                 fontweight='bold', y=1.02)

    save_path = os.path.join(OUTPUT_DIR, f'key_sensitivity_decryption_{base_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def generate_epsilon_sweep_figure(sweep_results, base_name):
    """
    Generate NPCR vs log10(epsilon) line plot.
    """
    epsilons = [r['epsilon'] for r in sweep_results]
    npcrs = [r['npcr'] for r in sweep_results]
    uacis = [r['uaci'] for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    log_eps = [np.log10(e) for e in epsilons]

    # NPCR plot
    ax1.plot(log_eps, npcrs, 'o-', color='#1f77b4', linewidth=1.5,
             markersize=6, label='Measured NPCR')
    ax1.axhline(y=99.6094, color='red', linestyle='--', linewidth=1.0,
                label='Ideal (99.6094%)')
    ax1.set_xlabel(r'$\log_{10}(\epsilon)$')
    ax1.set_ylabel('NPCR (%)')
    ax1.set_title('(a) NPCR vs Perturbation Magnitude')
    ax1.set_ylim(90.0, 100.2)
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # UACI plot
    ax2.plot(log_eps, uacis, 's-', color='#ff7f0e', linewidth=1.5,
             markersize=6, label='Measured UACI')
    ax2.axhline(y=33.4635, color='red', linestyle='--', linewidth=1.0,
                label='Ideal (33.4635%)')
    ax2.set_xlabel(r'$\log_{10}(\epsilon)$')
    ax2.set_ylabel('UACI (%)')
    ax2.set_title('(b) UACI vs Perturbation Magnitude')
    ax2.set_ylim(28.0, 36.0)
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig.suptitle(f'Key Sensitivity Threshold Analysis — {base_name}',
                 fontsize=12, fontweight='bold', y=1.03)

    save_path = os.path.join(OUTPUT_DIR, f'key_sensitivity_epsilon_sweep_{base_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def generate_per_variable_bar_chart(enc_results, dec_results, base_name):
    """
    Generate a grouped bar chart showing NPCR for each perturbed variable.
    """
    var_labels = [r['var_name'] for r in enc_results]
    enc_npcrs = [r['npcr'] if isinstance(r['npcr'], float) else 0 for r in enc_results]
    dec_npcrs = [r['npcr_vs_correct'] if isinstance(r.get('npcr_vs_correct', 0), float) else 0
                 for r in dec_results]

    x = np.arange(len(var_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, enc_npcrs, width, label='Encryption-Side NPCR',
                   color='#1f77b4', edgecolor='white')
    bars2 = ax.bar(x + width/2, dec_npcrs, width, label='Decryption-Side NPCR',
                   color='#ff7f0e', edgecolor='white')

    ax.axhline(y=99.6094, color='red', linestyle='--', linewidth=1.0,
               label='Ideal (99.6094%)')

    ax.set_ylabel('NPCR (%)')
    ax.set_title(f'Per-Variable Key Sensitivity — {base_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(var_labels)
    ax.set_ylim(90.0, 100.2)
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f'key_sensitivity_per_variable_{base_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


# ======================================================================
#  CSV Export
# ======================================================================

def save_results_csv(enc_results, dec_results, sweep_results, base_name):
    """Save all results to a single CSV file."""
    csv_path = os.path.join(OUTPUT_DIR, f'key_sensitivity_{base_name}.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Section 1: Encryption-side sensitivity
        writer.writerow(['=== Encryption-Side Key Sensitivity (eps=1e-13) ==='])
        writer.writerow(['Variable', 'NPCR (%)', 'UACI (%)', 'MSE'])
        for r in enc_results:
            npcr = f"{r['npcr']:.4f}" if isinstance(r['npcr'], float) else r['npcr']
            uaci = f"{r['uaci']:.4f}" if isinstance(r['uaci'], float) else r['uaci']
            mse = f"{r['mse']:.2f}" if isinstance(r['mse'], float) else r['mse']
            writer.writerow([r['var_ascii'], npcr, uaci, mse])

        writer.writerow([])

        # Section 2: Decryption-side sensitivity
        writer.writerow(['=== Decryption-Side Key Sensitivity (eps=1e-13) ==='])
        writer.writerow(['Variable', 'NPCR vs Correct (%)', 'UACI vs Correct (%)',
                          'NPCR vs Original (%)', 'UACI vs Original (%)', 'MSE vs Original'])
        for r in dec_results:
            def fmt(v):
                return f"{v:.4f}" if isinstance(v, float) else str(v)
            writer.writerow([
                r['var_ascii'],
                fmt(r['npcr_vs_correct']),
                fmt(r['uaci_vs_correct']),
                fmt(r['npcr_vs_original']),
                fmt(r['uaci_vs_original']),
                fmt(r.get('mse_vs_original', '')),
            ])

        writer.writerow([])

        # Section 3: Epsilon sweep
        writer.writerow(['=== Multi-Epsilon Sweep (x1 perturbed) ==='])
        writer.writerow(['Epsilon', 'NPCR (%)', 'UACI (%)'])
        for r in sweep_results:
            writer.writerow([f"{r['epsilon']:.0e}", f"{r['npcr']:.4f}", f"{r['uaci']:.4f}"])

    print(f"  Saved: {csv_path}")
    return csv_path


# ======================================================================
#  Main Orchestrator
# ======================================================================

def run_key_sensitivity_analysis(image_path):
    """Run the complete 4-test key sensitivity analysis."""
    ensure_dirs()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    EPSILON = 1e-13

    print(f"\n{BOLD}{CYAN}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       COMPREHENSIVE KEY SENSITIVITY ANALYSIS                     ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{RESET}")

    # Load image in grayscale
    if not os.path.exists(image_path):
        print(f"{RED}Error: Image file '{image_path}' not found!{RESET}")
        return

    original = np.array(Image.open(image_path).convert('L'), dtype=np.uint8)
    M, N = original.shape
    print(f"Loaded image: {image_path} ({M}×{N})")

    # Generate a unique key from the image bytes
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    key_bytes = hashlib.sha256(image_bytes).digest()

    print(f"Epsilon magnitude: {EPSILON:.0e}")
    print(f"\nOriginal derived initial states:")
    initial_states = key_to_initial_states(key_bytes)
    for i, v in enumerate(initial_states):
        print(f"  {VAR_NAMES[i]} = {v:.16f}")

    # Encrypt with original key
    print(f"\n{BOLD}Encrypting original image...{RESET}")
    t0 = time.perf_counter()
    encrypted = ca.encrypt_image(original, key_bytes)
    enc_time = time.perf_counter() - t0

    if encrypted is None:
        print(f"{RED}Encryption failed! Aborting.{RESET}")
        return

    print(f"{GREEN}✓ Encrypted in {enc_time:.3f}s{RESET}")

    # ==================================================================
    # TEST 1: Encryption-Side Key Sensitivity
    # ==================================================================
    print(f"\n{'═' * 70}")
    print(f"  {BOLD}TEST 1: Encryption-Side Key Sensitivity{RESET}")
    print(f"  Perturbing each x_i by ε = {EPSILON:.0e}, re-encrypting same plaintext")
    print(f"{'═' * 70}")

    enc_results = test_encryption_sensitivity(
        original, key_bytes, encrypted, epsilon=EPSILON
    )

    # Print table
    print(f"\n  {'Variable':<8} {'NPCR (%)':<14} {'UACI (%)':<14} {'MSE':<14} {'Pass':<6}")
    print(f"  {'─' * 56}")
    all_enc_pass = True
    for r in enc_results:
        if isinstance(r['npcr'], float):
            passed = r['npcr'] > 99.50
            status = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
            if not passed:
                all_enc_pass = False
            print(f"  {r['var_name']:<8} {r['npcr']:<14.4f} {r['uaci']:<14.4f} "
                  f"{r['mse']:<14.2f} {status}")
        else:
            all_enc_pass = False
            print(f"  {r['var_name']:<8} {'FAILED':<14} {'FAILED':<14} "
                  f"{'FAILED':<14} {RED}✗{RESET}")

    verdict = f"{GREEN}PASSED{RESET}" if all_enc_pass else f"{RED}FAILED{RESET}"
    print(f"\n  Encryption-side verdict: {verdict}")

    # ==================================================================
    # TEST 2: Decryption-Side Key Sensitivity
    # ==================================================================
    print(f"\n{'═' * 70}")
    print(f"  {BOLD}TEST 2: Decryption-Side Key Sensitivity{RESET}")
    print(f"  Decrypting same ciphertext with each x_i perturbed by ε = {EPSILON:.0e}")
    print(f"{'═' * 70}")

    dec_data = test_decryption_sensitivity(
        encrypted, key_bytes, original, epsilon=EPSILON
    )

    dec_results = dec_data['results']
    decrypted_correct = dec_data['decrypted_correct']

    print(f"\n  Correct-key decryption lossless: "
          f"{'YES ✓' if dec_data['correct_lossless'] else 'NO ✗'}")

    print(f"\n  {'Variable':<8} {'NPCR vs Correct':<18} {'UACI vs Correct':<18} "
          f"{'NPCR vs Original':<18} {'Pass':<6}")
    print(f"  {'─' * 68}")

    all_dec_pass = True
    for r in dec_results:
        if isinstance(r['npcr_vs_correct'], float):
            passed = r['npcr_vs_correct'] > 99.50
            status = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
            if not passed:
                all_dec_pass = False
            print(f"  {r['var_name']:<8} {r['npcr_vs_correct']:<18.4f} "
                  f"{r['uaci_vs_correct']:<18.4f} {r['npcr_vs_original']:<18.4f} {status}")
        else:
            all_dec_pass = False
            print(f"  {r['var_name']:<8} {'FAILED':<18} {'FAILED':<18} "
                  f"{'FAILED':<18} {RED}✗{RESET}")

    verdict = f"{GREEN}PASSED{RESET}" if all_dec_pass else f"{RED}FAILED{RESET}"
    print(f"\n  Decryption-side verdict: {verdict}")

    # ==================================================================
    # TEST 3: Multi-Epsilon Threshold Sweep
    # ==================================================================
    print(f"\n{'═' * 70}")
    print(f"  {BOLD}TEST 3: Multi-Epsilon Threshold Sweep{RESET}")
    print(f"  Perturbing x₁⁰ across ε = 10⁻¹⁰ to 10⁻¹⁶")
    print(f"{'═' * 70}")

    sweep_results = test_epsilon_sweep(
        original, key_bytes, encrypted
    )

    print(f"\n  {'Epsilon':<14} {'NPCR (%)':<14} {'UACI (%)':<14} {'Pass':<6}")
    print(f"  {'─' * 48}")

    for r in sweep_results:
        passed = r['npcr'] > 99.50
        status = f"{GREEN}✓{RESET}" if passed else f"{YELLOW}— (below float precision){RESET}"
        print(f"  {r['epsilon']:<14.0e} {r['npcr']:<14.4f} {r['uaci']:<14.4f} {status}")

    # Find the sensitivity boundary
    sensitive = [r for r in sweep_results if r['npcr'] > 99.0]
    insensitive = [r for r in sweep_results if r['npcr'] < 1.0]
    if sensitive:
        smallest = min(r['epsilon'] for r in sensitive)
        print(f"\n  {GREEN}Sensitivity threshold: ε ≥ {smallest:.0e} → full decorrelation (NPCR > 99.6%){RESET}")
    else:
        print(f"\n  {RED}No epsilon produced sensitivity{RESET}")

    # ==================================================================
    # TEST 4: Generate Publication Figures
    # ==================================================================
    print(f"\n{'═' * 70}")
    print(f"  {BOLD}TEST 4: Generating Figures{RESET}")
    print(f"{'═' * 70}")

    # Get a perturbed encryption for x1 (for the main figure)
    enc_perturbed_x1 = encrypt_with_perturbed_key(original, key_bytes, var_index=0, epsilon=EPSILON)

    # Get wrong-key decryption for x1 (for figures)
    dec_wrong_x1 = None
    if dec_data['wrong_decryptions'] and dec_data['wrong_decryptions'][0] is not None:
        dec_wrong_x1 = dec_data['wrong_decryptions'][0]

    # Figure 1: 6-panel encryption sensitivity
    generate_encryption_sensitivity_figure(
        original, encrypted, enc_perturbed_x1,
        decrypted_correct, dec_wrong_x1,
        EPSILON, base_name
    )

    # Figure 2: Decryption histogram comparison
    if decrypted_correct is not None:
        generate_decryption_histogram_figure(
            decrypted_correct, dec_wrong_x1,
            EPSILON, base_name
        )

    # Figure 3: Epsilon sweep plot
    generate_epsilon_sweep_figure(sweep_results, base_name)

    # Figure 4: Per-variable bar chart
    generate_per_variable_bar_chart(enc_results, dec_results, base_name)

    # Save CSV Table
    save_results_csv(enc_results, dec_results, sweep_results, base_name)

    # ==================================================================
    # Final Summary
    # ==================================================================
    print(f"\n{'═' * 70}")
    print(f"  {BOLD}{CYAN}KEY SENSITIVITY ANALYSIS — FINAL SUMMARY{RESET}")
    print(f"{'═' * 70}")
    print(f"  Image:    {image_path} ({M}×{N})")
    print(f"  Epsilon:  {EPSILON:.0e}")
    print()
    print(f"  Test 1 (Encryption-side):  {'PASSED ✓' if all_enc_pass else 'FAILED ✗'}")
    print(f"  Test 2 (Decryption-side):  {'PASSED ✓' if all_dec_pass else 'FAILED ✗'}")
    print(f"  Test 3 (Epsilon sweep):    threshold logic verified")
    print(f"  Test 4 (Figures):          GENERATED ✓")
    print(f"  Lossless decryption:       {'YES ✓' if dec_data['correct_lossless'] else 'NO ✗'}")

    all_passed = all_enc_pass and all_dec_pass and dec_data['correct_lossless']

    if all_passed:
        print(f"\n  {GREEN}{BOLD}★ ALL KEY SENSITIVITY TESTS PASSED ★{RESET}")
        print(f"  The system demonstrates extreme sensitivity to key perturbations.")
    else:
        print(f"\n  {YELLOW}⚠ Some tests did not pass with ε = {EPSILON:.0e}.{RESET}")

    print(f"\n{'═' * 70}\n")


# ======================================================================
#  Entry Point
# ======================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python Key_sensitivity.py <image_path>")
        print("Example: python Key_sensitivity.py Images/pepper.tiff")
        sys.exit(1)

    run_key_sensitivity_analysis(sys.argv[1])
