"""
NIST SP 800-22 Full Statistical Test Suite for Ciphertext Randomness.

Implements all 15 tests specified in NIST SP 800-22:
    1. Frequency (Monobit)
    2. Block Frequency
    3. Cumulative Sums (Cusum)
    4. Runs
    5. Longest Runs of Ones
    6. Rank (Binary Matrix Rank)
    7. Spectral DFT
    8. Nonperiodic Template Matchings
    9. Overlapping Template Matchings
    10. Universal Statistical (Maurer's)
    11. Approximate Entropy
    12. Random Excursions
    13. Random Excursions Variant
    14. Serial
    15. Linear Complexity

Usage:
    python attacks/NIST.py path/to/cipher_image.png
If no path is given, it automatically searches for ciphertext files in the results folder,
or generates one on-the-fly using our cellular automata encryption algorithm.
"""

import sys
import os
import math
import numpy as np
from scipy import special, fftpack
import csv
import json

# Initialize virtual terminal processing for Windows
if sys.platform == 'win32':
    os.system('')

# Ensure workspace packages can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import src.cellular_automata as ca
    from src.utils import generate_synthetic_image
    CA_AVAILABLE = True
except ImportError:
    CA_AVAILABLE = False


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert a byte string to a numpy array of 0/1 ints, MSB first per byte."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.int64)


# ----------------------------------------------------------------------
# 1. Frequency (Monobit) Test
# ----------------------------------------------------------------------
def frequency_monobit_test(bits):
    n = len(bits)
    s = np.sum(2 * bits - 1)
    s_obs = abs(s) / math.sqrt(n)
    p_value = special.erfc(s_obs / math.sqrt(2))
    return p_value


# ----------------------------------------------------------------------
# 2. Block Frequency Test
# ----------------------------------------------------------------------
def block_frequency_test(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return None
    bits_sliced = bits[:n_blocks * block_size].reshape(n_blocks, block_size)
    pi_i = np.sum(bits_sliced, axis=1) / block_size
    chi_sq = 4 * block_size * np.sum((pi_i - 0.5) ** 2)
    p_value = special.gammaincc(n_blocks / 2.0, chi_sq / 2.0)
    return p_value


# ----------------------------------------------------------------------
# 3. Cumulative Sums (Cusum) Test
# ----------------------------------------------------------------------
def cumulative_sums_test(bits, mode='forward'):
    n = len(bits)
    x = 2 * bits - 1
    if mode == 'reverse':
        x = x[::-1]
    s = np.cumsum(x)
    z = int(np.max(np.abs(s)))
    if z == 0:
        return 0.0

    def norm_cdf(v):
        return 0.5 * (1 + math.erf(v / math.sqrt(2)))

    total = 0.0
    start1 = int((-n / z + 1) / 4)
    end1 = int((n / z - 1) / 4)
    for k in range(start1, end1 + 1):
        total += norm_cdf((4 * k + 1) * z / math.sqrt(n)) - norm_cdf((4 * k - 1) * z / math.sqrt(n))

    start2 = int((-n / z - 3) / 4)
    end2 = int((n / z - 1) / 4)
    total2 = 0.0
    for k in range(start2, end2 + 1):
        total2 += norm_cdf((4 * k + 3) * z / math.sqrt(n)) - norm_cdf((4 * k + 1) * z / math.sqrt(n))

    p_value = 1.0 - total + total2
    return max(0.0, min(1.0, p_value))


# ----------------------------------------------------------------------
# 4. Runs Test
# ----------------------------------------------------------------------
def runs_test(bits):
    n = len(bits)
    pi = np.sum(bits) / n
    if abs(pi - 0.5) >= (2.0 / math.sqrt(n)):
        return 0.0
    v_obs = 1 + np.sum(bits[1:] != bits[:-1])
    num = abs(v_obs - 2 * n * pi * (1 - pi))
    den = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    p_value = special.erfc(num / den)
    return p_value


# ----------------------------------------------------------------------
# 5. Longest Run of Ones in a Block Test
# ----------------------------------------------------------------------
def longest_run_ones_test(bits):
    n = len(bits)
    if n < 128:
        return None
    if n < 6272:
        M, K, N = 8, 3, n // 8
        pi_vals = [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        M, K, N = 128, 5, n // 128
        pi_vals = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        M, K, N = 10000, 6, n // 10000
        pi_vals = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    if N == 0:
        return None

    blocks = bits[:N * M].reshape(N, M)
    v_counts = np.zeros(len(pi_vals), dtype=int)

    for row in blocks:
        max_run = 0
        cur = 0
        for b in row:
            if b == 1:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 0
        if M == 8:
            if max_run <= 1: v_counts[0] += 1
            elif max_run == 2: v_counts[1] += 1
            elif max_run == 3: v_counts[2] += 1
            else: v_counts[3] += 1
        elif M == 128:
            if max_run <= 4: v_counts[0] += 1
            elif max_run == 5: v_counts[1] += 1
            elif max_run == 6: v_counts[2] += 1
            elif max_run == 7: v_counts[3] += 1
            elif max_run == 8: v_counts[4] += 1
            else: v_counts[5] += 1
        else:
            if max_run <= 10: v_counts[0] += 1
            elif max_run == 11: v_counts[1] += 1
            elif max_run == 12: v_counts[2] += 1
            elif max_run == 13: v_counts[3] += 1
            elif max_run == 14: v_counts[4] += 1
            elif max_run == 15: v_counts[5] += 1
            else: v_counts[6] += 1

    chi_sq = np.sum((v_counts - N * np.array(pi_vals)) ** 2 / (N * np.array(pi_vals)))
    p_value = special.gammaincc(K / 2.0, chi_sq / 2.0)
    return p_value


# ----------------------------------------------------------------------
# 6. Binary Matrix Rank Test
# ----------------------------------------------------------------------
def gf2_rank(matrix):
    """Compute rank of a binary matrix over GF(2) using Gaussian elimination."""
    m, n = matrix.shape
    rank = 0
    mat = matrix.copy()
    for col in range(n):
        pivot = -1
        for row in range(rank, mat.shape[0]):
            if mat[row, col] == 1:
                pivot = row
                break
        if pivot != -1:
            if pivot != rank:
                mat[[rank, pivot]] = mat[[pivot, rank]]
            for row in range(rank + 1, m):
                if mat[row, col] == 1:
                    mat[row] ^= mat[rank]
            rank += 1
            if rank == m:
                break
    return rank


def binary_matrix_rank_test(bits, M=32, Q=32):
    n = len(bits)
    block_size = M * Q
    n_blocks = n // block_size
    if n_blocks == 0:
        return None
    
    ranks = np.zeros(n_blocks, dtype=int)
    for i in range(n_blocks):
        block = bits[i * block_size : (i + 1) * block_size].reshape(M, Q)
        ranks[i] = gf2_rank(block)
        
    F_m = np.sum(ranks == M)
    F_m1 = np.sum(ranks == (M-1))
    F_other = n_blocks - F_m - F_m1
    
    expected_m = 0.2887880552 * n_blocks
    expected_m1 = 0.5735703882 * n_blocks
    expected_other = 0.1376415566 * n_blocks
    
    if expected_m == 0 or expected_m1 == 0 or expected_other == 0:
        return None
        
    chi_sq = ((F_m - expected_m) ** 2 / expected_m +
              (F_m1 - expected_m1) ** 2 / expected_m1 +
              (F_other - expected_other) ** 2 / expected_other)
              
    p_value = special.gammaincc(1, chi_sq / 2.0)
    return p_value


# ----------------------------------------------------------------------
# 7. Discrete Fourier Transform (Spectral) Test
# ----------------------------------------------------------------------
def dft_spectral_test(bits):
    n = len(bits)
    x = 2 * bits - 1
    s = fftpack.fft(x)
    modulus = np.abs(s[:n // 2])
    t = math.sqrt(math.log(1.0 / 0.05) * n)
    n0 = 0.95 * n / 2.0
    n1 = np.sum(modulus < t)
    d = (n1 - n0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    p_value = special.erfc(abs(d) / math.sqrt(2))
    return p_value


# ----------------------------------------------------------------------
# 8. Non-overlapping Template Matching Test
# ----------------------------------------------------------------------
def non_overlapping_template_matching_test(bits, template="000000001", block_size=1032):
    n = len(bits)
    m = len(template)
    n_blocks = n // block_size
    if n_blocks == 0:
        return None
        
    template_bits = np.array([int(c) for c in template], dtype=np.int64)
    counts = np.zeros(n_blocks)
    
    for i in range(n_blocks):
        block = bits[i * block_size : (i + 1) * block_size]
        j = 0
        w = 0
        while j <= block_size - m:
            if np.array_equal(block[j : j + m], template_bits):
                w += 1
                j += m
            else:
                j += 1
        counts[i] = w
        
    mu = (block_size - m + 1) / (2**m)
    var = block_size * ((1 / (2**m)) - (2 * m - 1) / (2**(2*m)))
    
    if var == 0:
        return None
        
    chi_sq = np.sum((counts - mu) ** 2 / var)
    p_value = special.gammaincc(n_blocks / 2.0, chi_sq / 2.0)
    return p_value


# ----------------------------------------------------------------------
# 9. Overlapping Template Matching Test
# ----------------------------------------------------------------------
def overlapping_template_matching_test(bits, m=9, block_size=1032):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return None
        
    template_bits = np.ones(m, dtype=np.int64)
    counts = np.zeros(n_blocks)
    
    pi = [0.36487, 0.185659, 0.139381, 0.100571, 0.070908, 0.138611]
    
    for i in range(n_blocks):
        block = bits[i * block_size : (i + 1) * block_size]
        w = 0
        for j in range(block_size - m + 1):
            if np.array_equal(block[j : j + m], template_bits):
                w += 1
        counts[i] = w
        
    v = np.zeros(6)
    for c in counts:
        if c == 0: v[0] += 1
        elif c == 1: v[1] += 1
        elif c == 2: v[2] += 1
        elif c == 3: v[3] += 1
        elif c == 4: v[4] += 1
        else: v[5] += 1
        
    chi_sq = 0.0
    for i in range(6):
        expected = n_blocks * pi[i]
        if expected == 0:
            return None
        chi_sq += (v[i] - expected) ** 2 / expected
        
    p_value = special.gammaincc(2.5, chi_sq / 2.0)
    return p_value


# ----------------------------------------------------------------------
# 10. Universal Statistical Test (Maurer's)
# ----------------------------------------------------------------------
def maurers_universal_test(bits, L=7, Q=1280):
    n = len(bits)
    K = (n // L) - Q
    if K <= 0:
        return None
        
    if L == 6:
        expected, var = 5.217705, 2.954
    elif L == 7:
        expected, var = 6.1962507, 3.125
    elif L == 8:
        expected, var = 7.1836656, 3.238
    else:
        expected = L - 0.7
        var = 3.0
        
    table = {}
    for i in range(Q):
        block = tuple(bits[i * L : (i + 1) * L])
        table[block] = i + 1
        
    sum_log = 0.0
    for i in range(Q, Q + K):
        block = tuple(bits[i * L : (i + 1) * L])
        step = i + 1
        if block in table:
            dist = step - table[block]
        else:
            dist = step
        table[block] = step
        sum_log += math.log2(dist)
        
    fn = sum_log / K
    c = 0.7 - 0.8 / L + (4.0 + 32.0 / L) * (K ** (-3.0 / L)) / 15.0
    sigma = c * math.sqrt(var / K)
    
    if sigma == 0:
        return None
        
    p_value = special.erfc(abs(fn - expected) / (math.sqrt(2) * sigma))
    return p_value


# ----------------------------------------------------------------------
# 11. Approximate Entropy Test
# ----------------------------------------------------------------------
def approximate_entropy_test(bits, m=2):
    n = len(bits)

    def phi(m_):
        if m_ == 0:
            return 0.0
        padded = np.concatenate([bits, bits[:m_ - 1]])
        counts = {}
        for i in range(n):
            pattern = tuple(padded[i:i + m_])
            counts[pattern] = counts.get(pattern, 0) + 1
        freqs = np.array(list(counts.values())) / n
        return np.sum(freqs * np.log(freqs))

    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    ap_en = phi_m - phi_m1
    chi_sq = 2 * n * (math.log(2) - ap_en)
    p_value = special.gammaincc(2 ** (m - 1), chi_sq / 2.0)
    return p_value


# ----------------------------------------------------------------------
# 12. Random Excursions Test
# ----------------------------------------------------------------------
def random_excursions_test(bits):
    n = len(bits)
    x = 2 * bits - 1
    s = np.concatenate([[0], np.cumsum(x), [0]])
    
    zero_indices = np.where(s == 0)[0]
    J = len(zero_indices) - 1
    
    if J < 10:
        return [None] * 8
        
    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    p_values = []
    
    pi = {
        1: [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0312],
        2: [0.7500, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791],
        3: [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0803],
        4: [0.8750, 0.0156, 0.0137, 0.0120, 0.0105, 0.0732]
    }
    
    visit_counts = {state: np.zeros(6) for state in states}
    
    for i in range(J):
        cycle = s[zero_indices[i] + 1 : zero_indices[i + 1]]
        for state in states:
            c = np.sum(cycle == state)
            if c == 0: visit_counts[state][0] += 1
            elif c == 1: visit_counts[state][1] += 1
            elif c == 2: visit_counts[state][2] += 1
            elif c == 3: visit_counts[state][3] += 1
            elif c == 4: visit_counts[state][4] += 1
            else: visit_counts[state][5] += 1
            
    for state in states:
        abs_state = abs(state)
        p_vals = pi[abs_state]
        chi_sq = 0.0
        for k in range(6):
            expected = J * p_vals[k]
            chi_sq += (visit_counts[state][k] - expected) ** 2 / expected
        p_val = special.gammaincc(2.5, chi_sq / 2.0)
        p_values.append(p_val)
        
    return p_values


# ----------------------------------------------------------------------
# 13. Random Excursions Variant Test
# ----------------------------------------------------------------------
def random_excursions_variant_test(bits):
    n = len(bits)
    x = 2 * bits - 1
    s = np.concatenate([[0], np.cumsum(x), [0]])
    
    zero_indices = np.where(s == 0)[0]
    J = len(zero_indices) - 1
    
    if J < 10:
        return [None] * 18
        
    states = list(range(-9, 0)) + list(range(1, 10))
    p_values = []
    
    for state in states:
        xi = np.sum(s[1:-1] == state)
        num = abs(xi - J)
        den = math.sqrt(2 * J * (4 * abs(state) - 2))
        p_val = special.erfc(num / den)
        p_values.append(p_val)
        
    return p_values


# ----------------------------------------------------------------------
# 14. Serial Test
# ----------------------------------------------------------------------
def serial_test(bits, m=2):
    n = len(bits)

    def psi_sq(m_):
        if m_ <= 0:
            return 0.0
        padded = np.concatenate([bits, bits[:m_ - 1]])
        counts = {}
        for i in range(n):
            pattern = tuple(padded[i:i + m_])
            counts[pattern] = counts.get(pattern, 0) + 1
        vals = np.array(list(counts.values()))
        return (2 ** m_ / n) * np.sum(vals ** 2) - n

    psi_m = psi_sq(m)
    psi_m1 = psi_sq(m - 1)
    psi_m2 = psi_sq(m - 2)
    delta1 = psi_m - psi_m1
    delta2 = psi_m - 2 * psi_m1 + psi_m2

    p1 = special.gammaincc(2 ** (m - 2), delta1 / 2.0) if m >= 1 else None
    p2 = special.gammaincc(2 ** (m - 3), delta2 / 2.0) if m >= 2 else None
    return p1, p2


# ----------------------------------------------------------------------
# 15. Linear Complexity Test
# ----------------------------------------------------------------------
def berlekamp_massey(bits):
    N = len(bits)
    # Pad bits with N zeros at the beginning to handle bits[i-j] for i-j < 0 correctly
    padded = np.zeros(2 * N, dtype=np.int8)
    padded[N:] = bits
    
    b = np.zeros(N, dtype=np.int8)
    c = np.zeros(N, dtype=np.int8)
    b[0] = 1
    c[0] = 1
    L = 0
    m = -1
    
    for i in range(N):
        # i in bits corresponds to i + N in padded
        idx = i + N
        if L == 0:
            d = bits[i]
        else:
            d = (bits[i] ^ np.dot(c[1:L+1], padded[idx - 1 : idx - L - 1 : -1])) % 2
            
        if d != 0:
            t = c.copy()
            shift = i - m
            c[shift:N] ^= b[:N-shift]
            if 2 * L <= i:
                L = i + 1 - L
                m = i
                b = t
    return L


def linear_complexity_test(bits, M=500):
    n = len(bits)
    n_blocks = n // M
    if n_blocks == 0:
        return None
        
    pi = [0.01047, 0.03129, 0.12560, 0.50077, 0.25300, 0.05584, 0.02303]
    v = np.zeros(7)
    
    mu = M / 2.0 + (9.0 + (-1)**(M + 1)) / 36.0 - (M / 3.0 + 2.0 / 9.0) / (2**M)
    
    for i in range(n_blocks):
        block = bits[i * M : (i + 1) * M]
        L = berlekamp_massey(block)
        T = ((-1)**M) * (L - mu) + 2.0/9.0
        
        if T <= -2.5: v[0] += 1
        elif T <= -1.5: v[1] += 1
        elif T <= -0.5: v[2] += 1
        elif T <= 0.5: v[3] += 1
        elif T <= 1.5: v[4] += 1
        elif T <= 2.5: v[5] += 1
        else: v[6] += 1
        
    chi_sq = 0.0
    for i in range(7):
        expected = n_blocks * pi[i]
        if expected == 0:
            return None
        chi_sq += (v[i] - expected) ** 2 / expected
        
    p_value = special.gammaincc(3.0, chi_sq / 2.0)
    return p_value


# ----------------------------------------------------------------------
# Run Suite and Format Outputs
# ----------------------------------------------------------------------
ALPHA = 0.01

TEST_CATEGORIES = [
    "Frequency",
    "Block Frequency",
    "Cumulative Sums",
    "Runs",
    "Longest Runs of Ones",
    "Rank",
    "Spectral DFT",
    "Nonperiodic Template Matchings",
    "Overlapping Template Matchings",
    "Universal Statistical",
    "Approximate Entropy",
    "Random Excursions",
    "Random Excursions Variant",
    "Serial",
    "Linear Complexity",
]


def run_all_tests(bits, label="sequence"):
    results = {}
    
    # 1. Frequency
    results["Frequency"] = frequency_monobit_test(bits)
    
    # 2. Block Frequency
    results["Block Frequency"] = block_frequency_test(bits)
    
    # 3. Cumulative Sums
    p_cusum_f = cumulative_sums_test(bits, 'forward')
    p_cusum_r = cumulative_sums_test(bits, 'reverse')
    results["Cumulative Sums"] = min(p_cusum_f, p_cusum_r)
    
    # 4. Runs
    results["Runs"] = runs_test(bits)
    
    # 5. Longest Runs of Ones
    results["Longest Runs of Ones"] = longest_run_ones_test(bits)
    
    # 6. Rank
    results["Rank"] = binary_matrix_rank_test(bits)
    
    # 7. Spectral DFT
    results["Spectral DFT"] = dft_spectral_test(bits)
    
    # 8. Nonperiodic Template Matchings
    results["Nonperiodic Template Matchings"] = non_overlapping_template_matching_test(bits)
    
    # 9. Overlapping Template Matchings
    results["Overlapping Template Matchings"] = overlapping_template_matching_test(bits)
    
    # 10. Universal Statistical
    results["Universal Statistical"] = maurers_universal_test(bits)
    
    # 11. Approximate Entropy
    results["Approximate Entropy"] = approximate_entropy_test(bits)
    
    # 12. Random Excursions
    re_pvals = random_excursions_test(bits)
    valid_re = [p for p in re_pvals if p is not None]
    results["Random Excursions"] = min(valid_re) if len(valid_re) > 0 else None
    
    # 13. Random Excursions Variant
    rev_pvals = random_excursions_variant_test(bits)
    valid_rev = [p for p in rev_pvals if p is not None]
    results["Random Excursions Variant"] = min(valid_rev) if len(valid_rev) > 0 else None
    
    # 14. Serial
    p1, p2 = serial_test(bits)
    if p1 is not None and p2 is not None:
        results["Serial"] = min(p1, p2)
    else:
        results["Serial"] = None
        
    # 15. Linear Complexity
    results["Linear Complexity"] = linear_complexity_test(bits)

    print(f"\n{'='*75}\n  NIST SP 800-22 FULL STATISTICAL TEST SUITE: {label}\n{'='*75}")
    print(f"  Sequence length: {len(bits)} bits ({len(bits)//8} bytes)")
    print(f"  {'Test':<36}{'p-value':<16}{'Result'}")
    print("  " + "-" * 67)
    
    n_pass = 0
    n_total = 0
    
    for name in TEST_CATEGORIES:
        p = results.get(name)
        if p is None:
            print(f"  {name:<36}{'N/A (seq too short)':<16}\033[93mWARNING\033[0m")
            continue
            
        n_total += 1
        passed = p >= ALPHA
        n_pass += int(passed)
        
        # Try printing with unicode checkmark/cross, fallback if encoding error
        try:
            if passed:
                status = f"\033[1;32mPASS [\u2713]\033[0m"
            else:
                status = f"\033[1;31mFAIL [\u2717]\033[0m"
            print(f"  {name:<36}{p:<16.6f}{status}")
        except UnicodeEncodeError:
            if passed:
                status = "\033[1;32mPASS [OK]\033[0m"
            else:
                status = "\033[1;31mFAIL [X]\033[0m"
            print(f"  {name:<36}{p:<16.6f}{status}")
        
    print("  " + "-" * 67)
    print(f"  Passed {n_pass}/{n_total} tests at significance level alpha = {ALPHA}")
    print(f"{'='*75}\n")
    
    return results, n_pass, n_total


def load_bits_from_path(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.png', '.tiff', '.tif', '.jpg', '.jpeg', '.bmp'):
        from PIL import Image
        img = np.array(Image.open(path))
        data = img.tobytes()
    else:
        with open(path, 'rb') as f:
            data = f.read()
    return bytes_to_bits(data)


def save_tables(results, label):
    os.makedirs("results/tables", exist_ok=True)
    
    # Save CSV
    out_csv = "results/tables/nist.csv"
    try:
        file_exists = os.path.exists(out_csv)
        with open(out_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Sequence", "Test", "p-value", "Pass (alpha=0.01)"])
            for name in TEST_CATEGORIES:
                p = results.get(name)
                if p is None:
                    continue
                writer.writerow([label, name, f"{p:.6f}", "YES" if p >= ALPHA else "NO"])
    except PermissionError:
        print(f"\n[WARNING] Permission denied: Could not write to {out_csv}.")
        
    # Save JSON
    out_json = "results/tables/nist.json"
    try:
        json_data = []
        if os.path.exists(out_json):
            try:
                with open(out_json, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except Exception:
                json_data = []
                
        # Append current sequence results
        seq_entry = {"sequence": label, "tests": {}}
        for name in TEST_CATEGORIES:
            p = results.get(name)
            if p is not None:
                seq_entry["tests"][name] = {
                    "p_value": float(p),
                    "pass": bool(p >= ALPHA)
                }
        json_data.append(seq_entry)
        
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
    except PermissionError:
        print(f"\n[WARNING] Permission denied: Could not write to {out_json}.")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Loading bit sequence from: {path}")
        bits = load_bits_from_path(path)
        results, n_pass, n_total = run_all_tests(bits, label=os.path.basename(path))
        save_tables(results, os.path.basename(path))
        print("Results successfully saved in folder results/tables with files nist.csv and nist.json")
    else:
        # Search for generated ciphertext files
        candidates = []
        enc_dir = "results/encrypted"
        if os.path.exists(enc_dir):
            for f in os.listdir(enc_dir):
                if f.startswith("enc_") and f.endswith(".png"):
                    candidates.append(os.path.join(enc_dir, f))
        if os.path.exists("ciphertext.png"):
            candidates.append("ciphertext.png")
            
        if candidates:
            path = candidates[0]
            print(f"No file path specified. Automatically testing generated ciphertext: {path}")
            bits = load_bits_from_path(path)
            results, n_pass, n_total = run_all_tests(bits, label=os.path.basename(path))
            save_tables(results, os.path.basename(path))
            print("Results successfully saved in folder results/tables with files nist.csv and nist.json")
        elif CA_AVAILABLE:
            print("No generated ciphertext found. Running encryption on-the-fly to validate our cryptosystem...")
            key = b"IEEE_CA_Secure_Key_256_Bits_2026"
            # Generate a test image and encrypt
            img = generate_synthetic_image()
            cipher = ca.encrypt_image(img, key)
            bits = bytes_to_bits(cipher.tobytes())
            results, n_pass, n_total = run_all_tests(bits, label="cellular_automata_ciphertext")
            save_tables(results, "cellular_automata_ciphertext")
            print("Results successfully saved in folder results/tables with files nist.csv and nist.json")
        else:
            print("NIST self-test validation:")
            rand_bits = bytes_to_bits(os.urandom(128000))
            results, n_pass, n_total = run_all_tests(rand_bits, label="os.urandom()")
            save_tables(results, "os_urandom")
            print("Results successfully saved in folder results/tables with files nist.csv and nist.json")


if __name__ == "__main__":
    main()
