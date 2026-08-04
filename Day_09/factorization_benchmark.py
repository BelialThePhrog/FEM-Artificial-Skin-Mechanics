import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def benchmark_factorizations(max_n=1000, window=25):
    """
    Benchmarks LU Decomposition vs Cholesky Factorization on Symmetric 
    Positive-Definite (SPD) matrices. Generates a smoothed performance chart.
    """
    print(f"Running factorization benchmark up to N={max_n}... This might take a moment.")
    
    matrix_sizes = []
    time_lu = []
    time_chol = []
    
    for i in range(1, max_n):
        # 1. Generate a Symmetric Positive-Definite (SPD) matrix
        A = np.random.rand(i, i)
        A_spd = np.dot(A, A.T)
        
        # 2. Measure LU Decomposition (General purpose)
        start_time = time.perf_counter()
        P, L, U = la.lu(A_spd)
        time_lu.append(time.perf_counter() - start_time)
        
        # 3. Measure Cholesky Decomposition (Optimized for SPD)
        start_time = time.perf_counter()
        c = la.cholesky(A_spd)  # Corrected unpacking and assignment
        time_chol.append(time.perf_counter() - start_time)
        
        matrix_sizes.append(i)

    # --- VISUALIZATION SECTION ---
    print("Calculations finished. Generating performance plot...")
    
    # Calculating a moving average to smooth out CPU spikes
    y1_smooth = np.convolve(time_lu, np.ones(window)/window, mode='valid')
    y2_smooth = np.convolve(time_chol, np.ones(window)/window, mode='valid')
    x_smooth = matrix_sizes[(window-1):]

    # Creating a professional, larger figure
    plt.figure(figsize=(10, 6))

    # Plotting the raw, noisy data with high transparency (alpha=0.2)
    plt.plot(matrix_sizes, time_lu, color='#1f77b4', alpha=0.2, label='LU (Raw Data)')
    plt.plot(matrix_sizes, time_chol, color='#ff7f0e', alpha=0.2, label='Cholesky (Raw Data)')

    # Plotting the smoothed trends with thicker lines
    plt.plot(x_smooth, y1_smooth, color='#1f77b4', linewidth=2.5, label='LU Decomposition (Trend)')
    plt.plot(x_smooth, y2_smooth, color='#ff7f0e', linewidth=2.5, linestyle='--', label='Cholesky Factorization (Trend)')

    # Styling the plot
    plt.xlabel("Matrix Size (N x N)", fontweight='bold')
    plt.ylabel("Execution Time (Seconds)", fontweight='bold')
    plt.title("Performance Benchmark: Cholesky vs LU Solver (SPD Matrices)", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display the final polished plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    benchmark_factorizations(max_n=1000, window=25)
