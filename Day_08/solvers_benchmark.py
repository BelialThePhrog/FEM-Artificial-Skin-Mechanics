import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def run_solver_benchmark(max_n=2000, window=25):
    """
    Benchmarks the performance of explicit matrix inversion vs. SciPy's direct solver.
    Generates a smoothed trend plot of execution times.
    """
    print(f"Running benchmark up to N={max_n}... This might take a moment.")
    
    matrix_sizes = []
    time_inv = []
    time_solve = []
    
    for i in range(1, max_n):
        # Generate random matrix A and vector b
        A = np.random.rand(i, i)
        b = np.random.rand(i)
        
        # Measure Inverse Matrix Method (O(N^3) bottleneck)
        start_time = time.perf_counter()
        A_inv = la.inv(A)
        x_bad = np.matmul(A_inv, b)
        time_inv.append(time.perf_counter() - start_time)
        
        # Measure Direct Solve Method (LAPACK/BLAS optimized)
        start_time = time.perf_counter()
        x_good = la.solve(A, b)
        time_solve.append(time.perf_counter() - start_time)
        
        matrix_sizes.append(i)

    # --- VISUALIZATION SECTION ---
    print("Calculations finished. Generating performance plot...")
    
    # Calculating a moving average to smooth out CPU spikes
    y1_smooth = np.convolve(time_inv, np.ones(window)/window, mode='valid')
    y2_smooth = np.convolve(time_solve, np.ones(window)/window, mode='valid')
    x_smooth = matrix_sizes[(window-1):]

    # Create a professional figure
    plt.figure(figsize=(10, 6))

    # Plotting the raw, noisy data with high transparency (alpha=0.2)
    plt.plot(matrix_sizes, time_inv, color='#1f77b4', alpha=0.2, label='inv() raw data')
    plt.plot(matrix_sizes, time_solve, color='#ff7f0e', alpha=0.2, label='solve() raw data')

    # Plotting the smoothed trends with thicker lines
    plt.plot(x_smooth, y1_smooth, color='#1f77b4', linewidth=2.5, label='Inverse Method (Trend)')
    plt.plot(x_smooth, y2_smooth, color='#ff7f0e', linewidth=2.5, linestyle='--', label='Solve Method (Trend)')

    # Styling the plot
    plt.xlabel("Matrix Size (N x N)", fontweight='bold')
    plt.ylabel("Execution Time (Seconds)", fontweight='bold')
    plt.title("Performance Benchmark: $A^{-1}$ vs Linear Solver", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Display the final polished plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_solver_benchmark(max_n=2000, window=25)
