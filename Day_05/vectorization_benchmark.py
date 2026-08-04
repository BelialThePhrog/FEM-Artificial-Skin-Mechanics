import numpy as np
import time
import matplotlib.pyplot as plt

def run_performance_benchmark():
    """
    Executes a benchmark comparing traditional Python loops against 
    NumPy vectorization for tensor operations.
    """
    # ==========================================
    # 1. DATA PREPARATION
    # ==========================================
    N = 100_000_000
    print(f"Generating data: vector of {N} elements...\n")
    
    # Using np.random.rand for optimized large-scale generation
    data_vector = np.random.rand(N)
    # Converting to a standard Python list to fairly simulate pure Python loop overhead
    data_list = data_vector.tolist()
    
    # ==========================================
    # 2. CALCULATIONS AND TIME MEASUREMENTS
    # ==========================================
    
    # Method 1: Traditional FOR loop (Python developer's nightmare)
    start_time = time.perf_counter()
    sum_loop = 0
    for x in data_list:
        sum_loop += x**2
    time_loop = time.perf_counter() - start_time
    
    # Method 2: NumPy Vectorization - exponentiation and sum (Deep Tech Standard)
    start_time = time.perf_counter()
    sum_vector = np.sum(data_vector**2)
    time_vector = time.perf_counter() - start_time
    
    # Method 3: NumPy Vectorization - pure dot product (Hardware-optimized)
    start_time = time.perf_counter()
    sum_dot = np.dot(data_vector, data_vector)
    time_dot = time.perf_counter() - start_time
    
    # ==========================================
    # 3. CONSOLE REPORT
    # ==========================================
    print("--- CALCULATION RESULTS ---")
    print(f"1. FOR Loop:      Sum = {sum_loop:.2f} | Time = {time_loop:.5f} s")
    print(f"2. NumPy sum():   Sum = {sum_vector:.2f} | Time = {time_vector:.5f} s")
    print(f"3. NumPy dot():   Sum = {sum_dot:.2f} | Time = {time_dot:.5f} s\n")
    
    print("--- SPEEDUP ---")
    print(f"np.sum(x**2) is {time_loop / time_vector:.1f}x faster than the loop.")
    print(f"np.dot(x, x) is {time_loop / time_dot:.1f}x faster than the loop!\n")
    
    # ==========================================
    # 4. MATPLOTLIB VISUALIZATION
    # ==========================================
    methods = ['FOR Loop\n(Performance nightmare)', 'NumPy\nnp.sum(x**2)', 'NumPy\nnp.dot(x, x)']
    times = [time_loop, time_vector, time_dot]
    colors = ['#d9534f', '#5bc0de', '#0275d8'] 
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, times, color=colors, edgecolor='black', linewidth=1.2)
    
    # Overlaying exact numerical values above the bars
    for bar in bars:
        yval = bar.get_height()
        # Adding a slight Y-axis offset for the logarithmic scale readability
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.2, 
                 f'{yval:.5f} s', ha='center', va='bottom', 
                 fontsize=11, fontweight='bold', color='black')
    
    # Chart styling
    plt.title('Tensor operation performance: FOR Loop vs. NumPy Vectorization\n(Sum of squares of 100,000,000 elements)', 
              fontsize=14, fontweight='bold', pad=20, color='#1c2833')
    plt.ylabel('Execution time [seconds] (Logarithmic Scale)', fontsize=12, fontweight='bold')
    
    # Key element: logarithmic scale to capture massive performance gaps
    plt.yscale('log') 
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Footer explaining the scale
    plt.annotate('A logarithmic Y-axis scale was used due to the drastic difference in orders of magnitude.', 
                 xy=(0.5, -0.15), xycoords='axes fraction', 
                 ha='center', fontsize=10, color='#555555', style='italic')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_performance_benchmark()
