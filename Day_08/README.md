# Week 2: Numerical Problem Solving & SciPy Solvers

## 📖 Learning Objective
Achieve fluency in utilizing modern numerical solvers, specifically `scipy.linalg.solve`, for linear systems ($Ax=b$) instead of explicitly computing the inverse matrix $A^{-1}$.

## 🔬 Theoretical Background
* **The Problem with $A^{-1}$:** Explicitly calculating the inverse of an $N \times N$ matrix requires $O(N^3)$ operations, creating a severe performance bottleneck for larger datasets[cite: 12]. Furthermore, floating-point arithmetic introduces rounding errors which are heavily magnified when dealing with ill-conditioned matrices.
* **Hardware-Level Optimization:** Modern solvers delegate operations to LAPACK (Linear Algebra PACKage) and BLAS (Basic Linear Algebra Subprograms)[cite: 12]. These optimized C/Fortran routines solve systems directly using factorization methods (e.g., LU Decomposition) and ensure optimal memory cache management.

## 🚀 Execution & Benchmarking
* **`solvers_benchmark.py`**: A Python script testing the execution time of explicit inversion (`la.inv()`) versus the direct solver (`la.solve()`) across growing matrix dimensions (from $1 \times 1$ up to $2000 \times 2000$). 
* **Visualization:** The script generates a comparative performance chart[cite: 11]. It utilizes moving averages to smooth out CPU-induced spikes, visually proving the speed and stability of the direct linear solver approach.
