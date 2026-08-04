
## 📖 Learning Objective
Establish a highly optimized numerical foundation for Phase 2 (FEM simulations of Artificial Skin). The focus is on exploiting the mathematical properties of physical systems to reduce computational complexity.

## 🔬 Methodology: LU vs. Cholesky Decomposition
In biomechanical simulations, stiffness matrices are fundamentally **Symmetric Positive-Definite (SPD)**. This structural property allows us to bypass general-purpose solvers and utilize specialized factorization methods:
*   **SPD Matrix Generation:** For testing purposes, valid SPD matrices are constructed using the dot product of a random matrix and its transpose ($A \cdot A^T$)[cite: 4].
*   **LU Decomposition (`scipy.linalg.lu`):** A general-purpose algorithm that factors a matrix into a lower triangular matrix ($L$) and an upper triangular matrix ($U$)[cite: 4]. Time complexity is approximately $O(2N^3/3)$.
*   **Cholesky Decomposition (`scipy.linalg.cholesky`):** A highly specialized algorithm exclusively for SPD matrices[cite: 4]. It factors the matrix into $L \cdot L^T$. Time complexity is exactly half of LU: $O(N^3/3)$.

## 🚀 Benchmark Implementation
The provided benchmarking script evaluates the execution time of both methods across matrices up to $1000 \times 1000$ elements[cite: 4]. The visualization proves that hardware-level optimization combined with algorithmic specialization (Cholesky) yields a ~2x performance gain, which is critical for scaling our future 3D tissue models.
