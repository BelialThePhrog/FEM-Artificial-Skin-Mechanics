# Day 01: Engineering Linear Algebra

## 📖 Learning Objective
Understand the process of LU factorization using elementary matrices E and optimize the use of space on the worksheet.

## 🔬 Theoretical Background
* **Gaussian Elimination:** Applying elementary row operations to transform matrix A into an upper triangular matrix U[cite: 3].
* **Elementary Matrices:** Each elimination step can be written as multiplication by an elementary matrix, which is an identity matrix containing the multiplier used to zero out a specific element.
* **Matrix L Construction:** The lower triangular matrix L is constructed by multiplying the inverses of the elementary matrices (e.g., L = E1^-1 E2^-1 ... Ek^-1).
* **Inverse Matrices:** The inverse of an elementary matrix is created by simply changing the sign of its multiplier.

## 🚀 Training & Exercises
* Solved multiple Ax=b systems across different dimensions (2x2, 3x3, and 4x4).
* The problem-solving scheme for each system included: determining E, L, and U matrices, solving the Ly=b equation, and finally solving Ux=y.
