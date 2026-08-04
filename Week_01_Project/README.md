# Week 1 Capstone Project: 3D Transformation Generator

## 📖 Learning Objective
Consolidate linear algebra fundamentals and NumPy vectorization techniques by building an optimized 3D transformation engine[cite: 1, 10]. The goal is to prove computational proficiency by manipulating a spatial cloud of 1,000,000 points in milliseconds without memory overflow.

## 🔬 Theoretical Background
* **Homogeneous Coordinates:** A standard in computer graphics and robotics. Extending 3D vectors $(x, y, z)$ to 4D $(x, y, z, 1)$ allows us to combine rotation, scaling, and translation into a single $4 \times 4$ matrix multiplication.
* **Affine Transformations:** Utilizing trigonometric functions $(\cos(\theta), \sin(\theta))$ to rotate points around a specified axis, while simultaneously applying spatial translations $(t_x, t_y, t_z)$.
* **Vectorized Matrix Multiplication:** Operating on matrices of shape $(N, 4)$. Using the `@` operator (dot product) to apply the transformation to all $10^6$ points concurrently, eliminating the need for iteration.

## 🚀 Execution & Benchmarking
* **`transform_3d.py`**: An object-oriented, hardware-optimized Python script that generates a random cloud of 1,000,000 points.
* **Performance:** The vectorized matrix dot product resolves the transformation in approximately `0.004 - 0.015` seconds (hardware dependent), proving the superiority of C-backend NumPy operations over native Python loops.
* **Visualization:** Includes a Matplotlib 3D scatter plot of a downsampled dataset to visually verify the mathematical correctness of the translation and rotation.
