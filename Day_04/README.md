# Day 04: Computational Overhead & NumPy Broadcasting

## 📖 Learning Objective
Master the dimension matching mechanism (broadcasting) in NumPy to shift computational heavy lifting to C-optimized memory blocks, drastically improving performance over standard Python `for` loops.

## 🔬 Theoretical Background
* **Computational Overhead:** Pure Python `for` loops are extremely slow due to dynamic typing overhead, lack of cache contiguity (reading addresses from RAM on every iteration), and the cost of repeatedly calling Python's universal addition functions.
* **C-Optimized Memory:** NumPy circumvents these limitations by operating on contiguous C-array memory layouts and pushing loops down to optimized processor instructions (SIMD / C-API).
* **Broadcasting Rules:** NumPy evaluates array dimensions from right to left (starting from the innermost axis). Dimensions are deemed compatible if they are equal, or if one of them is 1 (in which case the dimension is virtually expanded to match the other).

## 🚀 Training & Exercises
* **Matrix Operations:** Loopless addition of a 1D vector to a 2D matrix.
* **Advanced Broadcasting:** Manipulating axes using `np.newaxis` for column-wise operations and generating 2D mathematical grids (Outer Operations).
* **Practical Data Science Applications:** 
  * Normalizing 3D image arrays (RGB channel adjustments).
  * Calculating Euclidean distance matrices between multiple sets of coordinates.
  * Applying feature weights across multidimensional arrays.
  * Standardizing large datasets via mean centering and standard deviation scaling.
