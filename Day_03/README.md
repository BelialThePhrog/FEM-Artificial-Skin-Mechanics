# Day 03: Complete Compendium: Cauchy Stress Tensor & Strain Simulation

## 📖 Learning Objective
Master the mathematical formalism of Continuum Mechanics and implement numerical algorithms to compute 2D and 3D Cauchy strain tensors for artificial skin deformation.

## 🔬 Theoretical Background
* **Continuum Mechanics & Cauchy Tensor:** Understanding the transition from continuous physics to discrete matrix operations, distinguishing between normal (compression/tension) and shear stresses.
* **Hooke's Law & Failure Criteria:** Applying Lamé constants to determine stress from strain, and utilizing the Von Mises criterion to predict material yielding.
* **FEM Discretization:** Translating nodal displacements into a continuous strain tensor using shape functions, the Jacobian matrix, and discrete displacement gradients.
* **3D Thickness Effect:** Moving beyond plane stress/strain to compute the full 3D tensor, capturing the Z-axis thinning inherent to incompressible materials like hydrogels and elastomers (Poisson's effect $\nu \approx 0.5$).

## 🚀 Training & Exercises
* **`strain_tensor_2d.py`**: Computes the Cauchy strain tensor for a 4-node quadrilateral (Q4) element at its center of mass, visualizing the displacement vectors and the transition from the original to the deformed shape.
* **`strain_tensor_3d.py`**: Solves the 3D deformation for an 8-node hexahedral (H8) element. It specifically demonstrates the realistic behavior of synthetic skin, including transverse compression and Z-axis thinning under a 15% longitudinal stretch.
