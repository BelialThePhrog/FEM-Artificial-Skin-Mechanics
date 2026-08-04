import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_homogeneous_points(n_points):
    """
    Generates a cloud of random 3D points and formats them 
    as homogeneous coordinates (N, 4).
    """
    print(f"Generating data: vector of {n_points} elements...")
    # Generate random points in 3D space [x, y, z]
    points_3d = np.random.rand(n_points, 3) * 10
    
    # Create a column of ones for the homogeneous coordinate
    ones_column = np.ones((n_points, 1))
    
    # Concatenate to get (N, 4) matrix -> [x, y, z, 1]
    homogeneous_points = np.hstack((points_3d, ones_column))
    return homogeneous_points

def build_transformation_matrix(tx, ty, tz, theta_z_deg):
    """
    Constructs a 4x4 affine transformation matrix.
    Combines rotation around the Z-axis and 3D translation.
    """
    # Convert degrees to radians
    theta = np.radians(theta_z_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # 4x4 Homogeneous Transformation Matrix
    T_matrix = np.array([
        [cos_t, -sin_t,  0,  tx],
        [sin_t,  cos_t,  0,  ty],
        [    0,      0,  1,  tz],
        [    0,      0,  0,   1]
    ])
    return T_matrix

def visualize_transformation(original, transformed, sample_size=500):
    """
    Generates a 3D plot comparing the original and transformed point clouds.
    Downsamples the dataset to avoid rendering engine lag.
    """
    print("Generating 3D plots...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Take a random subset of points for visualization
    idx = np.random.choice(original.shape[0], sample_size, replace=False)
    orig_sample = original[idx]
    trans_sample = transformed[idx]
    
    # Plot original points (Blue)
    ax.scatter(orig_sample[:, 0], orig_sample[:, 1], orig_sample[:, 2], 
               c='blue', marker='o', alpha=0.5, label='Original Space')
    
    # Plot transformed points (Red)
    ax.scatter(trans_sample[:, 0], trans_sample[:, 1], trans_sample[:, 2], 
               c='red', marker='^', alpha=0.6, label='Transformed Space')
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(f'3D Affine Transformation\n(Showing {sample_size} sample points)')
    ax.legend()
    plt.show()

def main():
    # --- 1. Parameters ---
    N_POINTS = 1_000_000
    TRANSLATION = (1.0, 2.0, 3.0) # tx, ty, tz
    ROTATION_Z = 45.0             # degrees
    
    # --- 2. Data Generation ---
    points = generate_homogeneous_points(N_POINTS)
    
    # --- 3. Matrix Construction ---
    T = build_transformation_matrix(*TRANSLATION, ROTATION_Z)
    
    print("\nTransformation Matrix (4x4):")
    print(np.round(T, 4))
    
    # --- 4. Benchmark: Vectorized Transformation ---
    start_time = time.perf_counter()
    
    # Vectorized computation: (N, 4) @ (4, 4).T -> (N, 4)
    # This pushes the loop entirely into C/BLAS level operations
    transformed_points = points @ T.T
    
    execution_time = time.perf_counter() - start_time
    print(f"\nSuccessfully transformed {N_POINTS} points.")
    print(f"It took {execution_time:.5f} seconds.")
    
    # --- 5. Output Preview & Visualization ---
    print("\nSample Output (First 3 transformed points):")
    print(np.round(transformed_points[:3], 4))
    
    visualize_transformation(points, transformed_points)

if __name__ == "__main__":
    main()
