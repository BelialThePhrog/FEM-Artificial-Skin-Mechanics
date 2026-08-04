import numpy as np

def demonstrate_broadcasting():
    print("--- 1. Basic Broadcasting: 2D Matrix + 1D Vector ---")
    matrix_2d = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    vector_1d = np.array([10, 20, 30])
    
    result = matrix_2d + vector_1d
    print(f"Resulting matrix (3x3):\n{result}\n")

    print("--- 2. 3D Image Normalization (RGB) ---")
    # 3D Image: 100x100 pixels, 3 color channels (RGB) -> Shape: (100, 100, 3)
    image = np.random.randint(0, 256, size=(100, 100, 3))
    
    # Mean values for R, G, B channels -> Shape: (3,)
    mean_channels = np.array([123.68, 116.779, 103.939])
    
    # Broadcasting automatically matches the (3,) array to the (100, 100, 3) image
    normalized_image = image - mean_channels
    print(f"Original image shape: {image.shape}")
    print(f"Normalized image shape: {normalized_image.shape}\n")

    print("--- 3. Distance Matrix (4 points x 3 points) ---")
    # Set A: 4 points in 2D space -> Shape (4, 2)
    A = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    
    # Set B: 3 points in 2D space -> Shape (3, 2)
    B = np.array([[1, 0], [0, 1], [2, 1]])
    
    # Expanding dimensions: A to (4, 1, 2) and B to (1, 3, 2)
    # The difference creates a combination grid of shape (4, 3, 2)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    
    # Calculate Euclidean distance along the last axis (X,Y coordinates)
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    print(f"Distance Matrix:\n{np.round(distances, 2)}\n")

    print("--- 4. Feature Weighting ---")
    # Data matrix: 5 samples, 3 features -> Shape (5, 3)
    X = np.array([
        [25, 5000, 2],
        [30, 7000, 5],
        [45, 12000, 10],
        [20, 3000, 1],
        [35, 8500, 7]
    ])
    
    # Weights for each feature -> Shape (3,)
    weights = np.array([0.1, 0.001, 2.0])
    
    # Broadcasting: Each column is multiplied by the corresponding weight
    X_weighted = X * weights
    print(f"Weighted data:\n{X_weighted}\n")

    print("--- 5. 2D Grid Creation ---")
    # Coordinate vectors X and Y -> Shape (5,)
    x = np.linspace(-np.pi, np.pi, 5) 
    y = np.linspace(-np.pi, np.pi, 5) 
    
    # Reshape x to a column vector (5, 1) while y remains horizontal (5,)
    # Broadcasting builds a full 5x5 grid
    Z = np.sin(x[:, np.newaxis]) + np.cos(y)
    print(f"Z values grid (5x5):\n{np.round(Z, 2)}\n")

    print("--- 6. Data Standardization ---")
    # Random data matrix: 1000 samples, 4 features
    data = np.random.normal(size=(1000, 4))
    
    # Calculate mean and standard deviation for each column -> Shape (4,)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Broadcasting subtracts the (4,) vector and divides by the (4,) vector for all 1000 rows
    data_standardized = (data - mean) / std
    print(f"New column means (close to 0): {np.round(np.mean(data_standardized, axis=0), 8)}")
    print(f"New column std devs (close to 1): {np.round(np.std(data_standardized, axis=0), 8)}")

if __name__ == "__main__":
    demonstrate_broadcasting()
