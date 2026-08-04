import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_strain_tensor_3d(node_coords, node_displacements):
    """
    Computes the 3-dimensional Cauchy strain tensor (3x3) for an 8-node 
    hexahedral element (H8) at its center of mass (xi=0, eta=0, zeta=0).
    
    Parameters:
    node_coords: np.array (8x3) - node coordinates [x, y, z] before deformation
    node_displacements: np.array (8x3) - node displacement vectors [ux, uy, uz]
    
    Returns:
    epsilon: np.array (3x3) - full 3D strain tensor matrix
    """
    # 1. Derivatives of the 8 shape functions N_i with respect to natural coordinates (xi, eta, zeta)
    # Evaluated at the element center (xi = 0, eta = 0, zeta = 0):
    
    # Signs for the 8 nodes in the natural coordinate system (-1 or +1):
    corner_signs = np.array([
        [-1, -1, -1],  # Node 1
        [ 1, -1, -1],  # Node 2
        [ 1,  1, -1],  # Node 3
        [-1,  1, -1],  # Node 4
        [-1, -1,  1],  # Node 5
        [ 1, -1,  1],  # Node 6
        [ 1,  1,  1],  # Node 7
        [-1,  1,  1]   # Node 8
    ])
    
    # Derivative matrix 3x8: [dN/d_xi; dN/d_eta; dN/d_zeta]
    dN_dnatural = 0.125 * corner_signs.T
    
    # 2. Compute 3D Jacobian Matrix (J) [3x3]
    J = np.dot(dN_dnatural, node_coords)
    invJ = np.linalg.inv(J)
    
    # 3. Shape function derivatives in global coordinates [dN/dx, dN/dy, dN/dz]
    dN_dx = np.dot(invJ, dN_dnatural)
    
    # 4. 3D Displacement gradient (grad_u) [3x3]
    # grad_u[i, j] = du_i / dx_j
    grad_u = np.dot(node_displacements.T, dN_dx.T)
    
    # 5. 3D Cauchy strain tensor: epsilon = 0.5 * (grad_u + grad_u^T)
    epsilon = 0.5 * (grad_u + grad_u.T)
    
    return epsilon


def visualize_3d_deformation(nodes, u, epsilon):
    """
    Creates a 3D plot of the skin hexahedron before and after deformation.
    """
    deformed_nodes = nodes + u
    
    # Edges connecting the 8 nodes of the hexahedron
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face (z=0)
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face (z=h)
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical pillars
    ]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original edges (black dashed)
    for edge in edges:
        ax.plot3D(*zip(*nodes[edge]), color='black', linestyle='--', alpha=0.6, label='Primary' if edge == [0,1] else "")
        
    # Plot deformed edges (red solid)
    for edge in edges:
        ax.plot3D(*zip(*deformed_nodes[edge]), color='red', linewidth=2, label='Deformed' if edge == [0,1] else "")

    # Plot nodes
    ax.scatter3D(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='black', s=20)
    ax.scatter3D(deformed_nodes[:, 0], deformed_nodes[:, 1], deformed_nodes[:, 2], color='red', s=40)

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title('Synthetic Skin Modeling', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    
    # Display 3D tensor on the plot
    info = (f"3D Tensor (epsilon):\\n"
            f"exx: {epsilon[0,0]:.3f} | exy: {epsilon[0,1]:.3f} | exz: {epsilon[0,2]:.3f}\\n"
            f"eyx: {epsilon[1,0]:.3f} | eyy: {epsilon[1,1]:.3f} | eyz: {epsilon[1,2]:.3f}\\n"
            f"ezx: {epsilon[2,0]:.3f} | ezy: {epsilon[2,1]:.3f} | ezz: {epsilon[2,2]:.3f}")
    
    plt.figtext(0.15, 0.05, info, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.show()


# ==============================================================================
# --- TEST SECTION ---
# ==============================================================================
if __name__ == "__main__":

    # [3D SKIN ELEMENT] Dimensions: length X = 1.0mm, width Y = 1.0mm, THICKNESS Z = 0.2mm
    nodes_3d = np.array([
        [0.0, 0.0, 0.0],  # 1: bottom-left-front
        [1.0, 0.0, 0.0],  # 2: bottom-right-front
        [1.0, 1.0, 0.0],  # 3: bottom-right-back
        [0.0, 1.0, 0.0],  # 4: bottom-left-back
        [0.0, 0.0, 0.2],  # 5: top-left-front
        [1.0, 0.0, 0.2],  # 6: top-right-front
        [1.0, 1.0, 0.2],  # 7: top-right-back
        [0.0, 1.0, 0.2]   # 8: top-left-back
    ])

    # [REALISTIC DEFORMATION] 
    # Stretch along the X-axis (+15%), causing the skin to narrow in Y (-4%) and THIN in Z (-4%)
    u_3d = np.array([
        [0.00,  0.00,  0.000],  # 1
        [0.15, -0.02, -0.004],  # 2
        [0.15, -0.04, -0.004],  # 3
        [0.00, -0.02,  0.000],  # 4
        [0.00,  0.00, -0.008],  # 5 (top layer drops down -> thinning!)
        [0.15, -0.02, -0.012],  # 6
        [0.15, -0.04, -0.012],  # 7
        [0.00, -0.02, -0.008]   # 8
    ])

    eps_3d = compute_strain_tensor_3d(nodes_3d, u_3d)

    print("==================================================")
    print("3D STRAIN TENSOR MATRIX (3x3):")
    print("==================================================")
    print(np.round(eps_3d, 4))
    print("==================================================")
    print(f"Longitudinal stretch    (eps_xx): {eps_3d[0,0]:.4f}")
    print(f"Transverse compression  (eps_yy): {eps_3d[1,1]:.4f}")
    print(f"Z-axis patch thinning   (eps_zz): {eps_3d[2,2]:.4f}  <-- THIS IS THE THICKNESS EFFECT!")
    print("==================================================")

    visualize_3d_deformation(nodes_3d, u_3d, eps_3d)
