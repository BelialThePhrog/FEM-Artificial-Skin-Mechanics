import numpy as np
import matplotlib.pyplot as plt

def compute_strain_tensor_2d(node_coords, node_displacements):
    """
    Computes the Cauchy strain tensor for a 4-node quadrilateral (Q4) element
    at its center of mass (xi = 0, eta = 0).
    """
    # 1. Shape function derivatives in the local coordinate system (xi=0, eta=0 - element center)
    dN_dnatural = np.array([
        [-0.25,  0.25, 0.25, -0.25],  # dN/d_xi  (for nodes 1, 2, 3, 4)
        [-0.25, -0.25, 0.25,  0.25]   # dN/d_eta (for nodes 1, 2, 3, 4)
    ])
    
    # 2. Compute the Jacobian Matrix (J) - mapping from local to global coordinates
    J = np.dot(dN_dnatural, node_coords)
    invJ = np.linalg.inv(J)
    
    # 3. Shape function derivatives in the global coordinate system (x, y): [dN/dx, dN/dy]
    dN_dx = np.dot(invJ, dN_dnatural)
    
    # 4. Compute the displacement gradient (grad_u)
    # grad_u[i, j] = du_i / dx_j
    grad_u = np.dot(node_displacements.T, dN_dx.T)
    
    # 5. Cauchy strain tensor: epsilon = 0.5 * (grad_u + grad_u^T)
    epsilon = 0.5 * (grad_u + grad_u.T)
    
    return epsilon


def visualize_element_deformation(nodes, u, epsilon):
    """
    Visualizes the finite element deformation and overlays the displacement vectors.
    """
    deformed_nodes = nodes + u
    
    # Close the quadrilateral loop (add the 1st node at the end)
    closed_original = np.vstack([nodes, nodes[0]])
    closed_deformed = np.vstack([deformed_nodes, deformed_nodes[0]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot original and deformed shapes
    ax.plot(closed_original[:, 0], closed_original[:, 1], 'k--o', label=r'Original shape $\Omega_0$', linewidth=1.5)
    ax.plot(closed_deformed[:, 0], closed_deformed[:, 1], 'r-o', label=r'Deformed shape $\Omega_t$', linewidth=2)
    
    # Plot displacement vectors (arrows)
    for i in range(len(nodes)):
        ax.quiver(nodes[i, 0], nodes[i, 1], u[i, 0], u[i, 1], 
                  angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6,
                  width=0.005, headwidth=4)
        ax.text(nodes[i, 0] - 0.05, nodes[i, 1] - 0.05, f'N{i+1}', fontsize=10, fontweight='bold')
    
    # Add tensor values information to the plot
    info_text = (f"Strain Tensor $\\epsilon$ (at center):\\n"
                 f"$\\epsilon_{{xx}}$ (X tension): {epsilon[0,0]:.4f}\\n"
                 f"$\\epsilon_{{yy}}$ (Y tension): {epsilon[1,1]:.4f}\\n"
                 f"$\\epsilon_{{xy}}$ (Shear):     {epsilon[0,1]:.4f}")
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('X Axis [mm]')
    ax.set_ylabel('Y Axis [mm]')
    ax.set_title('Synthetic Skin Element Deformation Visualization (FEM Q4)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axis('equal')
    
    plt.show()


# ==============================================================================
# --- CONFIGURATION SECTION (MODIFY TEST PARAMETERS HERE) ---
# ==============================================================================
if __name__ == "__main__":
    
    # [VARIABLE 1] Node geometry before deformation [x, y] for 4 nodes:
    # Nodes ordered counter-clockwise (1: bottom-left, 2: bottom-right, 3: top-right, 4: top-left)
    nodes = np.array([
        [0.0, 0.0],  # Node 1
        [1.0, 0.0],  # Node 2
        [1.0, 1.0],  # Node 3
        [0.0, 1.0]   # Node 4
    ])
    
    # [VARIABLE 2] Node displacement vectors [u_x, u_y]:
    # Modify these values to observe the impact of pure tension, shear, or torsion!
    u = np.array([
        [0.00,  0.00],  # Displacement Node 1 [ux, uy]
        [0.10, -0.01],  # Displacement Node 2 [ux, uy]
        [0.10, -0.03],  # Displacement Node 3 [ux, uy]
        [0.00, -0.02]   # Displacement Node 4 [ux, uy]
    ])

    # --- COMPUTATION AND VISUALIZATION ---
    tensor_eps = compute_strain_tensor_2d(nodes, u)
    
    print("==================================================")
    print("STRAIN TENSOR COMPUTATION RESULTS (CENTER OF MASS):")
    print("==================================================")
    print(f"epsilon_xx (Axial tension/compression X) : {tensor_eps[0, 0]:.6f}")
    print(f"epsilon_yy (Axial tension/compression Y) : {tensor_eps[1, 1]:.6f}")
    print(f"epsilon_xy (Shear strain)                : {tensor_eps[0, 1]:.6f}")
    print("\nStrain Tensor Matrix epsilon:\n", tensor_eps)
    print("==================================================")
    
    # Trigger graphical visualization
    visualize_element_deformation(nodes, u, tensor_eps)
