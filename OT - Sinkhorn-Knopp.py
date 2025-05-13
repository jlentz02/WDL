#Import packages
import numpy as np
import matplotlib.pyplot as plt

#2d-discrete OT
#cost_matrix is the cost matrix, the source is a array of mass at each location
#the target also an array of mass, both normalized to 1, reg is a regularization parameter

#The data should probably start as 2 arrays of points. An x pos, y pos, and the mass at that point. We can then extract the source and target from the masses
#and compute the cost_matrix from the location information. 

def sinkhorn_knopp(cost_matrix, source, target, reg):
    eps = 1e-4

    P = np.exp(-cost_matrix / reg)
    P = P/P.sum()
    
    #Reshaping to correspond to rows and columns
    source = source.reshape(-1, 1)
    target = target.reshape(1,-1)

    #algorithm
    err = 1
    ii = 0
    P_prev = np.copy(P)
    while err > eps:
        ii + 1
        row_ratio = source/P.sum(axis = 1, keepdims = True)
        P = P*row_ratio
        col_ratio = target/P.sum(axis = 0, keepdims= True)
        P = P*col_ratio

        err = np.linalg.norm(P_prev - P, "fro")
        P_prev = np.copy(P)

    min_cost = np.sum(P*cost_matrix)
    return P, min_cost   

def visualize_transport(source_points, target_points, transport_plan):
    # Normalize mass_matrix for line width or color intensity
    mass_normalized = transport_plan / np.max(transport_plan)

    # Plot
    plt.figure(figsize=(8, 6))

    # Plot source points
    plt.scatter(source_points[:, 0], source_points[:, 1], c='blue', label='Source Points', s=100)

    # Plot target points
    plt.scatter(target_points[:, 0], target_points[:, 1], c='red', label='Target Points', s=100)

    # Plot lines from source to target with intensity proportional to mass
    for i in range(source_points.shape[0]):
        for j in range(target_points.shape[0]):
            if transport_plan[i, j] > 0:  # Only plot if mass is moved
                plt.plot(
                    [source_points[i, 0], target_points[j, 0]],
                    [source_points[i, 1], target_points[j, 1]],
                    linewidth=mass_normalized[i, j] * 5,  # Scale line width
                    alpha=mass_normalized[i,j],  # Transparency
                    color='green'  # Line color
                )

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Mass Movement Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

source_array = np.array([[0,1], [0,0], [1,0]])
target_array = np.array([[1,3], [3,3], [3,1]])


cost_matrix = np.array([[5,13,9], [10,18,10], [9,13,5]])
source = np.array([1/3,1/3,1/3])
target = np.array([1/4,1/2,1/3])
reg = .1

transport_plan, cost = sinkhorn_knopp(cost_matrix, source, target, reg)

visualize_transport(source_array, target_array, transport_plan)