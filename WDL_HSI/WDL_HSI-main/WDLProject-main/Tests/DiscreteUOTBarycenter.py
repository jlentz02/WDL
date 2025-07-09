import torch
import ot
import math
import numpy as np 
import matplotlib.pyplot as plt
import time
import math
from scipy.optimize import curve_fit


def discrete_gaussian(mean, std, size, support=None):
    """
    Create a discretely supported 1D Gaussian over a given support.

    Parameters:
    - mean: float, the mean of the Gaussian
    - std: float, the standard deviation
    - size: int, number of support points (length of output vector)
    - support: optional 1D torch.Tensor or list of x-values; if None, uses [0, 1, ..., size-1]

    Returns:
    - probs: torch.Tensor of shape (size,), normalized to sum to 1
    """
    if support is None:
        support = torch.arange(size, dtype=torch.double)
    else:
        support = torch.tensor(support, dtype=torch.double)

    probs = torch.exp(-0.5 * ((support - mean) / std) ** 2)
    probs /= probs.sum()
    return probs

#Generates the cost matrix
def make_cost_matrix(size, scale):
    # Create row and column index tensors
    i = torch.arange(size).unsqueeze(0)  # shape (1, n)
    j = torch.arange(size).unsqueeze(1)  # shape (n, 1)

    # Compute the matrix of absolute differences
    abs_diff_matrix = torch.abs(i - j)/scale
    return abs_diff_matrix

def kl(a,b):
    cost = a*torch.log(a/b) - a + b
    return torch.sum(cost)

def UOT_cost(X, C, a, b, reg_m):
    cost = torch.sum(X*C) + reg_m*kl(torch.sum(X, dim = 1), a) + reg_m*kl(torch.sum(X, dim = 0), b)
    return cost


torch.manual_seed(42)

dist_size = 2

""" 
x = discrete_gaussian(70, 5, dist_size)
y = discrete_gaussian(30, 5, dist_size) 
x = x/torch.sum(x)
y = y/torch.sum(y)
 """

# Your setup
a = torch.tensor([2, 1], dtype=torch.double)
b = torch.tensor([1, 3], dtype=torch.double)

lower_bound = torch.sum(torch.sqrt(a * b))
upper_bound = torch.sqrt(torch.sum(a) * torch.sum(b))
print("b =", lower_bound.item())
print("a =", upper_bound.item())

dist_size = len(a)
C = make_cost_matrix(dist_size, 1).to(torch.double)**2

# Generate (X, Y) from OT
x = []
y1 = []
y2 = []
y3 = []
for i in range(200):
    reg_m = 1.1 ** (i-10)
    plan = ot.unbalanced.mm_unbalanced(a, b, C, reg_m, div="kl")
    x.append(i)
    #y2.append(plan[1,1])
    w00 = (plan[0,0] + plan[0,1])*(plan[0,0] + plan[1,0])
    w10 = (plan[1,0] + plan[0,0])*(plan[1,0] + plan[1,1])
    w01 = (plan[0,1] + plan[0,0])*(plan[0,1] + plan[1,1])
    w11 = (plan[1,1] + plan[0,1])*(plan[1,1] + plan[1,0])
    y2.append((w00*w10*w01*w11)**(1/2))
    #y1.append(plan[:,1]@plan[0,:])
    #print(plan)
    
    

#plt.plot(x, y1)
plt.plot(x, y2)
plt.show()


exit()




print("UOT cost:                      ", UOT_cost(plan, C, x, y, reg_m))
print("Plan total:                    ", torch.sum(plan))
#print(UOT_cost(my_plan, C, x, y, reg_m))


# X-axis indices
indices = torch.arange(dist_size)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(indices, x)
plt.plot(indices, y)
plt.plot(indices, bary, marker='^', label='bary', linestyle='--', linewidth=2)
#plt.plot(indices, my_bary, marker='v', label='my_bary', linestyle='--', linewidth=2)

# Aesthetics
plt.title(f"Barycenters")
plt.xlabel("Index")
plt.ylabel("Probability")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

