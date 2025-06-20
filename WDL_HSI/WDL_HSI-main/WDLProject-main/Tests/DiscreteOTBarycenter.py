import torch
import ot
import math
import numpy as np 
import matplotlib.pyplot as plt
import time

#Generates the cost matrix
def make_cost_matrix(size, scale):
    # Create row and column index tensors
    i = torch.arange(size).unsqueeze(0)  # shape (1, n)
    j = torch.arange(size).unsqueeze(1)  # shape (n, 1)

    # Compute the matrix of absolute differences
    abs_diff_matrix = torch.abs(i - j)/scale
    return abs_diff_matrix

# Given two input distributions, returns the natural coupling
def make_coupling(a, b):
    """
    Compute the natural (greedy) coupling between two discrete distributions in PyTorch.

    Parameters:
    - a: 1D torch.Tensor of shape (n,)
        First distribution (source), must be non-negative and sum to 1.
    - b: 1D torch.Tensor of shape (m,)
        Second distribution (target), must be non-negative and sum to 1.

    Returns:
    - pi: 2D torch.Tensor of shape (n, m)
        The natural (greedy) coupling matrix.
    """
    a = a.clone().detach()
    b = b.clone().detach()
    n, m = a.shape[0], b.shape[0]
    pi = torch.zeros((n, m), dtype=a.dtype, device=a.device)

    i = j = 0
    while i < n and j < m:
        flow = torch.minimum(a[i], b[j])
        pi[i, j] = flow
        a[i] -= flow
        b[j] -= flow

        if a[i] == 0:
            i += 1
        if b[j] == 0:
            j += 1

    return pi


# Project coupling matrix into a 1D barycenter vector via weighted diagonal projection
def do_barycenter_step(pi, weight):
    n, m = pi.shape
    support_size = n
    bary = torch.zeros(support_size, dtype=pi.dtype, device=pi.device)

    for i in range(n):
        for j in range(m):
            mass = pi[i, j]
            if mass == 0:
                continue

            k = weight * i + (1- weight) * j
            if k.is_integer():
                bary[int(k)] += mass
            elif weight == 0.5:
                floor_k = math.floor(k)
                ceil_k = math.ceil(k)
                bary[floor_k] += .5 * mass
                bary[ceil_k] += .5 * mass
            else:
                rounded_k = round(k)
                bary[rounded_k] += mass

    return bary

def compute_barycenter(A, weights):
    prior_distribution = A[:, 0]
    for i in range(len(weights) - 1):
        l = torch.sum(weights[0:i+1])/torch.sum(weights[0:i + 2])
        coupling = make_coupling(prior_distribution, A[:, i+1])
        l = float(l)
        b = do_barycenter_step(coupling, l)
        prior_distribution = b
    return b



def barycenter2(vectors, weights):
    """
    vectors: torch.Tensor of shape (n, m), each row sums to 1.
    Returns: list of extracted min values (length will be <= n*m)
    """
    vectors = vectors.clone().T # Don't modify original
    n, m = vectors.shape
    pointers = torch.zeros(n, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool)
    barycenter = torch.zeros(m, dtype = torch.double)

    while not done.all():
        valid = ~done
        active_vals = torch.full((n,), float('inf'), device=vectors.device, dtype = torch.double)

        # Only read active values where pointers are in bounds
        active_vals[valid] = vectors[torch.arange(n)[valid], pointers[valid]]
        minval = active_vals.min()

        # Updates barycenter
        k = int(torch.round(torch.sum(pointers*weights)))
        barycenter[k] += minval


        # Subtract minval from active entries
        vectors[torch.arange(n)[valid], pointers[valid]] -= minval

        # Check which entries became ~0 and move pointers
        new_zeros = (vectors[torch.arange(n)[valid], pointers[valid]] <= 1e-6)
        advanced = torch.arange(n)[valid][new_zeros]
        pointers[advanced] += 1

        # Update done mask
        done = pointers >= m

    return barycenter

def barycenter3(vectors, weights):
    n, m = vectors.shape

    # Flatten the vectors to shape (n * m,) 
    flat_values = vectors.flatten()

    # Create source indices: 0 repeated m times, 1 repeated m times, ..., n-1
    source_indices = torch.arange(n).repeat_interleave(m)

    # Sort by values
    sorted_vals, sort_idx = flat_values.sort(descending=False)  # or True if you want descending

    # Apply the sort to source indices
    sorted_sources = source_indices[sort_idx]

    prior_value = 0
    current_indices = torch.zeros(n)
    barycenter = torch.zeros(m)
    for i in range(n*m - n + 1):
        #Get current value in vector
        current_value = sorted_vals[i]
        #Determine barycenter position of value and add value
        k = int(torch.round(torch.sum(current_indices*weights)))
        barycenter[k] += current_value - prior_value
        #Update prior value and indices
        prior_value = current_value
        current_indices[sorted_sources[i]] += 1
    return barycenter


def random_distribution(length, alpha=1.0):
    """
    Generate a random probability distribution of a given length using Dirichlet sampling.

    Parameters:
    - length: int, number of entries in the distribution
    - alpha: float or list, concentration parameter(s) for Dirichlet

    Returns:
    - probs: torch.Tensor of shape (length,), summing to 1
    """
    alpha_vec = torch.full((length,), alpha)
    return torch.distributions.Dirichlet(alpha_vec).sample()


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

def kl(a,b):
    cost = a*torch.log(a/b) - a + b
    return torch.sum(cost)




torch.manual_seed(42)

x = discrete_gaussian(700, 50, 1000)
y = discrete_gaussian(300, 50, 1000) 
x = x/torch.sum(x)
y = y/torch.sum(y)



num_dists = 2
dist_size = 1000
dists = [0 for x in range(num_dists)]
for i in range(num_dists):
    dists[i] = random_distribution(dist_size)


X = torch.stack([x,y])
X_cdf = torch.cumsum(X, dim = 1)
X = X.T
reg = 0.001
weights_prelim = [1/num_dists for x in range(num_dists)]
weights = torch.tensor(weights_prelim, dtype = torch.double)
#weights = random_distribution(num_dists).double()
weights_manual = torch.tensor([.40,.6], dtype = torch.double)
#weights = weights_manual
C = make_cost_matrix(dist_size,1).to(torch.double)**2/(dist_size)**2


start1 = time.time()
bary = ot.barycenter(X, C, reg, weights)
end1 = time.time()
print(f"Timer 1 took {end1 - start1:.4f} seconds")

start2 = time.time()
my_bary = barycenter3(X_cdf, weights)
end2 = time.time()

print(f"Timer 2 took {end2 - start2:.4f} seconds")
print(f"Time ratio: {(end1 - start1)/(end2 -start2 + 1e-8):.4f}")

computed_cost = 0
my_cost = 0
for i in range(X.shape[1]):
    computed_barycenter = ot.emd(bary, X[:,i], C, numItermax=1e6)
    computed_cost += torch.sum(computed_barycenter*C)*weights[i]
    my_cost += torch.sum(ot.emd(my_bary, X[:,i], C, numItermax=1e6)*C)*weights[i]
computed_cost = float(computed_cost)
my_cost = float(my_cost)
print("Computed bary total cost: ", computed_cost)
print("My bary total cost:       ", my_cost)
print("Price factor:             ", computed_cost/my_cost)

# X-axis indices
indices = torch.arange(len(X[:,0]))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(indices, x)
plt.plot(indices, y)
plt.plot(indices, bary, marker='^', label='bary', linestyle='--', linewidth=2)
plt.plot(indices, my_bary, marker='v', label='my_bary', linestyle='--', linewidth=2)

# Aesthetics
plt.title(f"Barycenters, reg: {reg}, price ratio: {computed_cost/my_cost}")
plt.xlabel("Index")
plt.ylabel("Probability")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


exit()
