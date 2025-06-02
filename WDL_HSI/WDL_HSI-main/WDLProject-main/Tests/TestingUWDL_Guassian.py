import torch
import ot
from torch.optim import Adam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

#UOT_barycenter method analogous to the method used in POT library
#Rewritten to more explicity use pytorch tensors and to be able to compute k barycenters
#I will start by implementing the algorithm for k = 1 to get a feel for things, then vectorize it.


#A - n x d tensor of d dictionary atoms
#C - n x n cost matrix
#reg - entropic regularization term
#reg_m - marginal relaxation term (also known as tau)
#weights - d x 1 tensor of weights where k is the number of barycenters to be returned
def UOT_barycenter(A, C, reg, reg_m, weights = None, numItermax = 100):
    #Establishing how big things are
    dim, n_hists = A.shape
    
    #Initializing weights as uniform if none are given
    #k = 1 case right now
    if weights is None:
        weights = torch.ones(n_hists, dtype = torch.double) / n_hists

    #Initializing K, the central term for the bregman project loop
    K = torch.exp(-C/ reg)

    #Initializing loop variables
    #fi is the UOT scaling term I guess, see WDL paper
    #v is analogous to b in the WDL paper, the first marignal, though in this case since we are doing barycenters it is a bit different
    #u is analogous to a in the WDL paper, it is the second marginal
    #q is analogous to p, this is the barycenter approx
    #When k is not 1 I think q will need to have a second dimension, same for u
    fi = reg_m/(reg_m + reg)
    v = torch.ones((dim, n_hists), dtype = torch.double)
    u = torch.ones((dim, 1), dtype = torch.double)
    q = torch.ones(dim, dtype = torch.double)

    #Main loop 
    q_prev = 0
    for i in range(numItermax):
        Kv = torch.matmul(K, v)
        u = (A / Kv)**fi
        Ktu = torch.matmul(K.T, u)
        q = torch.matmul(Ktu**(1-fi), weights)
        q = q **(1/(1-fi))
        Q = q[:, None]
        v = (Q/Ktu)**fi

        #Breakout early if changes are small
        if torch.sum(q - q_prev) < 0.0001:
            return q
        q_prev = q.detach().clone()
    
    return q


#Implemetation of normal sinkhorn_knopp_unbalanced from POT library.
#Then we will make it behave like the currently method does
#a - n x 1 source distribution
#b - n x 1 target distribution
#C - cost matrix
#reg - entropic regularization 
#reg_m - marginal relaxation term
#numMaxIter - maximum number of iterations
def sinkhorn_unbalanced(a, b, C, reg, reg_m, numMaxIter = 100, plan = True):
    dim_a, dim_b = C.shape
    #initialization of sinkhorn scaling vectors
    u = torch.ones((dim_a), dtype = torch.double)
    v = torch.ones((dim_b), dtype = torch.double)

    #More initialized variables
    c = a[:, None] * b[None, :]

    K = torch.exp(-C / reg) * c
    fi = reg_m/(reg+reg_m)
    #Main loop
    for i in range(numMaxIter):
        Kv = torch.matmul(K, v)
        u = (a/Kv)**fi
        Ktu = torch.matmul(K.T, u)
        #Ktu = Ktu.unsqueeze(1)
        v = (b/Ktu)**fi

    transport_plan = u[:, None] * K * v[None, :]

    if plan:
        return transport_plan
    else:
        return torch.sum(transport_plan*C)


#A - n x d tensor of d dictionary atoms
#C - n x n cost matrix
#reg - entropic regularization term
#reg_m - marginal relaxation term (also known as tau)
#weights - d x m tensor of weights where m is the number of barycenters to be returned
#TODO Decide if weights should be d x m or m x d and then remove transpositions if necessary
def UOT_barycenter_batched(A, C, reg, reg_m, weights, numItermax = 100):
    #########TEST
    p = torch.exp(torch.log(A) @ weights)
    return p



    #Establishing how big things are
    dim, n_hists = A.shape

    #Checks if a is single or multidimensional and runs the single dimensional version if not
    try:
        n_barys = weights.shape[1]
    except:
        q = UOT_barycenter(A, C, reg, reg_m, weights)
        return q
    
    #Initializing weights as uniform if none are given
    #k = 1 case right now
    if weights is None:
        weights = torch.ones(n_hists, dtype = torch.double) / n_hists

    #Initializing K, the central term for the bregman project loop
    K = torch.exp(-C/ reg)

    #Initializing loop variables
    #fi is the UOT scaling term I guess, see WDL paper
    #v is analogous to b in the WDL paper, the first marignal, though in this case since we are doing barycenters it is a bit different
    #u is analogous to a in the WDL paper, it is the second marginal
    #q is analogous to p, this is the barycenter approx
    #When k is not 1 I think q will need to have a second dimension, same for u
    fi = reg_m/(reg_m + reg)
    v = torch.ones((n_barys,dim, n_hists), dtype = torch.double)
    u = torch.ones((n_barys, dim, 1), dtype = torch.double)
    q = torch.ones((n_barys,dim), dtype = torch.double)

    #Changing sizes of things for bmm
    weights = weights.T.unsqueeze(2)


    #Main loop 
    q_prev = 0
    for i in range(numItermax):
        Kv = torch.matmul(K.unsqueeze(0), v)
        u = (A.unsqueeze(0) / Kv)**fi
        Ktu = torch.matmul(K.T.unsqueeze(0), u)
        q = torch.bmm(Ktu**(1-fi), weights)
        q = q.squeeze(2)
        q = q **(1/(1-fi))
        Q = q.unsqueeze(2)
        v = (Q/Ktu)**fi
        
        #Breakout early if changes are small
        if torch.sum(q - q_prev) < 0.0001:
            return q.T
        q_prev = q.detach().clone()

    return q.T



#Implemetation of normal sinkhorn_knopp_unbalanced from POT library.
#Modified to do batches of size m
#This just solves the OT problem and returns the transport cost between a_i and b_i
#We actually don't need a and b to be supported on the same size set
#a - n x m source distribution
#b - n x m target distribution
#C - cost matrix
#reg - entropic regularization 
#reg_m - marginal relaxation term
#numMaxIter - maximum number of iterations
def sinkhorn_unbalanced_batched(a, b, C, reg, reg_m, numMaxIter = 100):
    dim_a, dim_b = C.shape
    n_hists = a.shape[1]
    #initialization of sinkhorn scaling vectors
    u = torch.ones((n_hists,dim_a), dtype = torch.double)
    v = torch.ones((n_hists,dim_b), dtype = torch.double)

    #More initialized variables
    c = a.permute(1, 0).unsqueeze(2) * b.permute(1, 0).unsqueeze(1)
    K = torch.exp(-C / reg) * c
    fi = reg_m/(reg+reg_m)
    #Main loop
    Ktu_prev = 0
    for i in range(numMaxIter):
        Kv = torch.bmm(K, v.unsqueeze(2)).squeeze(2)
        u = (a.T/Kv)**fi
        K_t = K.transpose(1,2)
        Ktu = torch.bmm(K_t, u.unsqueeze(2)).squeeze(2)
        #Ktu = Ktu.unsqueeze(1)
        v = (b.T/Ktu)**fi

        #Breakout early if changes are small
        if torch.sum(Ktu - Ktu_prev) < 0.0001:
            plan = u[:,:, None] * K * v[:,None, :]
            costs = torch.sum(plan*C, dim = (1,2))
            return costs
        Ktu_prev = Ktu.detach().clone()

    plan = u[:,:, None] * K * v[:,None, :]
    linear_costs = torch.sum(plan*C, dim = (1,2))
    return linear_costs



def make_cost_matrix(X):
    n_hists = X.shape[1]
    C = torch.zeros(n_hists, n_hists)
    for i in range(0, n_hists):
        for j in range(0, n_hists):
            C[i,j] = torch.norm(X[:,i] - X[:,j], p = 2)

    return C

def WDL(X, D, C, reg, reg_m, num_epochs = 100, loss_type = "quadratic"):
    weight_length = D.shape[1]
    num_weight_vectors = X.shape[1]
    alpha = torch.ones(weight_length)
    weights = torch.distributions.Dirichlet(alpha).sample((num_weight_vectors, ))

    weights = weights.detach().clone().requires_grad_(True).to(torch.double).T

    D = torch.nn.Parameter(D.clone().detach())
    weights = torch.nn.Parameter(weights.clone().detach())

    params = [D, weights]  # D: (k, n), weights: (batch_size, k)
    optimizer = Adam(params, lr=1e-2)

    #New Step:
    #Precompute the cost of transport a point in X onto itself.
    #Then we are going to make the cost = abs(cost - base_cost)
    base_cost = sinkhorn_unbalanced_batched(X, X, C, reg, reg_m)

    #New Step:
    #Store loss information for each data point on each iteration
    #in order to visualize the learning process.
    #I suspect their are local non-zero minimum that the algorithm is finding
    loss_data = [0 for x in range(0,num_epochs)]


    for epoch in range(num_epochs):
        optimizer.zero_grad()

        w = torch.nn.functional.softmax(weights, dim = 0)
        # 1. Reconstruct via barycenter
        p = UOT_barycenter_batched(D, C, reg, reg_m, w)
        #Testing weighted geomeans
        # 2. Compute loss
        if loss_type == "quadratic":
            loss_vector = (p - X)**2
        elif loss_type == "kl":
            loss_vector = p*torch.log((p/X)) - p + X
        elif loss_type == "uot":
            loss_vector = (sinkhorn_unbalanced_batched(X, p, C, reg, reg_m) - base_cost)**2
        else:
            print("Loss_type " + loss_type + " is not supported.")
            exit()

        # 2.5 Store loss values
        loss_data[epoch] = loss_vector

        loss = loss_vector.sum()
        if loss.isnan():
            print("Loss is nan on epoch:" + str(epoch))
            exit()
        # 3. Backprop and update
        loss.backward()
        optimizer.step()
        D.data = D.data.clamp(min = 1e-8)
        if epoch%25 == 0:
            print("Loss on iteration:", epoch, loss)
    print("Final loss: " + str(loss))

    final_weights = torch.nn.functional.softmax(weights, dim=0)
    #print(D)
    #print(torch.sum(D, dim = 0))

    rec = torch.zeros(D.shape[0], final_weights.shape[1])

    for i in range(final_weights.shape[1]):
        rec[:,i] = UOT_barycenter_batched(D, C, reg, reg_m, final_weights.T[i])
    #print(final_weights)
    #print(rec)
    #print(D)
    

    #Returns reconstruction and the final atoms
    return rec, D, loss_data

#Todo for Gaussian:
#Pull code to make guassian distributions and then form samples by adding constants to different gaussians
#Make symmetric cost matrix
#Construct D by forming k additional gaussians scaled randomly, or selecting from the original data
#Hypothesis: With k = 2 the gaussians will move towards the high and low range of the gaussians in the data
#and the weight values will give decent reconstructions of the gaussians in between
#Add visualization plotting all of the stuff. As before data in blue, atoms in green, reconstructions in red


def plot_reconstruction(X, rec, D, num_epochs, reg, reg_m):
    X = X.detach().clone()
    rec = rec.detach().clone()
    D = D.detach().clone()

    plt.figure(figsize=(10, 6))

    # Plot original data (X) in blue
    for i in range(X.size(0)):
        plt.plot(X[i].numpy(), color='blue', alpha=0.5, label='X' if i == 0 else "")

    # Plot reconstructed data (rec) in red
    for i in range(rec.size(0)):
        plt.plot(rec[i].numpy(), color='red', alpha=0.5, label='rec' if i == 0 else "")

    # Plot dictionary atoms (D) in green
    for i in range(D.size(0)):
        plt.plot(D[i].numpy(), color='green', alpha=0.5, linestyle='--', label='D' if i == 0 else "")

    plt.title("Original Data (X), Reconstruction (rec), Dictionary Atoms (D), num epochs: " + str(num_epochs) + " reg: " + str(reg) + " reg_m: " + str(reg_m))
    plt.legend(prop = {'size': 16})
    plt.xlabel("Support")
    plt.ylabel("Mass")
    plt.grid(True)
    plt.show()

def plot_loss_data(loss_data):
    loss_data = torch.stack(loss_data)
    loss_data = loss_data.clone().detach()
    num_lines = loss_data.shape[1]
    for i in range(num_lines):
        plt.plot(loss_data[:, i], label = f'Dim {i}')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.title('Loss values over time for each data point reconstruction')
    plt.grid(True)
    plt.show()

# Generate the vector
def gaussian_vector(mean, std_dev, num_samples, scale):
    # Generate x values centered around the mean
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, num_samples)

    # Gaussian function
    gaussian_vector = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2) + scale + 1

    return gaussian_vector

#Generates the cost matrix
def make_cost_matrix(size, scale):
    # Create row and column index tensors
    i = torch.arange(size).unsqueeze(0)  # shape (1, n)
    j = torch.arange(size).unsqueeze(1)  # shape (n, 1)

    # Compute the matrix of absolute differences
    abs_diff_matrix = torch.abs(i - j)/scale
    return abs_diff_matrix

# Parameters for the Gaussian distribution
mean = 10       # Center of the distribution
std_dev = 1.0    # Standard deviation (spread)
num_samples = 100  # Length of the vector


#Make X
num_hists = 10
spread = [.70,.85,1,1.15, 1.3, 4.7, 4.85, 5, 5.15, 5.3]
spread = [1,2,3,4,5,6,7,8,9,10]
X = [0 for x in range(num_hists)]
for i in range(num_hists):
    X[i] = gaussian_vector(mean, std_dev, num_samples, spread[i]).tolist()

X = torch.tensor(X, dtype = torch.double)

#Make D by sampling randomly from X
num_atoms = 2
#random_indices = torch.randperm(X.size(0))[:num_atoms]
random_indices = torch.tensor([0, 9])
D = X[random_indices,:]
D.requires_grad_(True)

#Transposing to make them n x m where n is the size of the support and m is the number of data points
X = X.T
D = D.T

#Make C
C = make_cost_matrix(100, 100).to(torch.double)
C = C


reg = 0.001
reg_m = 1000
""" 
num_epochs = 500
rec, D, loss_data = WDL(X, D, C, reg, reg_m, num_epochs=num_epochs, loss_type = "kl")
print(torch.sum(D, dim = 0))
print(torch.sum(X, dim = 0))
print(torch.sum(rec, dim = 0))
plot_reconstruction(X.T, rec.T, D.T, num_epochs, reg, reg_m)
#plot_loss_data(loss_data)
 """


#Make C
C = make_cost_matrix(5, 1).to(torch.double)
C = C

A = torch.tensor([2,2,1,1,1], dtype = torch.double)
B = torch.tensor([2,3,1,1,8], dtype = torch.double)
x = torch.tensor([[2,2,1,1,1], [1,3,1,1,8], [1,4,4,5,1]], dtype = torch.double).T
weights = torch.tensor([.25,.25,.5], dtype = torch.double)
A = A + 1e-8
B = B + 1e-8
c = UOT_barycenter(x, C, reg, reg_m, weights)
print("Barycenter: ", c)
geo_mean = torch.exp(torch.sum(weights*torch.log(x + 1e-8), dim=1))
print("Weighted geometric mean: ", geo_mean)


""" 
A = torch.tensor([1,1,1], dtype = torch.double)

X = [(x+1)/50 for x in range(0,100)]
Y = [0 for x in range(0,100)]



for i in range(0,100):
    A_test = A
    B_test = A*X[i]
    cost = sinkhorn_unbalanced(A_test, B_test, C, reg, reg_m, plan = False)
    Y[i] = cost
    print(cost)

# Create scatter plot
plt.plot(X, Y, color='blue', marker='o')

# Add labels and title
plt.xlabel('Mass multiplier on A')
plt.ylabel('Cost')
plt.title('Scatter Plot of Inputs vs Outputs' + " reg: " + str(reg) + " reg_m: " + str(reg_m))

# Show plot
plt.grid(True)
plt.show()
 """





        

    
