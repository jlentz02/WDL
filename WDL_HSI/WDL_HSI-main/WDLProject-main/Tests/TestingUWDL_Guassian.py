import torch
import ot
from torch.optim import Adam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import numpy as np
import random
import math

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

""" #########TEST
    p = torch.exp(torch.log(A) @ weights)
    return p """

def UOT_barycenter_batched(A, C, reg, reg_m, weights, numItermax = 100):
    
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


def bregmanBary(D: torch.Tensor,
                weights: torch.Tensor,
                K: torch.Tensor,
                reg: float = 1.0,
                maxiter: int = 5,
                dev: torch.device = torch.device("cpu"),
                ):
    """

    :param D: a (d x m) tensor where d is the size of the support of the distributions
    and m is the number of dictionary atoms
    (each column is a new dictionary atom)
    :param weights: a (m x k) tensor where k is the number of barycenters to compute
    (each column is a set of weights)
    :param K: exp(-C/reg)
    :param reg: the entropic regularization parameter
    :param maxiter: the maximum number of sinkhorn iterations to run
    :return: the (d x k) tensor of k barycenters
    """
    if len(weights.shape) == 1:
        weights = weights.view(-1, 1)
    n_barys = weights.shape[1]
    n_hists = D.shape[1]
    # init variables
    b = torch.ones((n_barys, D.shape[0], D.shape[1]), device=dev).to(torch.double)
    Kt = K.mT

    # bregman projection loop (as in WDL paper)
    for i in range(maxiter):
        phi = torch.matmul(Kt, torch.div(D, torch.matmul(K, b)))
        p = torch.bmm(phi.log().view(n_barys, -1, n_hists),
                      weights.mT.view(n_barys, n_hists, 1)).exp()
        # p = torch.matmul(phi.log(), weights).exp()
        b = torch.div(p, phi)
    
    return p.view(n_barys, -1).mT



def make_cost_matrix(X):
    n_hists = X.shape[1]
    C = torch.zeros(n_hists, n_hists)
    for i in range(0, n_hists):
        for j in range(0, n_hists):
            C[i,j] = torch.norm(X[:,i] - X[:,j], p = 2)

    return C

def WDL(X, D, C, reg, reg_m, num_epochs = 100, loss_type = "quadratic", type = "uot"):
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
    #Store loss information for each data point on each iteration
    #in order to visualize the learning process.
    #I suspect their are local non-zero minimum that the algorithm is finding
    loss_data = [0 for x in range(0,num_epochs)]


    for epoch in range(num_epochs):
        optimizer.zero_grad()

        w = torch.nn.functional.softmax(weights, dim = 0)
        # 1. Reconstruct via barycenter
        if type == "uot":
            p = UOT_barycenter_batched(D, C, reg, reg_m, w)
        elif type == "ot":
            K = torch.exp(-C / reg)
            p = bregmanBary(D, weights, K, reg = reg, maxiter= 100)
        #Testing weighted geomeans
        # 2. Compute loss
        if loss_type == "quadratic":
            loss_vector = (p - X)**2
        elif loss_type == "kl":
            loss_vector = p*torch.log((p/X)) - p + X
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
        D.data = D.data.clamp(min = 1e-15)
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

def plot_reconstruction_compare(X, rec1, D1, rec2, D2, num_epochs, reg, reg_m):
    # Clone tensors to avoid modifying originals
    X = X.detach().clone()
    rec1 = rec1.detach().clone()
    D1 = D1.detach().clone()
    rec2 = rec2.detach().clone()
    D2 = D2.detach().clone()

    plt.figure(figsize=(10, 6))

    # Plot original data (X) in blue
    for i in range(X.size(0)):
        plt.plot(X[i].numpy(), color='blue', alpha=0.5, label='X' if i == 0 else "")

    # Plot first reconstruction (rec1) in red
    for i in range(rec1.size(0)):
        plt.plot(rec1[i].numpy(), color='red', alpha=0.5, label='rec1' if i == 0 else "")

    # Plot first dictionary atoms (D1) in green dashed
    for i in range(D1.size(0)):
        plt.plot(D1[i].numpy(), color='green', alpha=0.5, linestyle='--', label='D1' if i == 0 else "")

    # Plot second reconstruction (rec2) in orange
    for i in range(rec2.size(0)):
        plt.plot(rec2[i].numpy(), color='orange', alpha=0.5, label='rec2' if i == 0 else "")

    # Plot second dictionary atoms (D2) in purple dashed
    for i in range(D2.size(0)):
        plt.plot(D2[i].numpy(), color='purple', alpha=0.5, linestyle='--', label='D2' if i == 0 else "")

    plt.title(f"Comparison of Reconstructions and Dictionaries\n"
              f"num epochs: {num_epochs}, reg: {reg}, reg_m: {reg_m}")
    plt.legend(prop={'size': 12})
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
def gaussian_vector(mean, std_dev, num_samples, scale = 1):
    # Generate x values centered around the mean
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, num_samples)

    # Gaussian function
    gaussian_vector = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    gaussian_vector = gaussian_vector / np.sum(gaussian_vector)
    return gaussian_vector

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

def unbalanced_transport_cost(C, X, a, b, tau):
    """
    Compute the unbalanced OT cost:
    <C, X> + tau * KL(X1 || a) + tau * KL(X2 || b)

    Parameters:
    - C: (n, m) cost matrix
    - X: (n, m) transport plan
    - a: (n,) source marginal
    - b: (m,) target marginal
    - tau: scalar or scalar tensor
    - kl: function kl(p, q) -> scalar

    Returns:
    - total cost: scalar tensor
    """
    transport_cost = torch.sum(C * X)
    X1 = torch.sum(X, dim=1)  # row sums
    X2 = torch.sum(X, dim=0)  # column sums
    return transport_cost + tau * (kl(X1, a) + kl(X2, b))


# Parameters for the Gaussian distribution
mean = 10       # Center of the distribution
std_dev = 1.0    # Standard deviation (spread)
num_samples = 500  # Length of the vector


# --- Step 1: Define the grid ---
x = np.linspace(-5, 10, 500)

# --- Step 2: Define two Gaussian distributions with different means and different masses ---
def gaussian(x, mean, sigma, mass=1.0):
    g = np.exp(-0.5 * ((x - mean)/sigma)**2)
    g /= g.sum()  # normalize to 1
    g *= mass     # scale to desired total mass
    return g

mu1 = gaussian(x, mean=0, sigma=.5, mass=1.0)
mu2 = gaussian(x, mean=5, sigma=.5, mass=1)
mu3 = gaussian(x, mean = 2.5, sigma = .5, mass = 1)

Xinit = np.array([mu1, mu2, mu3])

X = torch.tensor(Xinit, dtype = torch.double)


#Make D by sampling randomly from X
num_atoms = 2
#random_indices = torch.randperm(X.size(0))[:num_atoms]
random_indices = torch.tensor([0, 1])
D = X[random_indices,:]
D.requires_grad_(True)

#Transposing to make them n x m where n is the size of the support and m is the number of data points
X = X.T
D = D.T

#Make C
C = make_cost_matrix(num_samples, num_samples).to(torch.double)


reg = 0.1
reg_m = 1

num_epochs = 500
rec_uot, D_uot, loss_data = WDL(X, D, C, reg, reg_m, num_epochs=num_epochs, loss_type = "quadratic", type = "uot")
rec_ot, D_ot, loss_data_ot = WDL(X, D, C, reg, reg_m, num_epochs=num_epochs, loss_type = "quadratic", type = "ot")
plot_reconstruction_compare(X.T, rec_uot.T, D_uot.T, rec_ot.T, D_ot.T, num_epochs=num_epochs, reg = reg, reg_m = reg_m)