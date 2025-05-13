import ot
import torch
import numpy as np
from helper import Cost, sample, data_loader
from torch.optim import Adam
import matplotlib.pyplot as plt

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
    costs = torch.sum(plan*C, dim = (1,2))
    return costs

def plot_data(X, rec, D, num_epochs):
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
        plt.plot(D[i].numpy(), color='green', alpha=1, linestyle='-', label='D' if i == 0 else "")

    plt.title("Original Data (X), Reconstruction (rec), Dictionary Atoms (D), num epochs: " + str(num_epochs))
    plt.legend()
    plt.grid(True)
    plt.show()


def WDL(X, D, C, reg, reg_m, num_epochs = 100):
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


    for epoch in range(num_epochs):
        optimizer.zero_grad()

        w = torch.nn.functional.softmax(weights, dim = 0)
        # 1. Reconstruct via barycenter
        p = UOT_barycenter_batched(D, C, reg, reg_m, w)
        # 2. Compute loss
        loss = (sinkhorn_unbalanced_batched(X, p, C, reg, reg_m) - base_cost).abs().sum()/weight_length
        if loss.isnan():
            print("Loss is nan on epoch:" + str(epoch))
            exit()
        # 3. Backprop and update
        loss.backward()
        optimizer.step()
        D.data = D.data.clamp(min = 1e-8)
        if epoch%25 == 0:
            print("Loss on iteration:", epoch, loss)

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
    return rec, D


C = Cost(1) #Cost matrix
data = data = data_loader('data')
(train_data, lst, train_classes) = sample(data, size = 100, mode = 'train_classes', n_labels=6, label_hard=[], balanced = False)
train_data = train_data.astype(np.float64)
X = train_data
X = torch.from_numpy(X).T #Base data set
X = X + 1e-15


# Sum over rows (dim=0) to get total mass per column
col_masses = X.sum(dim=0)

# Indices of columns with max and min total mass
max_col_idx = torch.argmax(col_masses)
min_col_idx = torch.argmin(col_masses)
indices = torch.tensor([min_col_idx, max_col_idx])
#Random 
num_atoms = 2
random_indices = torch.randperm(X.size(1))[:num_atoms]
# Pluck out the columns
D = X[:, indices]

D.requires_grad_(True)


reg = .01
reg_m = 100
num_epochs = 500
rec, D = WDL(X, D, C, reg, reg_m, num_epochs=num_epochs)
plot_data(X.T, rec.T, D.T, num_epochs)



