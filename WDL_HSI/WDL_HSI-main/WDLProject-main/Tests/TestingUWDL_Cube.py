import torch
import ot
from torch.optim import Adam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def make_cost_matrix(X):
    n_hists = X.shape[1]
    C = torch.zeros(n_hists, n_hists)
    for i in range(0, n_hists):
        for j in range(0, n_hists):
            C[i,j] = torch.norm(X[:,i] - X[:,j], p = 2)

    return C

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

def plot_3d_samples(samples1, samples2=None, samples3 = None, title="3D Sample Plot", color1='blue', size1=30, color2='red', size2=30):
    """
    Plots one or two sets of 3D points using matplotlib.

    Args:
        samples1 (Tensor): Primary set of 3D points (N, 3).
        title (str): Plot title.
        color1 (str): Color of the first set.
        size1 (int): Marker size for the first set.
        samples2 (Tensor, optional): Second set of 3D points (M, 3).
        color2 (str): Color of the second set.
        size2 (int): Marker size for the second set.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    print(samples1.size(), samples2.size())

    # Plot first set
    ax.scatter(samples1[:, 0], samples1[:, 1], samples1[:, 2], color=color1, s=size1, label='Original Data')

    # Optionally plot second set
    if samples2 is not None:
        samples2 = samples2.detach().clone()
        ax.scatter(samples2[:, 0], samples2[:, 1], samples2[:, 2], color=color2, s=size2, label='Reconstruction')

    # Optionally plot dictionary atoms
    if samples3 is not None:
        samples3 = samples3.detach().clone()
        ax.scatter(samples3[:, 0], samples3[:, 1], samples3[:, 2], color="green", s=size2, label='Dictionary Atoms')

    # Set axis limits and labels
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

#Sampling from hull of unit cube with bottom left point (1,1,1)
N = 500  # number of samples

samples = []
shift = .5
for _ in range(N):
    point = torch.rand(3)  # in [0, 1)^3
    face = torch.randint(0, 3, (1,)).item()  # choose x/y/z to fix
    side = torch.randint(0, 2, (1,)).item()  # choose 0 or 1 -> maps to 1 or 2 after shift
    point[face] = side  # fix one coordinate
    samples.append(point + shift)  # shift cube to [1,2]^3

samples = torch.stack(samples).T

D = torch.tensor([[0, 0, 0], [0,0,1], [0,1,0], [1,0,0], [0,1,1], [1,1,0], [1,0,1], [1,1,1]], dtype = torch.double, requires_grad= True).T + shift
C = torch.tensor([[0, 1, 2], [1, 0, 1], [2,1,0]], dtype = torch.double)

reg = .01
reg_m = 50

rec, D = WDL(samples, D, C, reg, reg_m, num_epochs=250)
plot_3d_samples(samples.T, rec.T, D.T)











        

    
