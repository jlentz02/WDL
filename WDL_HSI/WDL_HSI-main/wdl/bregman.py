import math

import ot
import torch

from ot.bregman import convolutional_barycenter2d_debiased, convolutional_barycenter2d
from ot import  barycenter_unbalanced
from ot.unbalanced import sinkhorn_unbalanced2, sinkhorn_stabilized_unbalanced
from ot.lp import emd2


def OT(C: torch.Tensor,
       method: str = "bregman-stabilized",
       reg: float = 1.0,
       sharp: bool = False,
       maxiter: int = 5,
       tol: float = 1e-6,
       height: int = None,
       width: int = None,
       plan: bool = False,
       dev: torch.device = torch.device("cpu"),
       ):
    """
    Returns the a function that solves the OT problem and returns the loss depending on the mode used

    :param C:
    :param reg:
    :param sharp: whether or not to include the entropy term in the loss calculation
    :param plan: true if plan should be returned, otherwise returns the loss
    :return: OTsolver(a, b)
    """

    if method == "bregman-stabilized":
        return lambda a, b: bregmanStabilizedOT(a, b, C, reg, maxiter, tol, plan=plan, dev=dev)
    elif method == "bregman":
        K = torch.exp(-C / reg).to(dev)
        return lambda a, b: bregmanOT(a, b, K, reg, maxiter, tol, sharp=sharp, C=C, plan=plan, dev=dev)
    elif method == "conv":
        # form convolution matrices
        norm = max(height, width)
        normalized_height = height / norm
        normalized_width = width / norm

        t = torch.linspace(0, normalized_width, width)
        Kx = torch.exp(-torch.pow(t.view(-1, 1) - t.view(1, -1), 2) / reg).to(dev)

        t = torch.linspace(0, normalized_height, height)
        Ky = torch.exp(-torch.pow(t.view(-1, 1) - t.view(1, -1), 2) / reg).to(dev)
        return lambda a, b: convolutionalOT(a, b, reg, height, width, Kx, Ky, maxiter, tol, plan=plan, dev=dev)
    elif method == "lp":
        # define helpful function
        def f(a, b, C):
            # deal with floating point rounding errors
            a = a.type(torch.float64)
            b = b.type(torch.float64)
            torch.divide(a, a.sum(dim=0), out=a)
            torch.divide(b, b.sum(dim=0), out=b) 

            # contiguous memory layout needed for emd2 lp solver
            return emd2(a.contiguous(), b.contiguous(), C)

        return lambda a, b: f(a, b, C)
    #For use in unbalanced OT. Solves the analogous unbalanced problem to bregman stabilized as above and returns the loss
    #reg_m is the marginal relaxation term usually tau
    elif method == "bregman_stabilized_unbalanced":
        return lambda a, b: sinkhorn_unbalanced_batched(a, b, C = C, reg = reg, reg_m = .1)
    elif method == "wasserstein_kmeans_unbalanced":
        return lambda a, b: sinkhorn_unbalanced(a, b, C = C, reg = reg, reg_m = .1)
    else:
        raise NotImplementedError(f"OT method {method} not implemented.")
    
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

    a = a.view(-1)
    b = b.view(-1)

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


def convolutionalOT(a: torch.Tensor,
                    b: torch.Tensor,
                    reg: float,
                    height: int,
                    width: int,
                    Kx: torch.Tensor,
                    Ky: torch.Tensor,
                    maxiter: int = 5,
                    tol: float = 1e-6,
                    plan: bool = False,
                    dev: torch.device = torch.device("cpu"), ):
    """
    convolutional wasserstein for images - normalizes height and width by max(height/width)

    :param a:
    :param b:
    :param K:
    :param reg:
    :param height:
    :param width:
    :param maxiter:
    :param tol:
    :param dev:
    :return:
    """

    if plan:
        raise NotImplementedError

    # convolution operator
    # applies Kx where x = vec(im) and K is the full kernel matrix as in the bregman iterations
    def conv(im):
        return torch.matmul(Ky, torch.matmul(im, Kx))

    v = torch.ones((height, width), device=dev)

    if len(a.shape) == 1:
        a = a.view(-1, 1)
    a_size = a.shape[0]
    a_n_hists = a.shape[1]

    # bregman convolutions
    b_size = b.shape[0]
    b_n_hists = b.shape[1]

    if a_size != b_size:
        raise ValueError("Dimension mismatch of a: {a_size} and b: {b_size}. Must match for convolutional OT.")

    if a_n_hists > 1:
        if a_n_hists != b_n_hists:
            raise ValueError("If number of histograms in a is greater than 1,"
                             " then it must equal the number of histograms in b.")
        a_im = a.mT.view(a_n_hists, height, width)
    else:
        a_im = a.view(height, width)

    b_im = b.mT.view(b_n_hists, height, width)

    for i in range(maxiter):
        u = torch.divide(a_im, conv(v))
        v = torch.divide(b_im, conv(u))

    # reshape dual variables for gradient and return cost
    if a_n_hists == 1:
        f, g = reg * torch.clip(u.log().view(b_n_hists, -1).mT, min=-1e2), reg * torch.clip(
            v.log().view(b_n_hists, -1).mT,
            min=-1e2)
        # deal with zero mass situations

        return torch.matmul(f.mT, a).view(b_n_hists) + torch.bmm(g.mT.view(b_n_hists, 1, b_size),
                                                                 b.mT.view(b_n_hists, b_size, 1)).view(b_n_hists)
    else:
        f, g = reg * torch.clip(u.log().view(a_n_hists, -1).mT, min=-1e2), reg * torch.clip(
            v.log().view(b_n_hists, -1).mT,
            min=-1e2)
        # return cost vector
        return torch.bmm(f.mT.view(a_n_hists, 1, a_size), a.mT.view(a_n_hists, a_size, 1)).view(a_n_hists) + torch.bmm(
            g.mT.view(b_n_hists, 1, b_size),
            b.mT.view(b_n_hists, b_size, 1)).view(b_n_hists)


def bregmanOT(a: torch.Tensor,
              b: torch.Tensor,
              K: torch.Tensor,
              reg: float,
              maxiter: int = 5,
              tol: float = 1e-6,
              sharp: bool = False,
              C: torch.Tensor = None,
              plan: bool = False,
              dev: torch.device = torch.device("cpu"),
              ):
    """
    The unstabilized sinkhorn iterations computing transport between distributions a and b

    :param a: (d1 x 1) source distribution
    :param b: (d2 x 1) target distribution
    :param K: the Kernel matrix
    :param reg: entropic regularization parameter
    :param maxiter: maximum number of sinkhorn iterations to do
    :param tol: tolerance for residual error
    :param sharp: whether or not to return the "sharp" transport cost (sharp means no entropy term in transport cost)

    :return: the entropic loss value of the OT problem
    """
    # rehshape a
    if len(a.shape) == 1:
        a = a.view(-1, 1)

    if len(b.shape) == 1:
        b = b.view(-1, 1)

    n_a = a.shape[1]
    n_b = b.shape[1]


    if n_a > 1 and n_a != n_b:
        raise ValueError("If a has more than 1 distribution, then it must have an equal number of distributions as b.")

    # ignore 0s in distributions if only comparing between 2 distributions
    if n_a == 1 and n_b == 1:
        a_non_zero = a != 0
        b_non_zero = b != 0
        a = a[a_non_zero].view(-1, 1)
        b = b[b_non_zero].view(-1, 1)
        b_size = b.shape[0]
        K = K[a_non_zero.view(-1)][:, b_non_zero.view(-1)]
        C = C[a_non_zero.view(-1)][:, b_non_zero.view(-1)]

        n_hists = 1
    elif n_a == 1:
        a_non_zero = a != 0
        a = a[a_non_zero].view(-1, 1)
        K = K[a_non_zero.view(-1)]
        C = C[a_non_zero.view(-1)]
        b_size = b.shape[0]
        n_hists = b.shape[1]
    else:
        n_hists = n_a

    # initialize variable
    v = torch.ones_like(b, device=dev)

    # turn of backprop calculation since we know the gradients
    if not sharp and not plan:
        torch.autograd.set_grad_enabled(False)
    for i in range(maxiter):
        u = torch.div(a, torch.matmul(K, v))
        v = torch.div(b, torch.matmul(K.mT, u))
    # turn back on backprop for end computation
    torch.autograd.set_grad_enabled(True)
    if plan:
        if n_hists == 1:
            return u.view(-1, 1) * K * v.view(1, -1)
        else:
            raise NotImplementedError
    if sharp:
        # requires C as a parameter
        return torch.einsum('ik,ij,jk,ij->k', u.view(-1, 1), K, v.view(-1, 1), C)
    else:
        f, g = reg * u.log(), reg * v.log()
        # fix neg infinity issues
        f = torch.clip(f, min=-1e2)
        g = torch.clip(g, min=-1e2)

        if n_a == 1:
            return torch.matmul(f.mT, a).view(n_b) + torch.bmm(g.mT.view(n_b, 1, b_size),
                                                               b.mT.view(n_b, b_size, 1)).view(n_b)
        else:
            return torch.bmm(f.mT.view(n_a, 1, -1), a.mT.view(n_a, -1, 1)).view(n_hists) + torch.bmm(
                g.mT.view(n_hists, 1, -1),
                b.mT.view(n_hists, -1, 1)).view(n_hists)


def bregmanStabilizedOT(a: torch.Tensor,
                        b: torch.Tensor,
                        C: torch.Tensor,
                        reg: float,
                        maxiter: int = 5,
                        tol: float = 1e-6,
                        plan: bool = False,
                        dev: torch.device = torch.device("cpu"),
                        ):
    """
    unstabilized bregman iterations for

    :param a:
    :param b:
    :param C:
    :return: a function that has the kernel precomputed and solves the OT problem
    """

    if plan:
        raise NotImplementedError
    # remove nonzeros
    a_non_zero = a != 0
    b_non_zero = b != 0
    a = a[a_non_zero]
    b = b[b_non_zero]
    C = C[a_non_zero][:, b_non_zero]

    # initialize variables:
    f = torch.zeros_like(a, device=dev)
    g = torch.zeros_like(b, device=dev)

    log_a = a.log()
    log_b = b.log()

    # for picking the right dimension to sum over, 0 if a/b is size 1 respectively
    f_dim = min(1, a.shape[0] - 1)
    g_dim = min(1, b.shape[0] - 1)

    # stabilized bregman iterations
    # (as in Jean Feydy's PhD thesis)

    # assuming convergence we disable the auto grad for a speedup
    torch.autograd.set_grad_enabled(False)
    for i in range(maxiter):
        f = reg * (log_a - torch.logsumexp((g.mT - C) / reg, dim=f_dim))
        g = reg * (log_b - torch.logsumexp((f.view(-1, 1) - C).mT / reg, dim=g_dim))

    # the entropic loss
    torch.autograd.set_grad_enabled(True)
    return f.view(-1).dot(a) + g.view(-1).dot(b)


def barycenter(C: torch.Tensor,
               method: str = "bregman",
               reg: float = 1.0,
               reg_m: float = 100000,
               maxiter: int = 5,
               maxsinkiter: int = 7,
               dev: torch.device = torch.device("cpu"),
               height: int = None,
               width: int = None,
               ):
    """
    Returns the specified barycenter method (such that the returned method takes in a dictionary and weights
    as the only arguments

    :param C: the (d x d) ground cost matrix
    :param method: the method for solving the barycenter problem
    :param reg: the entropic regularization parameter
    :param maxiter:  the maximum number of iterations for an iterative solver to use
    :return
    """

    if method == "bregman-stabilized":
        # stabilized solver using bregman iterations
        return lambda D, w: bregmanStabilizedBary(D, w, C, reg, maxiter, dev)
    elif method == "bregman":
        # unstabilized classical bregman iterations
        K = torch.exp(-C / reg).to(dev)
        return lambda D, w: bregmanBary(D, w, K, reg, maxiter, dev)
    elif method == "conv":
        # requires data to be 2D
        # form convolution matrices
        norm = max(height, width)
        normalized_height = height / norm
        normalized_width = width / norm

        t = torch.linspace(0, normalized_width, width)
        Kx = torch.exp(-torch.pow(t.view(-1, 1) - t.view(1, -1), 2) / reg).to(dev)

        t = torch.linspace(0, normalized_height, height)
        Ky = torch.exp(-torch.pow(t.view(-1, 1) - t.view(1, -1), 2) / reg).to(dev)
        return lambda D, w: convolutionalBary(D, w, height, width, Kx, Ky, reg, maxiter, dev)
    elif method == "conv-pot":
        # requires data to be 2D
        def convBary(D, w):
            p = convolutional_barycenter2d(D.mT.view(len(w), height, width), reg=reg,
                                           weights=w.view(-1, ), numItermax=maxiter,
                                           stopThr=0.0, warn=False)
            return p.view(-1, 1)

        return lambda D, w: convBary(D, w)
    elif method == "conv-debiased-pot":
        # requires data to be 2D
        def convBary(D, w):
            p = convolutional_barycenter2d_debiased(D.mT.view(len(w), height, width), reg=reg,
                                                    weights=w.view(-1, ), numItermax=maxiter,
                                                    stopThr=0.0, warn=False)
            return p.view(-1, 1)

        return lambda D, w: convBary(D, w)
    elif method == "sharp":
        K = torch.exp(-C / reg).to(dev)
        OTfunc = lambda a, b: bregmanOT(a, b, K, reg, C=C, sharp=True, maxiter=maxsinkiter, dev=dev)
        return lambda D, w: sharpBarycenter(D, w, OTfunc, maxiter=maxiter, dev=dev)
    
    #Method added by me for, hopefully, doing UOT barycenters
    elif method == "barycenter_unbalanced":
        def UOTBary(D, w):
            p = UOT_barycenter_batched(D, C = C, reg_m = reg_m, reg=reg, numItermax=maxiter, weights =w)
            return p
        return lambda D, w: UOTBary(D,w)
    else:
        raise NotImplementedError(f"No barycenter method matches \"{method}\"")


def convolutionalBary(D: torch.Tensor,
                      weights: torch.Tensor,
                      height: int,
                      width: int,
                      Kx: torch.Tensor,
                      Ky: torch.Tensor,
                      reg: float = 1.0,
                      maxiter: int = 5,
                      dev: torch.device = torch.device("cpu"),
                      ):
    if len(weights.shape) == 1:
        weights = weights.view(-1, 1)
    n_barys = weights.shape[1]
    b = torch.ones((n_barys, D.shape[0], D.shape[1]), device=dev)

    def conv(im):
        return torch.matmul(Ky, torch.matmul(im, Kx))

    n_hists = D.shape[1]

    b_im = b.mT.view(n_barys, n_hists, height, width)

    D_im = D.mT.view(n_hists, height, width)

    # bregman projection loop (as in WDL paper) as convolutions
    for i in range(maxiter):
        phi = conv(torch.div(D_im, conv(b_im)))
        p = torch.bmm(phi.log().view(n_barys, n_hists, -1).mT,
                      weights.mT.view(weights.shape[1], weights.shape[0], 1)).exp().view(
            n_barys, height, width)
        b_im = torch.div(p.view(n_barys, 1, height, width), phi)

    return p.view(n_barys, -1).mT


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
    b = torch.ones((n_barys, D.shape[0], D.shape[1]), device=dev)
    Kt = K.mT

    # bregman projection loop (as in WDL paper)
    for i in range(maxiter):
        phi = torch.matmul(Kt, torch.div(D, torch.matmul(K, b)))
        p = torch.bmm(phi.log().view(n_barys, -1, n_hists),
                      weights.mT.view(n_barys, n_hists, 1)).exp()
        # p = torch.matmul(phi.log(), weights).exp()
        b = torch.div(p, phi)
    
    return p.view(n_barys, -1).mT


#A - n x d tensor of d dictionary atoms
#C - n x n cost matrix
#reg - entropic regularization term
#reg_m - marginal relaxation term (also known as tau)
#weights - d x m tensor of weights where m is the number of barycenters to be returned
#TODO Decide if weights should be d x m or m x d and then remove transpositions if necessary
def UOT_barycenter_batched(A, C, reg, reg_m, weights, numItermax = 100):
    #Establishing how big things are
    dim, n_hists = A.shape
    n_barys = weights.shape[1]
    
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


def bregmanStabilizedBary(D: torch.Tensor,
                          weights: torch.tensor,
                          C: torch.Tensor,
                          reg: float = 1.0,
                          maxiter: int = 5,
                          dev: torch.device = torch.device("cpu"),
                          ):
    """
    :param D: the (d x n) tensor of histograms to form the barycenter between
    :param weights: the (d) dim tensor of weights on each histogram (needs to sum to 1)
    :param C: the (d x d) ground cost matrix
    :param reg: the entropic regularization parameter
    :param maxiter:  the maximum number of iterations for an iterative solver to use
    :return:
    """

    # init variables
    n = D.shape[1]
    d = D.shape[0]
    log_D = D.log()
    u = torch.zeros_like(D, device=dev)
    v = torch.zeros_like(D, device=dev)
    log_P = torch.zeros(d, device=dev)
    phi = torch.zeros_like(u, device=dev)
    nonzero_idxs = D != 0

    # verify weights sum to 1 (algorithm would change if they didnt, but result stays the same)
    weights /= weights.sum()

    # bregman iterations
    # (as in the Wasserstein Dictionary Learning Paper - Schmitz et al.)
    u[~nonzero_idxs] = -math.inf
    for _ in range(maxiter):
        for j in range(n):
            u[nonzero_idxs[:, j], j] = \
                reg * (log_D[nonzero_idxs[:, j], j]
                       - torch.logsumexp((v[:, j].mT - C[nonzero_idxs[:, j], :]) / reg, dim=0)) \
                + u[nonzero_idxs[:, j], j]
            phi[:, j] = torch.logsumexp((u[:, j] - C[nonzero_idxs[:, j], :]) / reg, dim=1)
            v[:, j] = reg * (log_P - phi[:, j]) + v[:, j]
        log_P = torch.matmul(phi, weights)

    return log_P.exp()


def sharpBarycenter(D,
                    weights,
                    OTfunc,
                    maxiter,
                    dev: torch.device = torch.device("cpu"), ):
    # setup initial variables
    d = D.shape[0]
    a_t = torch.ones((d, 1)) / d
    a_h = a_t
    t = 1
    t0 = 10

    grads = torch.zeros_like(D)

    for i in range(maxiter):
        beta_inv = 2 / (t + 1)
        a = (1 - beta_inv) * a_h + beta_inv * a_t
        a.requires_grad = True

        # solve each OT problem and get the gradient via auto-diff
        for j in range(len(weights)):
            a.grad = None
            otdist = OTfunc(a, D[:, j])
            otdist.backward()
            grads[:, j] = a.grad.view(-1)

        alpha = grads.mm(weights)
        a_t = torch.multiply(a_t, torch.exp(-t0 * alpha / beta_inv))
        a_t /= a_t.sum()

        a_h = (1 - beta_inv) * a_h + beta_inv * a_t

    return a


def w2helper(a, b, C):
    """
    computes the w2 distances using LP solver and removes nonzero elements from distributions first to speed up computations
    :param a:
    :param b:
    :param C:
    :return:
    """
    asel = a != 0.0
    bsel = b != 0.0

    a = a.to(torch.float64)
    b = b.to(torch.float64)

    a = a[asel] / a[asel].sum()
    b = b[bsel] / b[bsel].sum()

    return ot.lp.emd2(a, b, C[asel.view(-1)][:, bsel.view(-1)])