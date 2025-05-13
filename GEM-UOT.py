import numpy as np
import numdifftools as nd
from scipy.optimize import minimize, Bounds

k = 10
#a,b are n dimensional vectors of masses, epsilon, tau, and eta are tuning parameters
def GEMUOT(cost_matrix, a, b, n, epsilon, tau, eta):
    #Initialize arrays of x, y, x underscore, y underscore
    x = [0 for i in range(k)]
    x[0] = np.zeros(2*n+n^2)
    x_und = [0 for i in range(k)]
    x_und[0] = np.zeros(2*n+n^2)
    y = 0 #y^t-1
    y_prev = 0 #y^t-2

    #Initialize important constants
    D = compute_D(cost_matrix, a, b, tau, eta)
    L = compute_L(a, b, tau, D)
    mu = compute_mu(a, b, D, tau, eta)
    zeta = compute_zeta(L, mu)
    psi = 1/(1-zeta) - 1
    rho = (zeta*mu)/(1-zeta)

    #core loop
    for t in range(1, k+1):
        y_tilde = y + zeta(y - y_prev)
        x[k] = compute_M(y_tilde, x[k-1], rho)
        x_und[k] = (1/(1+psi))*(x[k] + psi*x_und[k-1])
        





#Computes L
def compute_L(a, b, tau, D):
    #Initialize variables
    alpha = a.sum()
    beta = b.sum()
    a_min = a.min()
    b_min = b.min()

    #Compute and return
    L = (alpha+beta)/(2*tau) + min(a_min, b_min)/tau*np.exp(-D/tau)
    return L

def compute_mu(a, b, D, tau, eta):
    #Initialize variables
    a_min = a.min()
    b_min = b.min()

    #Compute and return
    mu = min(min(a_min, b_min)/tau*np.exp(-D/tau), 1/(2*eta))
    return mu

def compute_zeta(L, mu):
    zeta = 1 - 1/(1 + np.sqrt(1+16*L/mu))
    return zeta

#Computes D which is used to compute L
def compute_D(cost_matrix, a, b, tau, eta):
    #Initialize variables
    C_inf = np.linalg.norm(cost_matrix, ord = np.inf)
    alpha = a.sum()
    beta = b.sum()
    a_min = a.min()
    b_min = b.min()

    #Compute and return
    D = C_inf + eta(alpha+beta) + tau*np.log((alpha+beta)/2) - tau*min(np.log(a_min), np.log(b_min))
    return D

def compute_w(x, eta, tau, n, a, b, D):
    u = x[:n]
    v = x[n:2*n]
    t = x[2*n:]
    a_min = a.min()
    b_min = b.min()

    w = min(a_min, b_min)/(2*tau)*np.exp(-D/tau)*(np.inner(u,u) + np.inner(v,v)) + 1/(4*eta)*np.inner(t,t)
    return w


def compute_f(x, tau, n, a, b, D):
    u = x[:n]
    v = x[n:2*n]
    t = x[2*n:]
    a_min = a.min()
    b_min = b.min()

    f = tau*np.inner(np.exp(-u/tau), a) + tau*np.inner(np.exp(-v/tau), b) - min(a_min, b_min)/(2*tau)*np.exp(-D/tau)*(np.inner(u,u) + np.inner(v,v))
    return f

def compute_P(x_0, x, eta, tau, n, a, b, D, mu):
    w_x = compute_w(x, eta, tau, n, a, b, D)
    w_x_0 = compute_w(x_0, eta, tau, n, a, b, D)

    def w(x):
        u = x[:n]
        v = x[n:2*n]
        t = x[2*n:]
        a_min = min(a)
        b_min = min(b)
        output = min(a_min, b_min)/(2*tau)*np.exp(-D/tau)*(np.inner(u,u) + np.inner(v,v)) + 1/(4*eta)*np.inner(t,t)
        return output

    grad_w = nd.Gradient(w)
    #Probably going to break
    grad_w_x_0 = grad_w(x_0)

    p = (1/mu)*(w_x - (w_x_0 + np.inner(grad_w_x_0, x - x_0)))
    return p

def h(x, g, x_0, theta, eta, tau, n, a, b, D, mu):
    output = np.inner(g, x) + compute_w(x, eta, tau, n, a, b, D) + theta*compute_P(x_0, x, eta, tau, n, a, b, D, mu)
    return output

def compute_bounds(x, n, a, b, tau, D):
    alpha = a.sum()
    beta = b.sum()
    u = x[:n]
    v = x[n:2*n]
    t = x[2*n:]
    #Bounds for u
    u_lower = tau*np.log(2*a/ (alpha+beta))
    u_upper = D*np.ones(n)

    #Bounds for v
    v_lower = tau*np.log(2*b/ (alpha+beta))
    v_upper = D*np.ones(n)

    #Bounds for t
    t_lower = -np.inf * np.ones(n * n)
    t_upper = np.inf * np.ones(n * n)

    # Combine all bounds
    lower_bounds = np.concatenate([u_lower, v_lower, t_lower])
    upper_bounds = np.concatenate([u_upper, v_upper, t_upper])

    return Bounds(lower_bounds, upper_bounds)

def compute_M(x, g, x_0, theta, eta, tau, n, a, b, D, mu):
    #gradient of h, the actual function we are minimizing over
    def h(x):
        output = np.inner(g, x) + compute_w(x, eta, tau, n, a, b, D) + theta*compute_P(x_0, x, eta, tau, n, a, b, D, mu)
        return output
    
    grad_h = nd.Gradient(h)

    #Initial guess for x
    initial_guess = np.concatenate([a, b, np.zeros(n^2)])

    #Bounds to minimize within
    bounds = compute_bounds(x, n, a, b, tau, D)

    #optimization
    result = minimize(h, initial_guess, jac = grad_h, bounds = bounds, method = 'L-BFGS-B')

    print(result)
    return result
