import numpy as np
import matplotlib.pyplot as plt
import ot
import matplotlib.cm as cm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import ot.unbalanced as otu
from scipy.optimize import brentq


def make_uot_bary_example():

    # --- Step 1: Define the grid ---
    x = np.linspace(-5, 10, 500)

    # --- Step 2: Define two Gaussian distributions with different means and different masses ---
    def gaussian(x, mean, sigma, mass=1.0):
        g = np.exp(-0.5 * ((x - mean)/sigma)**2)
        g /= g.sum()  # normalize to 1
        g *= mass     # scale to desired total mass
        return g

    mu1 = gaussian(x, mean=0, sigma=1, mass=1.0)
    mu2 = gaussian(x, mean=5, sigma=1, mass=1)

    # --- Step 3: Cost matrix (squared Euclidean distance) ---
    X = x.reshape((-1, 1))
    M = ot.dist(X, X, metric='sqeuclidean')
    M /= M.max()  # normalize cost for stability

    # --- Step 4: Compute unbalanced barycenters for a range of weights ---
    lambdas = np.linspace(0, 1, 11)

    barycenters = []
    for lam in lambdas:
        weights = np.array([lam, 1-lam])
        bary = ot.unbalanced.barycenter_unbalanced(
            np.transpose([mu1, mu2]), M, reg=0.001, reg_m=.5, weights=weights
        )
        barycenters.append((lam, bary))  # always store both lam and bary

    # --- Step 5: Plot with colors blending between the two Gaussians ---
    plt.figure(figsize=(10,6))
    plt.plot(x, mu1, color=(0.267004, 0.004874, 0.329415, 1.0), lw=2)
    plt.plot(x, mu2, color=(0.993248, 0.906157, 0.143936, 1.0), lw=2)

    # Create a colormap that blends from blue to red
    cmap = cm.get_cmap("viridis")

    for lam, bary in barycenters:
        color = cmap(0.9-lam)  # lam=1 → blue side, lam=0 → red side
        plt.plot(x, bary, color=color, lw=2, alpha = .5)

    plt.title("Unbalanced OT Barycenters between Two Gaussians")
    plt.xlabel("x")
    plt.ylabel("density / mass")
    plt.show()

def make_ot_bary_example():

    # --- Step 1: Define the grid ---
    x = np.linspace(-5, 10, 500)

    # --- Step 2: Define two Gaussian distributions with different means and different masses ---
    def gaussian(x, mean, sigma, mass=1.0):
        g = np.exp(-0.5 * ((x - mean)/sigma)**2)
        g /= g.sum()  # normalize to 1
        g *= mass     # scale to desired total mass
        return g

    mu1 = gaussian(x, mean=0, sigma=1, mass=1.0)
    mu2 = gaussian(x, mean=5, sigma=1, mass=1)

    # --- Step 3: Cost matrix (squared Euclidean distance) ---
    X = x.reshape((-1, 1))
    M = ot.dist(X, X, metric='sqeuclidean')
    M /= M.max()  # normalize cost for stability

    # --- Step 4: Compute unbalanced barycenters for a range of weights ---
    lambdas = np.linspace(0, 1, 11)

    barycenters = []
    for lam in lambdas:
        weights = np.array([lam, 1-lam])
        bary = ot.barycenter(
            np.transpose([mu1, mu2]), M, reg=0.0005, weights=weights
        )
        barycenters.append((lam, bary))  # always store both lam and bary

    # --- Step 5: Plot with colors blending between the two Gaussians ---
    plt.figure(figsize=(10,6))
    plt.plot(x, mu1, color=(0.267004, 0.004874, 0.329415, 1.0), lw=2)
    plt.plot(x, mu2, color=(0.993248, 0.906157, 0.143936, 1.0), lw=2)

    # Create a colormap that blends from blue to red
    cmap = cm.get_cmap("viridis")

    for lam, bary in barycenters:
        color = cmap(0.9-lam)  # lam=1 → blue side, lam=0 → red side
        plt.plot(x, bary, color=color, lw=2, alpha = .5)

    plt.title("Unbalanced OT Barycenters between Two Gaussians")
    plt.xlabel("x")
    plt.ylabel("density / mass")
    plt.show()

def compare_ot_uot_bary():
    # --- Step 1: Define the grid ---
    x = np.linspace(-5, 10, 500)

    # --- Step 2: Define two Gaussian distributions ---
    def gaussian(x, mean, sigma, mass=1.0):
        g = np.exp(-0.5 * ((x - mean)/sigma)**2)
        g /= g.sum()  # normalize to 1
        g *= mass     # scale to desired total mass
        return g

    mu1 = gaussian(x, mean=0, sigma=1, mass=1.0)
    mu2 = gaussian(x, mean=5, sigma=1, mass=1.0)

    # --- Step 3: Cost matrix ---
    X = x.reshape((-1, 1))
    M = ot.dist(X, X, metric='sqeuclidean')
    M /= M.max()  # normalize cost

    # --- Balanced OT barycenter (explicit Gaussian formula) ---
    def barycenter_balanced_nonentropic(means, stds, weights, x):
        m_bar = np.sum(weights * means)
        s_bar = np.sum(weights * stds)
        g = np.exp(-0.5 * ((x - m_bar)/s_bar)**2)
        g /= g.sum()
        return g

    # Parameters of input Gaussians (needed for closed form)
    means = np.array([0.0, 5.0])
    stds = np.array([1.0, 1.0])

    # --- Step 4: Compute barycenters ---
    lambdas = np.linspace(0, 1, 11)
    bary_ot, bary_uot = [], []

    for lam in lambdas:
        weights = np.array([lam, 1-lam])

        # Balanced OT barycenter (closed form)
        b_ot = barycenter_balanced_nonentropic(means, stds, weights, x)
        bary_ot.append((lam, b_ot))

        # Skip unbalanced endpoints (λ = 0 or 1)
        if lam > 0 and lam < 1:
            b_uot = otu.barycenter_unbalanced(
                np.transpose([mu1, mu2]), M,
                reg=0.001, reg_m=0.5, weights=weights
            )
            bary_uot.append((lam, b_uot))

    # --- Step 5: Plot adjacent ---
    plt.figure(figsize=(12,6))

    cmap = cm.get_cmap("viridis")

    # OT barycenters (left, original x)
    plt.plot(x, mu1, color=(0.267004, 0.004874, 0.329415, 1.0), alpha=0.7)
    plt.plot(x, mu2, color=(0.993248, 0.906157, 0.143936, 1.0), alpha=0.7)
    for lam, b in bary_ot:
        color = cmap(0.9-lam)
        plt.plot(x, b, color=color, lw=2, alpha=0.7)

    # UOT barycenters (shifted along x)
    shift = 16  # horizontal shift
    plt.plot(x + shift, mu1, color=(0.267004, 0.004874, 0.329415, 1.0), alpha=0.7)
    plt.plot(x + shift, mu2, color=(0.993248, 0.906157, 0.143936, 1.0), alpha=0.7)
    for lam, b in bary_uot:
        color = cmap(0.9-lam)
        plt.plot(x + shift, b, color=color, lw=2, alpha=0.7)

    plt.axvline(x=10.5, color="black", linewidth=1.5)

    plt.title("OT vs UOT Barycenters Between Two Gaussians")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.show()

compare_ot_uot_bary()