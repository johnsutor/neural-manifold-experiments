from collections import defaultdict
from functools import partial
from typing import List

import autograd.numpy as np
import pymanopt
import torch
import torch.nn as nn
from cvxopt import matrix, solvers
from pymanopt import Problem
from pymanopt.manifolds import Stiefel

# from pymanopt.solvers import ConjugateGradient
from pymanopt.optimizers.conjugate_gradient import ConjugateGradient
from scipy.linalg import qr
from torch.utils.data import Dataset
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from tqdm import tqdm

# Configure cvxopt solvers
solvers.options["show_progress"] = False
solvers.options["maxiters"] = 1000000
solvers.options["abstol"] = 1e-12
solvers.options["reltol"] = 1e-12
solvers.options["feastol"] = 1e-12


def make_manifold_data(
    dataset: Dataset,
    sampled_classes: int,
    examples_per_class: int,
    max_class: int = None,
):
    """
    Samples manifold data for use in later analysis

    Args:
        dataset: PyTorch style dataset, or iterable that contains (input, label) pairs
        sampled_classes: Number of classes to sample from (must be less than or equal to
            the number of classes in dataset)
        examples_per_class: Number of examples per class to draw (there should be at least
            this many examples per class in the dataset)
        max_class (optional): Maximum class to sample from. Defaults to sampled_classes if unspecified

    Returns:
        data: Iterable containing manifold input data
    """
    if max_class is None:
        max_class = sampled_classes
    assert sampled_classes <= max_class, "Not enough classes in the dataset"
    assert examples_per_class * max_class <= len(
        dataset
    ), "Not enough examples per class in dataset"

    # Storage for samples
    sampled_data = defaultdict(list)
    # Sample the labels
    sampled_labels = np.random.choice(
        list(range(max_class)), size=sampled_classes, replace=False
    )
    # Shuffle the order to iterate through the dataset
    idx = [i for i in range(len(dataset))]
    np.random.shuffle(idx)
    # Iterate through the dataset until enough samples are drawn
    for i in idx:
        sample, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()

        if label in sampled_labels and len(sampled_data[label]) < examples_per_class:
            sampled_data[label].append(sample)
        # Check if enough samples have been drawn
        complete = True
        for s in sampled_labels:
            if len(sampled_data[s]) < examples_per_class:
                complete = False
        if complete:
            break
    # Check that enough samples have been found
    assert complete, "Could not find enough examples for the sampled classes"
    # Combine the samples into batches
    data = []
    for s, d in sampled_data.items():
        data.append(torch.cat(d))
    return data


def get_feature_extractor(model, return_nodes):
    """
    Creates a feature extractor from a model

    Args:
        model: PyTorch model to extract features from
        return_nodes: List of nodes to return activations for

    Returns:
        feature_extractor: PyTorch module that extracts features from the model
    """
    # Create the feature extractor
    assert set(
        return_nodes
    ).issubset(
        get_graph_node_names(model)[1]
    ), f"Invalid node names, available nodes are:\n{','.join(get_graph_node_names(model)[1])}, \
        but requested nodes are:\n{','.join(return_nodes)}"

    model = model.eval()
    feature_extractor = create_feature_extractor(model, return_nodes=list(return_nodes))
    return feature_extractor


def extract_activations(dataset: List[torch.tensor], module: nn.Module):
    """
    Extracts the activations from a PyTorch feature extractor module

    Args:
        data: Iterable containing manifold input data
        module: PyTorch module to compute the statistics for
        (must be a feature extractor that returns a dictionary of layer activations, i.e. {'layer_name': activations})

    Returns:
        activations: Dictionary containing the activations for each node
    """
    # Compute the activations
    activations = defaultdict(list)
    device = next(
        module.parameters()
    ).device  # TODO: Not the best, this will screw up multi-GPU
    for data in dataset:
        # Flatten the data, if necessary
        if data.dim() > 4:
            B, T, C, H, W = data.shape
            data = data.view(B * T, C, H, W)
        acts = module.forward(data.to(device))
        for key in acts:
            activations[key].append(
                acts[key].reshape(acts[key].shape[0], -1).T.detach().cpu().numpy()
            )

    return activations


def manifold_analysis(activations: dict, compute=["full", "ellipsoid"]):
    """
    Computes the manifold statistics for a set of activations

    Args:
        activations: Dictionary containing the activations for each node

    Returns:
        statistics: Dictionary containing the manifold statistics for each node
    """
    # Compute the statistics
    statistics = {}
    for key in tqdm(activations.keys(), desc="Computing manifold statistics"):
        manifold_stats = manifold_analysis_corr(
            activations[key], 0, 300, n_reps=1, compute=compute
        )

        # Update the statistics dictionary structure
        # i.e. from statistics['node_name']['statistic_name']
        # to statistics['statistic_name']['node_name']
        for subkey, value in manifold_stats.items():
            if subkey not in statistics:
                statistics[subkey] = {}
            statistics[subkey][key] = value

    return statistics


def manifold_analysis_corr(
    XtotT,
    kappa,
    n_t,
    t_vecs=None,
    n_reps=10,
    projection=5000,
    compute=["full", "ellipsoid"],
):
    """
    Carry out the analysis on multiple manifolds.

    Args:
        XtotT: Sequence of 2D arrays of shape (N, P_i) where N is the dimensionality
                of the space, and P_i is the number of sampled points for the i_th manifold.
        kappa: Margin size to use in the analysis (scalar, kappa > 0)
        n_t: Number of gaussian vectors to sample per manifold
        t_vecs: Optional sequence of 2D arrays of shape (Dm_i, n_t) where Dm_i is the reduced
                dimensionality of the i_th manifold. Contains the gaussian vectors to be used in
                analysis.  If not supplied, they will be randomly sampled for each manifold.
        n_reps: Number of repetitions to use in the analysis
        projection: Dimension to project the data into before carrying out the analysis
        compute: List of strings containing the types of analysis to carry out.  Can contain


    Returns:
        a_Mfull_vec: 1D array containing the capacity calculated from each manifold
        R_M_vec: 1D array containing the calculated anchor radius of each manifold
        D_M_vec: 1D array containing the calculated anchor dimension of each manifold.
        res_coeff0: Residual correlation
        KK: Dimensionality of low rank structure
    """
    # Number of manifolds to analyze
    num_manifolds = len(XtotT)

    N = XtotT[0].shape[0]
    # If N is greater than projection, do the random projection to projection features
    if N > projection:
        M = np.random.randn(projection, N)
        M /= np.sqrt(np.sum(M * M, axis=1, keepdims=True))
        XtotT = [np.matmul(M, d) for d in XtotT]

    # Project
    # Compute the global mean over all samples
    Xori = np.concatenate(XtotT, axis=1)  # Shape (N, sum_i P_i)
    X_origin = np.mean(Xori, axis=1, keepdims=True)

    # Subtract the mean from each manifold
    Xtot0 = [XtotT[i] - X_origin for i in range(num_manifolds)]
    # Compute the mean for each manifold
    centers = [np.mean(XtotT[i], axis=1) for i in range(num_manifolds)]
    centers = np.stack(centers, axis=1)  # Centers is of shape (N, m) for m manifolds
    center_mean = np.mean(centers, axis=1, keepdims=True)  # (N, 1) mean of all centers

    # Center correlation analysis
    _, SS, _ = np.linalg.svd(centers - center_mean)
    # Compute the max K
    total = np.cumsum(np.square(SS) / np.sum(np.square(SS)))
    maxK = np.argmax([t if t < 0.95 else 0 for t in total]) + 11

    statistics = {}

    if "ellipsoid" in compute:
        sum_sq = abs(SS).sum() ** 2
        # Computes the sum of the squared eigenvalues
        sq_sum = (SS**2).sum()

        D = sum_sq / sq_sum
        R = sq_sum**0.5
        alpha = SS.sum()
        statistics = {
            "D_ellipsoid": D,
            "R_ellipsoid": R,
            "alpha_ellipsoid": alpha,
            "singular_values": SS,
        }

    if "full" in compute:
        # Compute the low rank structure
        norm_coeff, norm_coeff_vec, Proj, V1_mat, res_coeff, res_coeff0 = fun_FA(
            centers, maxK, 20000, n_reps
        )
        _, KK = min(res_coeff), np.argmin(res_coeff) + 1

        # Compute projection vector into the low rank structure
        V11 = np.matmul(Proj, V1_mat[KK - 1])
        X_norms = []
        XtotInput = []
        for i in range(num_manifolds):
            Xr = Xtot0[i]
            # Project manifold data into null space of center subspace
            Xr_ns = Xr - np.matmul(V11, np.matmul(V11.T, Xr))
            # Compute mean of the data in the center null space
            Xr0_ns = np.mean(Xr_ns, axis=1)
            # Compute norm of the mean
            Xr0_ns_norm = np.linalg.norm(Xr0_ns)
            X_norms.append(Xr0_ns_norm)
            # Center normalize the data
            Xrr_ns = (Xr_ns - Xr0_ns.reshape(-1, 1)) / Xr0_ns_norm
            XtotInput.append(Xrr_ns)

        a_Mfull_vec = np.zeros(num_manifolds)
        R_M_vec = np.zeros(num_manifolds)
        D_M_vec = np.zeros(num_manifolds)
        # Make the D+1 dimensional data
        for i in range(num_manifolds):
            S_r = XtotInput[i]
            D, m = S_r.shape
            # Project the data onto a smaller subspace
            if D > m:
                Q, R = qr(S_r, mode="economic")
                S_r = np.matmul(Q.T, S_r)
                # Get the new sizes
                D, m = S_r.shape
            # Add the center dimension
            sD1 = np.concatenate([S_r, np.ones((1, m))], axis=0)

            # Carry out the analysis on the i_th manifold
            if t_vecs is not None:
                a_Mfull, R_M, D_M = each_manifold_analysis_D1(
                    sD1, kappa, n_t, t_vec=t_vecs[i]
                )
            else:
                a_Mfull, R_M, D_M = each_manifold_analysis_D1(sD1, kappa, n_t)

            # Store the results
            a_Mfull_vec[i] = a_Mfull
            R_M_vec[i] = R_M
            D_M_vec[i] = D_M

        statistics = {
            **statistics,
            "alpha_full": 1 / np.mean(1 / a_Mfull_vec),
            "R_full": np.mean(R_M_vec),
            "D_full": np.mean(D_M_vec),
            "res_coeff0": res_coeff0,
            "KK": KK,
        }
    return statistics


def each_manifold_analysis_D1(sD1, kappa, n_t, eps=1e-8, t_vec=None):
    """
    This function computes the manifold capacity a_Mfull, the manifold radius R_M, and manifold dimension D_M
    with margin kappa using n_t randomly sampled vectors for a single manifold defined by a set of points sD1.

    Args:
        sD1: 2D array of shape (D+1, m) where m is number of manifold points
        kappa: Margin size (scalar)
        n_t: Number of randomly sampled vectors to use
        eps: Minimal distance (default 1e-8)
        t_vec: Optional 2D array of shape (D+1, m) containing sampled t vectors to use in evaluation

    Returns:
        a_Mfull: Calculated capacity (scalar)
        R_M: Calculated radius (scalar)
        D_M: Calculated dimension (scalar)
    """
    # Get the dimensionality and number of manifold points
    D1, m = sD1.shape  # D+1 dimensional data
    D = D1 - 1
    # Sample n_t vectors from a D+1 dimensional standard normal distribution unless a set is given
    if t_vec is None:
        t_vec = np.random.randn(D1, n_t)
    # Find the corresponding manifold point for each random vector
    ss, gg = maxproj(t_vec, sD1)

    # Compute V, S~ for each random vector
    s_all = np.empty((D1, n_t))
    f_all = np.zeros(n_t)
    for i in range(n_t):
        # Get the t vector to use (keeping dimensions)
        t = np.expand_dims(t_vec[:, i], axis=1)
        if gg[i] + kappa < 0:
            # For this case, a solution with V = T is allowed by the constraints, so we don't need to
            # find it numerically
            v_f = t
            s_f = ss[:, i].reshape(-1, 1)
        else:
            # Get the solution for this t vector
            v_f, _, _, alpha, vminustsqk = minimize_vt_sq(t, sD1, kappa=kappa)
            f_all[i] = vminustsqk
            # If the solution vector is within eps of t, set them equal (interior point)
            if np.linalg.norm(v_f - t) < eps:
                v_f = t
                s_f = ss[:, i].reshape(-1, 1)
            else:
                # Otherwise, compute S~ from the solution
                scale = np.sum(alpha)
                s_f = (t - v_f) / scale
        # Store the calculated values
        s_all[:, i] = s_f[:, 0]

    # Compute the capacity from eq. 16, 17 in 2018 PRX paper.
    max_ts = np.maximum(np.sum(t_vec * s_all, axis=0) + kappa, np.zeros(n_t))
    s_sum = np.sum(np.square(s_all), axis=0)
    lamb = np.asarray([max_ts[i] / s_sum[i] if s_sum[i] > 0 else 0 for i in range(n_t)])
    slam = np.square(lamb) * s_sum
    a_Mfull = 1 / np.mean(slam)

    # Compute R_M from eq. 28 of the 2018 PRX paper
    ds0 = s_all - s_all.mean(axis=1, keepdims=True)
    ds = ds0[0:-1, :] / s_all[-1, :]
    ds_sq_sum = np.sum(np.square(ds), axis=0)
    R_M = np.sqrt(np.mean(ds_sq_sum))

    # Compute D_M from eq. 29 of the 2018 PRX paper
    t_norms = np.sum(np.square(t_vec[0:D, :]), axis=0, keepdims=True)
    t_hat_vec = t_vec[0:D, :] / np.sqrt(t_norms)
    s_norms = np.sum(np.square(s_all[0:D, :]), axis=0, keepdims=True)
    s_hat_vec = s_all[0:D, :] / np.sqrt(s_norms + 1e-12)
    ts_dot = np.sum(t_hat_vec * s_hat_vec, axis=0)

    D_M = D * np.square(np.mean(ts_dot))

    return a_Mfull, R_M, D_M


def maxproj(t_vec, sD1, sc=1):
    """
    This function finds the point on a manifold (defined by a set of points sD1) with the largest projection onto
    each individual t vector given by t_vec.

    Args:
        t_vec: 2D array of shape (D+1, n_t) where D+1 is the dimension of the linear space, and n_t is the number
            of sampled vectors
        sD1: 2D array of shape (D+1, m) where m is number of manifold points
        sc: Value for center dimension (scalar, default 1)

    Returns:
        s0: 2D array of shape (D+1, n_t) containing the points with maximum projection onto corresponding t vector.
        gt: 1D array of shape (D+1) containing the value of the maximum projection of manifold points projected
            onto the corresponding t vector.
    """
    # get the dimension and number of the t vectors
    D1, n_t = t_vec.shape
    D = D1 - 1
    # Get the number of samples for the manifold to be processed
    # m = sD1.shape[1]
    # For each of the t vectors, find the maximum projection onto manifold points
    # Ignore the last of the D+1 dimensions (center dimension)
    s0 = np.zeros((D1, n_t))
    gt = np.zeros(n_t)
    for i in range(n_t):
        t = t_vec[:, i]
        # Project t onto the SD vectors and find the S vector with the largest projection
        max_S = np.argmax(np.dot(t[0:D], sD1[0:D]))
        sr = sD1[0:D, max_S]
        # Append sc to this vector
        s0[:, i] = np.append(sr, [sc])
        # Compute the projection of this onto t
        gt[i] = np.dot(t, s0[:, i])
    return s0, gt


def minimize_vt_sq(t, sD1, kappa=0):
    """
    This function carries out the constrained minimization decribed in Sec IIIa of the 2018 PRX paper.
    Instead of minimizing F = ||V-T||^2, The actual function that is minimized will be
        F' = 0.5 * V^2 - T * V
    Which is related to F by F' = 0.5 * (F - T^2).  The solution is the same for both functions.

    This makes use of cvxopt.

    Args:
        t: A single T vector encoded as a 2D array of shape (D+1, 1)
        sD1: 2D array of shape (D+1, m) where m is number of manifold points
        kappa: Size of margin (default 0)

    Returns:
        v_f: D+1 dimensional solution vector encoded as a 2D array of shape (D+1, 1)
        vt_f: Final value of the objective function (which does not include T^2). May be negative.
        exitflag: Not used, but equal to 1 if a local minimum is found.
        alphar: Vector of lagrange multipliers at the solution.
        normvt2: Final value of ||V-T||^2 at the solution.
    """
    D1 = t.shape[0]
    m = sD1.shape[1]
    # Construct the matrices needed for F' = 0.5 * V' * P * V - q' * V.
    # We will need P = Identity, and q = -T
    P = matrix(np.identity(D1))
    q = -t.astype(np.double)
    q = matrix(q)

    # Construct the constraints.  We need V * S - k > 0.
    # This means G = -S and h = -kappa
    G = sD1.T.astype(np.double)
    G = matrix(G)
    h = -np.ones(m) * kappa
    h = h.T.astype(np.double)
    h = matrix(h)

    # Carry out the constrained minimization
    output = solvers.qp(P, q, G, h)

    # Format the output
    v_f = np.array(output["x"])
    vt_f = output["primal objective"]
    if output["status"] == "optimal":
        exitflag = 1
    else:
        exitflag = 0
    alphar = np.array(output["z"])

    # Compute the true value of the objective function
    normvt2 = np.square(v_f - t).sum()
    return v_f, vt_f, exitflag, alphar, normvt2


def fun_FA(
    centers,
    maxK,
    max_iter,
    n_repeats,
    s_all=None,
    verbose=False,
    conjugate_gradient=True,
):
    """
    Extracts the low rank structure from the data given by centers

    Args:
        centers: 2D array of shape (N, P) where N is the ambient dimension and P is the number of centers
        maxK: Maximum rank to consider
        max_iter: Maximum number of iterations for the solver
        n_repeats: Number of repetitions to find the most stable solution at each iteration of K
        s: (Optional) iterable containing (P, 1) random normal vectors

    Returns:
        norm_coeff: Ratio of center norms before and after optimzation
        norm_coeff_vec: Mean ratio of center norms before and after optimization
        Proj: P-1 basis vectors
        V1_mat: Solution for each value of K
        res_coeff: Cost function after optimization for each K
        res_coeff0: Correlation before optimization
    """
    N, P = centers.shape
    # Configure the solver
    opts = {"max_iter": max_iter, "gtol": 1e-6, "xtol": 1e-6, "ftol": 1e-8}

    # Subtract the global mean
    mean = np.mean(centers.T, axis=0, keepdims=True)
    Xb = centers.T - mean
    xbnorm = np.sqrt(np.square(Xb).sum(axis=1, keepdims=True))

    # Gram-Schmidt into a P-1 dimensional basis
    q, r = qr(Xb.T, mode="economic")
    X = np.matmul(Xb, q[:, 0: P - 1])

    # Sore the (P, P-1) dimensional data before extracting the low rank structure
    X0 = X.copy()
    xnorm = np.sqrt(np.square(X0).sum(axis=1, keepdims=True))

    # Calculate the correlations
    C0 = np.matmul(X0, X0.T) / np.matmul(xnorm, xnorm.T)
    res_coeff0 = (np.sum(np.abs(C0)) - P) * 1 / (P * (P - 1))

    # Storage for the results
    V1_mat = []
    C0_mat = []
    norm_coeff = []
    norm_coeff_vec = []
    res_coeff = []

    # Compute the optimal low rank structure for rank 1 to maxK
    V1 = None
    for i in range(1, maxK + 1):
        best_stability = 0

        for j in range(1, n_repeats + 1):
            # Sample a random normal vector unless one is supplied
            if s_all is not None and len(s_all) >= i:
                s = s_all[i * j - 1]
            else:
                s = np.random.randn(P, 1)

            # Create initial V.
            sX = np.matmul(s.T, X)
            if V1 is None:
                V0 = sX
            else:
                V0 = np.concatenate([sX, V1.T], axis=0)
            V0, _ = qr(V0.T, mode="economic")  # (P-1, i)

            # Compute the optimal V for this i
            V1tmp, output = CGmanopt(
                V0, partial(square_corrcoeff_full_cost, grad=False), X, **opts
            )

            # Compute the cost
            cost_after, _ = square_corrcoeff_full_cost(V1tmp, X, grad=False)

            # Verify that the solution is orthogonal within tolerance
            assert (
                np.linalg.norm(np.matmul(V1tmp.T, V1tmp) - np.identity(i), ord="fro")
                < 1e-10
            )

            # Extract low rank structure
            X0 = X - np.matmul(np.matmul(X, V1tmp), V1tmp.T)

            # Compute stability of solution
            denom = np.sqrt(np.sum(np.square(X), axis=1))
            stability = min(np.sqrt(np.sum(np.square(X0), axis=1)) / denom)

            # Store the solution if it has the best stability
            if stability > best_stability:
                best_stability = stability
                best_V1 = V1tmp
            if n_repeats > 1 and verbose:
                print(j, "cost=", cost_after, "stability=", stability)

        # Use the best solution
        V1 = best_V1

        # Extract the low rank structure
        XV1 = np.matmul(X, V1)
        X0 = X - np.matmul(XV1, V1.T)

        # Compute the current (normalized) cost
        xnorm = np.sqrt(np.square(X0).sum(axis=1, keepdims=True))
        C0 = np.matmul(X0, X0.T) / np.matmul(xnorm, xnorm.T)
        current_cost = (np.sum(np.abs(C0)) - P) * 1 / (P * (P - 1))
        if verbose:
            print("K=", i, "mean=", current_cost)

        # Store the results
        V1_mat.append(V1)
        C0_mat.append(C0)
        norm_coeff.append((xnorm / xbnorm)[:, 0])
        norm_coeff_vec.append(np.mean(xnorm / xbnorm))
        res_coeff.append(current_cost)

        # Break the loop if there's been no reduction in cost for 3 consecutive iterations
        if (
            i > 4
            and res_coeff[i - 1] > res_coeff[i - 2]
            and res_coeff[i - 2] > res_coeff[i - 3]
            and res_coeff[i - 3] > res_coeff[i - 4]
        ):
            if verbose:
                print("Optimal K0 found")
            break
    return norm_coeff, norm_coeff_vec, q[:, 0: P - 1], V1_mat, res_coeff, res_coeff0


def CGmanopt(X, objective_function, A, **kwargs):
    """
    Minimizes the objective function subject to the constraint that X.T * X = I_k using the
    conjugate gradient method

    Args:
        X: Initial 2D array of shape (n, k) such that X.T * X = I_k
        objective_function: Objective function F(X, A) to minimize.
        A: Additional parameters for the objective function F(X, A)

    Keyword Args:
        None

    Returns:
        Xopt: Value of X that minimizes the objective subject to the constraint.
    """

    manifold = Stiefel(X.shape[0], X.shape[1])

    @pymanopt.function.autograd(manifold)
    def cost(X):
        c, _ = objective_function(X, A)
        return c

    problem = Problem(
        manifold=manifold,
        cost=cost,
        #   verbosity=0
    )
    solver = ConjugateGradient(log_verbosity=0)
    Xopt = solver.run(problem).point
    return Xopt, None


def square_corrcoeff_full_cost(V, X, grad=True):
    """
    The cost function for the correlation analysis. This effectively measures the square difference
    in correlation coefficients after transforming to an orthonormal basis given by V.

    Args:
        V: 2D array of shape (N, K) with V.T * V = I
        X: 2D array of shape (P, N) containing centers of P manifolds in an N=P-1 dimensional
            orthonormal basis
    """
    # Verify that the shapes are correct
    P, N = X.shape
    N_v, K = V.shape
    assert N_v == N

    # Calculate the cost
    C = np.matmul(X, X.T)
    c = np.matmul(X, V)
    c0 = np.diagonal(C).reshape(P, 1) - np.sum(np.square(c), axis=1, keepdims=True)
    Fmn = np.square(C - np.matmul(c, c.T)) / np.matmul(c0, c0.T)
    cost = np.sum(Fmn) / 2

    if grad is False:  # skip gradient calc since not needed, or autograd is used
        gradient = None
    else:
        # Calculate the gradient
        X1 = np.reshape(X, [1, P, N, 1])
        X2 = np.reshape(X, [P, 1, N, 1])
        C1 = np.reshape(c, [P, 1, 1, K])
        C2 = np.reshape(c, [1, P, 1, K])

        # Sum the terms in the gradient
        PF1 = ((C - np.matmul(c, c.T)) / np.matmul(c0, c0.T)).reshape(P, P, 1, 1)
        PF2 = (
            np.square(C - np.matmul(c, c.T)) / np.square(np.matmul(c0, c0.T))
        ).reshape(P, P, 1, 1)
        Gmni = -PF1 * C1 * X1
        Gmni += -PF1 * C2 * X2
        Gmni += PF2 * c0.reshape(P, 1, 1, 1) * C2 * X1
        Gmni += PF2 * (c0.T).reshape(1, P, 1, 1) * C1 * X2
        gradient = np.sum(Gmni, axis=(0, 1))

    return cost, gradient


def MGramSchmidt(V):
    """
    Carries out the Gram Schmidt process on the input vectors V

    Args:
        V: 2D array of shape (n, k) containing k vectors of dimension n

    Returns:
        V_out: 2D array of shape (n, k) containing k orthogonal vectors of dimension n
    """
    n, k = V.shape
    V_out = np.copy(V)
    for i in range(k):
        for j in range(i):
            V_out[:, i] = V_out[:, i] - proj(V_out[:, j], V_out[:, i])
        V_out[:, i] = V_out[:, i] / np.linalg.norm(V_out[:, i])
    return V_out


def proj(v1, v2):
    """
    Projects vector v2 onto vector v1

    Args:
        v1: Array containing vector v1 (can be 1D or 2D with shape (dimension, 1))
        v2: Array containing vector v1 (can be 1D or 2D with shape (dimension, 1))

    Returns:
        v: Array containing the projection of v2 onto v1.  Same shape as v1.
    """
    v = np.dot(v1.T, v2) / np.dot(v1.T, v1) * v1
    return v
