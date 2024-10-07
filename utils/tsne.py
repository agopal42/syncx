# Adapted to pytorch based on author's reference implementation in python.
# https://lvdmaaten.github.io/tsne/#implementations

import numpy as np
import matplotlib.pyplot as plt
import torch


def Hbeta(D: torch.Tensor, beta: float = 1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = torch.exp(-D.clone() * beta)
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X: torch.Tensor, tol: float = 1e-5, perplexity: float = 30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    n, d = X.shape
    sum_X = torch.sum(torch.square(X), dim=1)
    D = torch.add(torch.add(-2 * torch.matmul(X, X.T), sum_X).T, sum_X)
    P = torch.zeros(n, n, device=X.device)
    beta = torch.ones(n, 1, device=X.device)
    logU = torch.log(torch.tensor([perplexity], device=X.device))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = torch.tensor([-np.inf], device=X.device)
        betamax = torch.tensor([np.inf], device=X.device)
        Di = D[i, n_list[0:i]+n_list[i+1:n]]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 40:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax == torch.tensor([np.inf], device=betamax.device) or \
                    betamax == torch.tensor([-np.inf], device=betamax.device):
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin == torch.tensor([np.inf], device=betamin.device) or \
                    betamin == torch.tensor([-np.inf], device=betamin.device):
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP.float()

    # Return final P-matrix
    print("Mean value of sigma: %f" % torch.mean(torch.sqrt(1 / beta)))
    return P


def pca(X=torch.Tensor, no_dims: int = 50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.tile(torch.mean(X, 0), (n, 1))
    (l, M) = torch.linalg.eig(torch.matmul(X.T, X))
    if M.dtype == torch.complex64:
        M = M.real  # use real component only in the case complex eigenvalues
    Y = torch.matmul(X, M[:, 0:no_dims])
    return Y


def tsne(X=torch.Tensor, no_dims: int = 2, initial_dims: int = 50, perplexity: float = 30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims)
    (n, d) = X.shape
    max_iter = 200
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims, device=X.device)
    dY = torch.zeros(n, no_dims, device=X.device)
    iY = torch.zeros(n, no_dims, device=X.device)
    gains = torch.ones(n, no_dims, device=X.device)

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + torch.transpose(P, 0, 1)
    P = P / torch.sum(P)
    P = P * 4.									# early exaggeration
    P = torch.maximum(P, torch.tensor([1e-12], device=P.device))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(torch.square(Y), 1)
        num = -2. * torch.matmul(Y, Y.T)
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.maximum(Q, torch.tensor([1e-12], device=P.device))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum(torch.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.tile(torch.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = torch.from_numpy(np.loadtxt("mnist2500_X.txt"))
    labels = torch.from_numpy(np.loadtxt("mnist2500_labels.txt"))
    Y = tsne(X, 2, 50, 20.0)
    plt.scatter(Y[:, 0].numpy(), Y[:, 1].numpy(), 20, labels)
    plt.show()
