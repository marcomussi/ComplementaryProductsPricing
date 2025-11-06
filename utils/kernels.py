import numpy as np


def kernel_rbf(a, b, L):
    """
    Calculates the Radial Basis Function (RBF) kernel matrix between two sets of vectors.
    The formula implemented is:
    K(a_i, b_j) = exp(-L * ||a_i - b_j||^2)

    Where:
    - a_i is the i-th row (vector) in matrix `a`.
    - b_j is the j-th row (vector) in matrix `b`.
    - ||...||^2 is the squared Euclidean distance.
    - L is the length-scale parameter (often related to gamma, e.g., L = gamma).
      A smaller 'L' results in a wider kernel, meaning points further
      apart are considered more similar (smoother function). A larger 'L'
      results in a narrower kernel, focusing more on local similarity.

    Args:
        a (numpy.ndarray): A 2D array of shape (n_samples_a, n_features)
                           representing the first set of vectors.
        b (numpy.ndarray): A 2D array of shape (n_samples_b, n_features)
                           representing the second set of vectors.
                           `a` and `b` must have the same number of features.
        L (float): The length-scale parameter (gamma) of the kernel. Must be
                   a positive value.

    Returns:
        numpy.ndarray: A 2D array (kernel matrix) of shape (n_samples_a, n_samples_b)
                       where the element (i, j) is the RBF kernel similarity
                       between `a[i]` and `b[j]`.

    Example:
        >>> import numpy as np
        >>> a = np.array([[0, 0], [1, 1]])
        >>> b = np.array([[0, 0], [1, 1], [2, 2]])
        >>> L = 0.5
        >>> kernel_rbf(a, b, L)
        array([[1. , 0.3, 0.0],
               [0.3, 1. , 0.3]])
    """
    if not isinstance(a, np.ndarray) or a.ndim != 2:
        raise ValueError(f"Input 'a' must be a 2D numpy array. Got shape {a.shape}")
    if not isinstance(b, np.ndarray) or b.ndim != 2:
        raise ValueError(f"Input 'b' must be a 2D numpy array. Got shape {b.shape}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Inputs 'a' and 'b' must have the same number of features (columns). "
                         f"Got a.shape[1] = {a.shape[1]} and b.shape[1] = {b.shape[1]}")
    if not (isinstance(L, (int, float)) and L > 0):
        raise ValueError(f"Parameter 'L' must be a positive number. Got {L}")

    sq_dists = np.ones((a.shape[0], b.shape[0]))
    
    for i in range(a.shape[0]):
        
        for j in range(b.shape[0]):
            
            sq_dists[i, j] = np.power(np.linalg.norm(a[i, :] - b[j, :], 2), 2)
    
    return np.exp(-L * sq_dists)
