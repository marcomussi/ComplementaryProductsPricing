import numpy as np


def incr_inv(A_inv, B, C, D):
    """
    Calculates the inverse of a 2x2 block matrix using the block matrix
    inversion formula (Schur complement method).

    This function is "incremental" because it assumes the inverse of the
    top-left block 'A' (A_inv) is already known. It efficiently computes
    the inverse of the full-rank matrix M:
        M = [[A, B],
             [C, D]]
    
    Args:
        A_inv (numpy.ndarray): The inverse of the top-left block (A).
                               Shape (n, n).
        B (numpy.ndarray): The top-right block. Shape (n, n_add).
        C (numpy.ndarray): The bottom-left block. Shape (n_add, n).
        D (numpy.ndarray): The bottom-right block. Shape (n_add, n_add).

    Returns:
        numpy.ndarray: The inverse of the full block matrix M.
                       Shape (n + n_add, n + n_add).

    Raises:
        ValueError: If any input is not a 2D numpy array or if the matrix 
                    dimensions are inconsistent.
        numpy.linalg.LinAlgError: If the Schur complement (D - C @ A_inv @ B)
                                  is singular (not invertible).
    
    Note:
        This function assumes all input matrices are 2D and square where
        appropriate (A_inv is n x n, D is n_add x n_add). It does not
        compute pseudo-inverses.
    """

    n = A_inv.shape[0] # original number of rows/cols
    n_add = C.shape[0] # number of rows/cols to add
    
    if not isinstance(A_inv, np.ndarray) or A_inv.ndim != 2:
        raise ValueError(f"Input 'A_inv' must be a 2D numpy array. Got {type(A_inv)} with ndim={A_inv.ndim}")
    if not isinstance(B, np.ndarray) or B.ndim != 2:
        raise ValueError(f"Input 'B' must be a 2D numpy array. Got {type(B)} with ndim={B.ndim}")
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        raise ValueError(f"Input 'C' must be a 2D numpy array. Got {type(C)} with ndim={C.ndim}")
    if not isinstance(D, np.ndarray) or D.ndim != 2:
        raise ValueError(f"Input 'D' must be a 2D numpy array. Got {type(D)} with ndim={D.ndim}")
    if A_inv.shape != (n, n):
        raise ValueError(f"Input 'A_inv' must be square (n, n). Got {A_inv.shape}")
    if B.shape != (n, n_add):
        raise ValueError(f"Shape mismatch: 'B' must be (n, n_add). "
                         f"Got B.shape={B.shape}, expected ({n}, {n_add})")
    if C.shape != (n_add, n):
        raise ValueError(f"Shape mismatch: 'C' must be (n_add, n). "
                         f"Got C.shape={C.shape}, expected ({n_add}, {n})")
    if D.shape != (n_add, n_add):
        raise ValueError(f"Shape mismatch: 'D' must be (n_add, n_add). "
                         f"Got D.shape={D.shape}, expected ({n_add}, {n_add})")

    schur_complement = D - C @ A_inv @ B
    schur_complement_inv = np.linalg.solve(schur_complement, np.eye(n_add))

    block1 = A_inv + A_inv @ B @ schur_complement_inv @ C @ A_inv
    block2 = -A_inv @ B @ schur_complement_inv
    block3 = -schur_complement_inv @ C @ A_inv
    block4 = schur_complement_inv

    res1 = np.concatenate((block1, block2), axis=1)
    res2 = np.concatenate((block3, block4), axis=1)

    return np.concatenate((res1, res2), axis=0)
