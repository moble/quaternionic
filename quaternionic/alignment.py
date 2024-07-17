# Copyright (c) 2024, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

import numpy as np
from scipy.linalg import eigh

from . import jit, array
from .interpolation import unflip_rotors

def align(a, b, w=None):
    """Find the rotation aligning vectors `a` to `b`

    This function solves Wahba's problem, essentially computing a
    quaternion corresponding to the rotation ℛ that minimizes the loss
    function

        L(ℛ) ≔ Σᵢ wᵢ ‖a⃗ᵢ - ℛ b⃗ᵢ‖²

    Note that currently this function is only designed to work with
    3-vectors, and will raise an error if the input arrays `a` and `b`
    are not of shape (N, 3).  The optional `w` parameter, if present,
    should be of shape (N,); otherwise all weights are assumed to be
    1.

    """
    assert a.ndim == b.ndim == 2
    assert a.shape[0] == b.shape[0]
    assert a.shape[1] == b.shape[1] == 3
    if w is not None:
        assert w.shape[0] == a.shape[0]
        assert w.ndim == 1
    S = np.zeros_like(a, shape=(3,3))
    _construct_S(S, a, b, w)

    # This is Eq. (5.11) from Markley and Crassidis, modified to suit our
    # conventions by flipping the sign of ``z``, and moving the final dimension
    # to the first dimension.
    M = np.array([
        [S[0,0]+S[1,1]+S[2,2],      S[2,1]-S[1,2],         S[0,2]-S[2,0],           S[1,0]-S[0,1],    ],
        [    S[2,1]-S[1,2],      S[0,0]-S[1,1]-S[2,2],     S[0,1]+S[1,0],           S[0,2]+S[2,0],    ],
        [    S[0,2]-S[2,0],         S[0,1]+S[1,0],      -S[0,0]+S[1,1]-S[2,2],      S[1,2]+S[2,1],    ],
        [    S[1,0]-S[0,1],         S[0,2]+S[2,0],         S[1,2]+S[2,1],       -S[0,0]-S[1,1]+S[2,2],],
    ])
    # This extracts the dominant eigenvector, and interprets it as a quaternion
    return unflip_rotors(array(eigh(M, subset_by_index=(3, 3))[1][:, 0]))

@jit
def _construct_S(S, a, b, w=None):
    if w is None:
        for i in range(len(a)):
            for j in range(3):
                for k in range(3):
                    S[j,k] += a[i,j] * b[i,k]
    else:
        assert w.shape[0] == a.shape[0]
        assert w.ndim == 1
        for i in range(len(a)):
            for j in range(3):
                for k in range(3):
                    S[j,k] += w[i] * a[i,j] * b[i,k]
    return
