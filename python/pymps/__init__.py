import dataclasses
import os

import numpy as np
import scipy as sp

from pymps._pymps import ConversionError, MpsError, ParseError, UnsupportedMpsError
from pymps._pymps import read_f64 as _read_f64

__all__ = [
    "QuadraticProgram",
    "read_f64",
    "MpsError",
    "ConversionError",
    "ParseError",
    "UnsupportedMpsError",
]


@dataclasses.dataclass(kw_only=True, frozen=True)
class QuadraticProgram:
    """Represents a primal QP problem:
    min.    0.5 * x.T @ p @ x + c @ x + r
    s.t.    a @ x + s == b,
            s[:j] == 0,
            s[j:] >= 0
    """

    p: sp.sparse.csc_matrix
    c: np.ndarray
    r: float
    a: sp.sparse.csc_matrix
    b: np.ndarray
    j: int


def read_f64(filepath: str | os.PathLike) -> QuadraticProgram:
    """Read an MPS file."""
    quadprog = _read_f64(filepath)

    return QuadraticProgram(
        p=sp.sparse.csc_matrix(
            (quadprog.p.data, quadprog.p.indices, quadprog.p.indptr),
            shape=quadprog.p.shape,
            copy=False,
        ),
        c=quadprog.c,
        r=quadprog.r,
        a=sp.sparse.csc_matrix(
            (quadprog.a.data, quadprog.a.indices, quadprog.a.indptr),
            shape=quadprog.a.shape,
            copy=False,
        ),
        b=quadprog.b,
        j=quadprog.j,
    )
