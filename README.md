# pymps

Python bindings for reading MPS/QPS files into a `QuadraticProgram` with scipy sparse matrices. Wraps the Rust [`mps`](https://crates.io/crates/mps) crate via PyO3/maturin.

## Quadratic program form

A parsed file becomes a `QuadraticProgram` describing:

```
min.    0.5 * x.T @ p @ x + c @ x + r
s.t.    a @ x + s == b,
        s[:j] == 0,
        s[j:] >= 0
```

| Field | Type                  | Description                                    |
|-------|-----------------------|------------------------------------------------|
| `p`   | `scipy.sparse.csc_matrix` | Symmetric quadratic objective matrix (`n × n`) |
| `c`   | `numpy.ndarray`       | Linear objective coefficients (`n`)            |
| `r`   | `float`               | Constant objective offset                      |
| `a`   | `scipy.sparse.csc_matrix` | Constraint matrix (`m × n`)                |
| `b`   | `numpy.ndarray`       | Constraint right-hand side (`m`)               |
| `j`   | `int`                 | Number of equality constraints (first `j` rows of `a`) |

Variable bounds from the BOUNDS section are folded into additional rows of `a`.

## Install

Requires Rust (stable) and Python ≥ 3.13.

```bash
uv sync
uv run maturin develop --release
```

## Usage

```python
import pymps

qp = pymps.read_f64("problem.qps")
qp.p          # scipy.sparse.csc_matrix
qp.c          # numpy.ndarray
qp.a, qp.b, qp.r, qp.j
```

`read_f64` accepts `str` or `pathlib.Path`.

### Solving with Clarabel

```python
import clarabel
import scipy.sparse as sp

m = qp.a.shape[0]
cones = [clarabel.ZeroConeT(qp.j), clarabel.NonnegativeConeT(m - qp.j)]
solver = clarabel.DefaultSolver(
    sp.triu(qp.p, format="csc"),  # Clarabel wants the upper triangle
    qp.c, qp.a, qp.b, cones, clarabel.DefaultSettings(),
)
solution = solver.solve()
full_opt = solution.obj_val + qp.r
```

## Errors

All exceptions live in `pymps.errors` and inherit from a common `MpsError` base:

- `ParseError` — the underlying MPS parser rejected the file.
- `UnsupportedMpsError` — file uses a feature pymps doesn't support (see below).
- `ConversionError` — the file parsed but couldn't be mapped to the QP form (e.g. unknown row/column reference, integer bound type).

```python
from pymps.errors import MpsError, UnsupportedMpsError

try:
    qp = pymps.read_f64("problem.qps")
except UnsupportedMpsError as e:
    ...
except MpsError as e:
    ...
```

### Unsupported features

The following MPS/QPS sections raise `UnsupportedMpsError`:

- `OBJSENSE MAX` (maximization)
- `REFROW`, `USERCUTS`, `SOS`, `QCMATRIX`, `INDICATORS`, `LAZYCONS`, `CSECTION`, `BRANCH`

Integer/binary variable bound types (`BV`, `LI`, `UI`, `SC`) raise `ConversionError`.

### Conventions

- **Objective constant.** MPS convention `obj = c'x - rhs_obj`, so `r = -rhs_obj`.
- **Quadratic objective.** Assumes QUADOBJ/QSECTION format (each off-diagonal pair listed once); the parser symmetrizes by emitting `(i, j)` and `(j, i)` entries.
- **Bounds.** Default `[0, +∞]`. `MI` uses strict-MPS semantics: `[-∞, 0]`.

## Development

```bash
uv sync --group test --group dev
uv run maturin develop
uv run pytest
```

### Project layout

- `src/` — Rust crate: `lib.rs` exposes the `_pymps` extension module; `pymps.rs` contains the MPS→QP conversion; `errors.rs` defines the exception types.
- `python/pymps/` — Python package; re-exports `read_f64` and `QuadraticProgram` from the extension, plus `pymps.errors`.
- `tests/` — pytest suite that parses and solves the full Maros-Meszaros test set against reference optima, comparing with Clarabel.
- `scripts/upload_problems.py` — zips `.mps`/`.qps` files and uploads to a Cloudflare R2 bucket for the test suite to download.
