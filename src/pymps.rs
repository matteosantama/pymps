use crate::errors::ConversionFailure;
use mps::Parser;
use mps::types::{BoundType, BoundsLine, RowType};
use numpy::{IntoPyArray, PyArray1};
use pyo3::{Python, pyclass, pymethods};
use std::collections::{HashMap, HashSet};

/// Represents a primal QP problem:
///     min.    0.5 * x.T @ p @ x + c @ x + r
///     s.t.    a @ x + s == b,
///             s[:j] == 0,
///             s[j:] >= 0
#[pyclass]
pub(crate) struct QuadraticProgram {
    #[pyo3(get)]
    p: CscMatrix,
    c: Vec<f64>,
    #[pyo3(get)]
    r: f64,
    #[pyo3(get)]
    a: CscMatrix,
    b: Vec<f64>,
    #[pyo3(get)]
    j: usize,
}

#[pymethods]
impl QuadraticProgram {
    #[getter]
    fn c<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, PyArray1<f64>> {
        self.c.clone().into_pyarray(py)
    }
    #[getter]
    fn b<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, PyArray1<f64>> {
        self.b.clone().into_pyarray(py)
    }
}

impl TryFrom<Parser<'_, f64>> for QuadraticProgram {
    type Error = ConversionFailure;

    fn try_from(parser: Parser<'_, f64>) -> Result<Self, Self::Error> {
        let obj_name = find_objective_row(&parser)?;
        let nr_rows: HashSet<&str> = parser
            .rows
            .iter()
            .filter(|r| matches!(r.row_type, RowType::Nr))
            .map(|r| r.row_name)
            .collect();

        // Variables are identified by first appearance in COLUMNS. BOUNDS and
        // QUADOBJ may reference them but cannot introduce new ones.
        // Invariant: col_ids maps each name to a unique id in [0, n).
        let mut col_ids: HashMap<&str, u32> = HashMap::new();
        for entry in &parser.columns {
            if !col_ids.contains_key(entry.name) {
                let id = col_ids.len() as u32;
                col_ids.insert(entry.name, id);
            }
        }
        let n = col_ids.len() as u32;

        // Non-N rows in input order with a temporary index. Final row IDs are
        // assigned later, after ranges expand some rows and bound-derived rows are
        // appended.
        let mut row_info: Vec<(&str, RowType)> = Vec::new();
        for row in &parser.rows {
            if !matches!(row.row_type, RowType::Nr) {
                row_info.push((row.row_name, row.row_type.clone()));
            }
        }
        let row_temp_idx: HashMap<&str, usize> = row_info
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (*name, i))
            .collect();

        // Walk COLUMNS once: route each (col, row, value) entry to either the
        // linear objective vector (if row is the objective) or the per-row
        // coefficient map (if row is a real constraint). Duplicates accumulate.
        let mut c = vec![0.0_f64; n as usize];
        let mut a_by_row: Vec<HashMap<u32, f64>> = vec![HashMap::new(); row_info.len()];
        for entry in &parser.columns {
            let col_id = col_ids[entry.name];
            let pairs = std::iter::once(&entry.first_pair).chain(entry.second_pair.as_ref());
            for pair in pairs {
                if pair.row_name == obj_name {
                    c[col_id as usize] += pair.value;
                } else if nr_rows.contains(pair.row_name) {
                    // auxiliary N row — discarded
                } else if let Some(&idx) = row_temp_idx.get(pair.row_name) {
                    *a_by_row[idx].entry(col_id).or_insert(0.0) += pair.value;
                } else {
                    return Err(ConversionFailure(format!(
                        "column '{}' references unknown row '{}'",
                        entry.name, pair.row_name
                    )));
                }
            }
        }

        // RHS values per row (temporary indexing). RHS on the objective row encodes
        // an additive constant: MPS convention is obj = c'x - rhs_obj, so we store
        // r = -rhs_obj to keep our `+ r` form.
        let mut b_temp = vec![0.0_f64; row_info.len()];
        let mut r = 0.0_f64;
        if let Some(rhs_entries) = &parser.rhs {
            for entry in rhs_entries {
                let pairs = std::iter::once(&entry.first_pair).chain(entry.second_pair.as_ref());
                for pair in pairs {
                    if pair.row_name == obj_name {
                        r = -pair.value;
                    } else if nr_rows.contains(pair.row_name) {
                        // ignore
                    } else if let Some(&idx) = row_temp_idx.get(pair.row_name) {
                        b_temp[idx] = pair.value;
                    } else {
                        return Err(ConversionFailure(format!(
                            "RHS references unknown row '{}'",
                            pair.row_name
                        )));
                    }
                }
            }
        }

        // RANGES values per row (temporary indexing). Sign matters for Eq rows
        // (positive vs negative R picks which side of b extends); see `range_limits`.
        let mut ranges_by_row: HashMap<usize, f64> = HashMap::new();
        if let Some(range_entries) = &parser.ranges {
            for entry in range_entries {
                let pairs = std::iter::once(&entry.first_pair).chain(entry.second_pair.as_ref());
                for pair in pairs {
                    if nr_rows.contains(pair.row_name) {
                        continue;
                    }
                    if let Some(&idx) = row_temp_idx.get(pair.row_name) {
                        ranges_by_row.insert(idx, pair.value);
                    } else {
                        return Err(ConversionFailure(format!(
                            "RANGES references unknown row '{}'",
                            pair.row_name
                        )));
                    }
                }
            }
        }

        // Fold (row type, RHS, range) into a canonical (L, U) pair per row, then
        // classify. After this point row direction is fully determined and the
        // original RowType / range distinction is discarded.
        let resolved: Vec<ResolvedRow> = row_info
            .iter()
            .enumerate()
            .map(|(idx, (_, row_type))| {
                let b = b_temp[idx];
                let range = ranges_by_row.get(&idx).copied();
                let (lo, up) = range_limits(row_type, b, range);
                if lo == up {
                    ResolvedRow::Eq(lo)
                } else if lo == f64::NEG_INFINITY {
                    ResolvedRow::Leq(up)
                } else if up == f64::INFINITY {
                    ResolvedRow::Geq(lo)
                } else {
                    ResolvedRow::Both(lo, up)
                }
            })
            .collect();

        // Per-variable (lower, upper) state. Default is [0, +inf]; each BOUNDS line
        // refines the relevant side, with later lines overwriting earlier ones.
        let mut lower = vec![0.0_f64; n as usize];
        let mut upper = vec![f64::INFINITY; n as usize];
        if let Some(bounds) = &parser.bounds {
            for bound in bounds {
                let col = *col_ids.get(bound.column_name).ok_or_else(|| {
                    ConversionFailure(format!(
                        "bound references unknown column '{}'",
                        bound.column_name
                    ))
                })? as usize;
                apply_bound(&mut lower, &mut upper, col, bound)?;
            }
        }

        // Variable bounds aren't first-class in the QP form, so each finite bound
        // becomes an explicit row. Fixed (lo == up) collapses to one equality;
        // otherwise each finite side contributes a single inequality row.
        let mut bound_rows: Vec<(u32, BoundKind, f64)> = Vec::new();
        for col in 0..(n as usize) {
            let lo = lower[col];
            let up = upper[col];
            if lo == up {
                if lo.is_finite() {
                    bound_rows.push((col as u32, BoundKind::Eq, lo));
                }
            } else {
                if lo.is_finite() {
                    bound_rows.push((col as u32, BoundKind::Geq, lo));
                }
                if up.is_finite() {
                    bound_rows.push((col as u32, BoundKind::Leq, up));
                }
            }
        }

        // Final row layout. The QP form requires `s[:j] == 0` then `s[j:] >= 0`,
        // so all equality rows (originals + fixed-bound rows) are emitted first; j
        // captures the equality count. Geq rows are flipped (negate coef and b) so
        // every inequality fits the `+s, s >= 0` convention. Two-sided rows
        // (`Both`) split into a Leq + Geq pair sharing the same coefficients.
        // Invariants after this block: rows [0, j) are equalities, [j, m) are
        // inequalities; a_entries.row.max() < m; b.len() == m.
        let mut a_entries: Vec<(u32, u32, f64)> = Vec::new();
        let mut b: Vec<f64> = Vec::new();
        let mut next_row: u32 = 0;

        for (idx, row) in resolved.iter().enumerate() {
            if let ResolvedRow::Eq(val) = row {
                for (&col, &coef) in &a_by_row[idx] {
                    a_entries.push((next_row, col, coef));
                }
                b.push(*val);
                next_row += 1;
            }
        }
        for &(col, kind, val) in &bound_rows {
            if matches!(kind, BoundKind::Eq) {
                a_entries.push((next_row, col, 1.0));
                b.push(val);
                next_row += 1;
            }
        }
        let j = next_row as usize;

        for (idx, row) in resolved.iter().enumerate() {
            match row {
                ResolvedRow::Eq(_) => continue,
                ResolvedRow::Leq(val) => {
                    for (&col, &coef) in &a_by_row[idx] {
                        a_entries.push((next_row, col, coef));
                    }
                    b.push(*val);
                    next_row += 1;
                }
                ResolvedRow::Geq(val) => {
                    for (&col, &coef) in &a_by_row[idx] {
                        a_entries.push((next_row, col, -coef));
                    }
                    b.push(-val);
                    next_row += 1;
                }
                ResolvedRow::Both(lo, up) => {
                    for (&col, &coef) in &a_by_row[idx] {
                        a_entries.push((next_row, col, coef));
                    }
                    b.push(*up);
                    next_row += 1;
                    for (&col, &coef) in &a_by_row[idx] {
                        a_entries.push((next_row, col, -coef));
                    }
                    b.push(-lo);
                    next_row += 1;
                }
            }
        }
        for &(col, kind, val) in &bound_rows {
            let (coef, b_val) = match kind {
                BoundKind::Leq => (1.0, val),
                BoundKind::Geq => (-1.0, -val),
                BoundKind::Eq => continue,
            };
            a_entries.push((next_row, col, coef));
            b.push(b_val);
            next_row += 1;
        }
        let m = next_row;

        let a_mat = build_csc(a_entries, m, n);

        // Quadratic objective. Assumes QUADOBJ/QSECTION convention (each off-diagonal
        // pair listed once); symmetrized by emitting both (i,k) and (k,i).
        let mut p_entries: Vec<(u32, u32, f64)> = Vec::new();
        if let Some(quad) = &parser.quadratic_objective {
            for term in quad {
                let i = *col_ids.get(term.var1).ok_or_else(|| {
                    ConversionFailure(format!(
                        "quadratic objective references unknown variable '{}'",
                        term.var1
                    ))
                })?;
                let k = *col_ids.get(term.var2).ok_or_else(|| {
                    ConversionFailure(format!(
                        "quadratic objective references unknown variable '{}'",
                        term.var2
                    ))
                })?;
                p_entries.push((i, k, term.coefficient));
                if i != k {
                    p_entries.push((k, i, term.coefficient));
                }
            }
        }
        let p_mat = build_csc(p_entries, n, n);

        let qp = QuadraticProgram {
            p: p_mat,
            c,
            r,
            a: a_mat,
            b,
            j,
        };
        Ok(qp)
    }
}

#[derive(Clone, Copy)]
enum BoundKind {
    Eq,
    Leq,
    Geq,
}

enum ResolvedRow {
    Eq(f64),
    Leq(f64),
    Geq(f64),
    Both(f64, f64),
}

/// Returns the (L, U) double-sided limits implied by a row's type, RHS, and
/// optional RANGES value. Encodes the Maros CTSM range table: ranged Eq rows
/// extend toward +R or -R depending on sign, ranged Leq/Geq rows extend by |R|
/// on the unbounded side. Unbounded sides are returned as ±infinity.
fn range_limits(row_type: &RowType, b: f64, range: Option<f64>) -> (f64, f64) {
    match range {
        None => match row_type {
            RowType::Eq => (b, b),
            RowType::Leq => (f64::NEG_INFINITY, b),
            RowType::Geq => (b, f64::INFINITY),
            RowType::Nr => unreachable!("N rows are filtered out before resolution"),
        },
        Some(r) => {
            let abs_r = r.abs();
            match row_type {
                RowType::Leq => (b - abs_r, b),
                RowType::Geq => (b, b + abs_r),
                RowType::Eq => {
                    if r >= 0.0 {
                        (b, b + abs_r)
                    } else {
                        (b - abs_r, b)
                    }
                }
                RowType::Nr => unreachable!("N rows are filtered out before resolution"),
            }
        }
    }
}

/// Returns the name of the objective row: either the explicit `objective_name`
/// from the OBJSENSE/OBJNAME section (validated to be of type N) or the first N
/// row in declaration order.
fn find_objective_row<'a>(parser: &Parser<'a, f64>) -> Result<&'a str, ConversionFailure> {
    if let Some(name) = parser.objective_name {
        let ok = parser
            .rows
            .iter()
            .any(|r| r.row_name == name && matches!(r.row_type, RowType::Nr));
        if !ok {
            return Err(ConversionFailure(format!(
                "objective row '{name}' not found in ROWS (must be type N)"
            )));
        }
        return Ok(name);
    }
    parser
        .rows
        .iter()
        .find(|r| matches!(r.row_type, RowType::Nr))
        .map(|r| r.row_name)
        .ok_or_else(|| ConversionFailure("no objective row (type N) found".to_string()))
}

/// Applies one BOUNDS line to a variable's (lower, upper) state. `LO`/`UP`/`FX`
/// require an associated value; `MI`/`PL`/`FR` are valueless. `MI` follows the
/// strict-MPS convention: lower becomes -inf and upper becomes 0 (overridden if
/// a later UP bound is more permissive).
fn apply_bound(
    lower: &mut [f64],
    upper: &mut [f64],
    col: usize,
    bound: &BoundsLine<f64>,
) -> Result<(), ConversionFailure> {
    match bound.bound_type {
        BoundType::Lo => {
            lower[col] = bound
                .value
                .ok_or_else(|| ConversionFailure("LO bound requires a value".into()))?;
        }
        BoundType::Up => {
            upper[col] = bound
                .value
                .ok_or_else(|| ConversionFailure("UP bound requires a value".into()))?;
        }
        BoundType::Fx => {
            let v = bound
                .value
                .ok_or_else(|| ConversionFailure("FX bound requires a value".into()))?;
            lower[col] = v;
            upper[col] = v;
        }
        BoundType::Fr => {
            lower[col] = f64::NEG_INFINITY;
            upper[col] = f64::INFINITY;
        }
        BoundType::Mi => {
            lower[col] = f64::NEG_INFINITY;
        }
        BoundType::Pl => {
            upper[col] = f64::INFINITY;
        }
        BoundType::Bv | BoundType::Li | BoundType::Ui | BoundType::Sc => {
            return Err(ConversionFailure(format!(
                "bound type {:?} (integer/semi-continuous) is not supported",
                bound.bound_type
            )));
        }
    }
    Ok(())
}

/// Builds a CSC matrix from (row, col, value) triples. Triples sharing the same
/// (row, col) are summed; entries that sum to zero are dropped. Output is in
/// canonical CSC form: indices sorted ascending within each column.
fn build_csc(mut entries: Vec<(u32, u32, f64)>, nrows: u32, ncols: u32) -> CscMatrix {
    entries.sort_unstable_by_key(|&(r, c, _)| (c, r));

    let mut data = Vec::with_capacity(entries.len());
    let mut indices = Vec::with_capacity(entries.len());
    let mut indptr = vec![0u32; ncols as usize + 1];

    let mut i = 0;
    while i < entries.len() {
        let (r, c, mut v) = entries[i];
        let mut k = i + 1;
        while k < entries.len() && entries[k].0 == r && entries[k].1 == c {
            v += entries[k].2;
            k += 1;
        }
        if v != 0.0 {
            data.push(v);
            indices.push(r);
            indptr[c as usize + 1] += 1;
        }
        i = k;
    }

    for k in 1..indptr.len() {
        indptr[k] += indptr[k - 1];
    }

    CscMatrix {
        data,
        indices,
        indptr,
        shape: (nrows, ncols),
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct CscMatrix {
    data: Vec<f64>,
    indices: Vec<u32>,
    indptr: Vec<u32>,
    #[pyo3(get)]
    shape: (u32, u32),
}

#[pymethods]
impl CscMatrix {
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, PyArray1<f64>> {
        self.data.clone().into_pyarray(py)
    }
    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, PyArray1<u32>> {
        self.indices.clone().into_pyarray(py)
    }
    #[getter]
    fn indptr<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, PyArray1<u32>> {
        self.indptr.clone().into_pyarray(py)
    }
}
