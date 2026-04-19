use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_pymps, MpsError, PyException);
create_exception!(_pymps, ParseError, MpsError);
create_exception!(_pymps, UnsupportedMpsError, MpsError);
create_exception!(_pymps, ConversionError, MpsError);

#[derive(Debug)]
pub(crate) enum UnsupportedFeature {
    MaximizationObjective,
    ReferenceRow,
    UserCuts,
    SpecialOrderedSets,
    QuadraticConstraints,
    Indicators,
    LazyConstraints,
    ConeConstraints,
    BranchPriorities,
}

impl std::fmt::Display for UnsupportedFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::MaximizationObjective => "maximization objectives",
            Self::ReferenceRow => "reference rows",
            Self::UserCuts => "user cuts",
            Self::SpecialOrderedSets => "special ordered sets",
            Self::QuadraticConstraints => "quadratic constraints",
            Self::Indicators => "indicator constraints",
            Self::LazyConstraints => "lazy constraints",
            Self::ConeConstraints => "cone constraints",
            Self::BranchPriorities => "branch priorities",
        };
        write!(f, "unsupported MPS feature: {name}")
    }
}

impl From<UnsupportedFeature> for PyErr {
    fn from(err: UnsupportedFeature) -> Self {
        UnsupportedMpsError::new_err(err.to_string())
    }
}

#[derive(Debug)]
pub(crate) struct ConversionFailure(pub String);

impl std::fmt::Display for ConversionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<ConversionFailure> for PyErr {
    fn from(err: ConversionFailure) -> Self {
        ConversionError::new_err(err.to_string())
    }
}
