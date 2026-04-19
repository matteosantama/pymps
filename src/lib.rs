mod errors;
mod pymps;

use pyo3::prelude::*;

#[pymodule]
mod _pymps {
    use crate::errors::UnsupportedFeature;
    use mps::Parser;
    use mps::types::ObjectiveSense;
    use pyo3::prelude::*;
    use std::fs;

    #[pymodule_export]
    use crate::errors::{ConversionError, MpsError, ParseError, UnsupportedMpsError};
    #[pymodule_export]
    use crate::pymps::QuadraticProgram;

    #[pyfunction]
    fn read_f64(filepath: String) -> PyResult<QuadraticProgram> {
        let content = fs::read_to_string(&filepath)?;

        let parser =
            Parser::<f64>::parse(&content).map_err(|e| ParseError::new_err(e.to_string()))?;

        check_supported(&parser)?;

        let quadprog = QuadraticProgram::try_from(parser)?;
        Ok(quadprog)
    }

    fn check_supported(parser: &Parser<'_, f64>) -> Result<(), UnsupportedFeature> {
        if matches!(parser.objective_sense, Some(ObjectiveSense::Max)) {
            return Err(UnsupportedFeature::MaximizationObjective);
        }
        if parser.reference_row.is_some() {
            return Err(UnsupportedFeature::ReferenceRow);
        }
        if parser.user_cuts.is_some() {
            return Err(UnsupportedFeature::UserCuts);
        }
        if parser.special_ordered_sets.is_some() {
            return Err(UnsupportedFeature::SpecialOrderedSets);
        }
        if parser.quadratic_constraints.is_some() {
            return Err(UnsupportedFeature::QuadraticConstraints);
        }
        if parser.indicators.is_some() {
            return Err(UnsupportedFeature::Indicators);
        }
        if parser.lazy_constraints.is_some() {
            return Err(UnsupportedFeature::LazyConstraints);
        }
        if parser.cone_constraints.is_some() {
            return Err(UnsupportedFeature::ConeConstraints);
        }
        if parser.branch_priorities.is_some() {
            return Err(UnsupportedFeature::BranchPriorities);
        }
        Ok(())
    }
}
