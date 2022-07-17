use std::error::Error;
use std::fmt::{Display, Formatter, Debug};

///
/// Represents a regression-related error
/// 
#[derive(Debug)]
pub enum RegressionError {
    CholeskyFailure,
    ConvergenceFailure(usize),
    InvalidDistribution(String)
}

impl Display for RegressionError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            Self::CholeskyFailure => write!(f, "Cholesky Decomposition Failure"),
            Self::ConvergenceFailure(iter) => write!(f, "{}", format!("Failed to converge in {} iterations", iter)),
            Self::InvalidDistribution(msg) => write!(f, "{}", msg)
        }
    }
}

impl Error for RegressionError {}
