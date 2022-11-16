//!
//! This crate provides functionality for regression models trained using variational inference
//!

pub mod distribution;
pub mod error;
pub mod linear;
pub mod logistic;
mod math;
mod util;

pub use distribution::{ScalarDistribution, GammaDistribution, GaussianDistribution, BernoulliDistribution};
pub use linear::{VariationalLinearRegression, LinearTrainConfig};
pub use logistic::{VariationalLogisticRegression, LogisticTrainConfig};
pub use error::RegressionError;
