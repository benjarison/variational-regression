pub mod distribution;
pub mod error;
pub mod linear;
pub mod logistic;
mod math;
mod util;

pub use distribution::ScalarDistribution;
pub use linear::{VariationalLinearRegression, LinearConfig};
pub use logistic::{VariationalLogisticRegression, LogisticConfig};
pub use error::RegressionError;
