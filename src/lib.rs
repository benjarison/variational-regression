use nalgebra::{DMatrix, DVector};

pub mod config;
pub mod error;
pub mod linear;
pub mod math;

pub type DenseVector = DVector<f64>;
pub type DenseMatrix = DMatrix<f64>;
