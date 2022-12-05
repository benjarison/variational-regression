//!
//! This crate provides functionality for regression models trained using variational inference
//!

pub mod distribution;
pub mod error;
pub mod linear;
pub mod logistic;
mod math;

pub use distribution::{ScalarDistribution, GammaDistribution, GaussianDistribution, BernoulliDistribution};
pub use linear::{VariationalLinearRegression, LinearTrainConfig};
pub use logistic::{VariationalLogisticRegression, LogisticTrainConfig};
pub use error::RegressionError;

use serde::{Serialize, Deserialize};
use nalgebra::{DMatrix, DVector};
type DenseMatrix = DMatrix<f64>;
type DenseVector  = DVector<f64>;

///
/// Represents a trained variational regression model with
/// the specified predictive distribution type
/// 
pub trait VariationalRegression<D: ScalarDistribution> {

    ///
    /// Computes the predictive distribution for the provided features
    /// 
    /// # Arguments
    /// 
    /// `features` - The input features
    /// 
    fn predict(&self, features: &[f64]) -> Result<D, RegressionError>;

    ///
    /// Provides the model weight parameters
    /// 
    fn weights(&self) -> Vec<Parameter>;

    ///
    /// Provides the model bias parameter, if specified for training
    /// 
    fn bias(&self) -> Option<Parameter>;
}

///
/// Represents two dimensional, dense feature data
/// 
pub trait Features {

    ///
    /// Processes the features into a dense matrix
    /// 
    /// # Arguments
    /// 
    /// `bias` - Whether or not to include a bias term, which
    ///          results in prepending a column of 1's
    ///          
    fn into_matrix(self, bias: bool) -> DenseMatrix;
}

impl Features for Vec<Vec<f64>> {
    fn into_matrix(self, bias: bool) -> DenseMatrix {
        (&self).into_matrix(bias)
    }
}

impl Features for &Vec<Vec<f64>> {
    fn into_matrix(self, bias: bool) -> DenseMatrix {
        let temp: Vec<_> = self.iter().map(|row| row.as_slice()).collect();
        temp.as_slice().into_matrix(bias)
    }
}

impl Features for &[&[f64]] {
    fn into_matrix(self, bias: bool) -> DenseMatrix {
        let offset = if bias { 1 } else { 0 };
        let n = self.len();
        let d = self[0].len() + offset;
        let mut x = DenseMatrix::zeros(n, d);
        for i in 0..n {
            if bias {
                x[(i, 0)] = 1.0;
            }
            for j in offset..d {
                x[(i, j)] = self[i][j - offset];
            }
        }
        return x;
    }
}

///
/// Represents continuous, real-valued label data
/// 
pub trait RealLabels {

    ///
    /// Processes the labels into a dense vector
    /// 
    fn into_vector(self) -> DenseVector;
}

impl RealLabels for Vec<f64> {
    fn into_vector(self) -> DenseVector {
        DenseVector::from_vec(self)
    }
}

impl RealLabels for &Vec<f64> {
    fn into_vector(self) -> DenseVector {
        self.as_slice().into_vector()
    }
}

impl RealLabels for &[f64] {
    fn into_vector(self) -> DenseVector {
        DenseVector::from_column_slice(self)
    }
}

///
/// Represents binary label data
/// 
pub trait BinaryLabels {

    ///
    /// Processes the labels into a dense vector
    /// 
    fn into_vector(self) -> DenseVector;
}

impl BinaryLabels for Vec<bool> {
    fn into_vector(self) -> DenseVector {
        self.as_slice().into_vector()
    }
}

impl BinaryLabels for &Vec<bool> {
    fn into_vector(self) -> DenseVector {
        self.as_slice().into_vector()
    }
}

impl BinaryLabels for &[bool] {
    fn into_vector(self) -> DenseVector {
        let iter = self.iter().map(|&label| if label { 1.0 } else { 0.0 });
        DenseVector::from_iterator(self.len(), iter)
    }
}

pub (crate) fn design_vector(features: &[f64], bias: bool) -> DenseVector {
    let offset = if bias { 1 } else { 0 };
    let d = features.len() + offset;
    let mut x = DenseVector::zeros(d);
    if bias {
        x[0] = 1.0;
    }
    for j in offset..d {
        x[j] = features[j - offset];
    }
    return x;
}

///
/// Represents a parameter (weight or bias)
/// from a trained regression model
/// 
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Parameter {
    /// The parameter value
    pub value: f64,
    /// The parameter standard error
    pub error: f64
}

impl std::fmt::Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.5} ({:.5})", self.value, self.error)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub (crate) struct Stats {
    pub mean: f64,
    pub sd: f64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub (crate) struct Standardizer {
    stats: Vec<Stats>
}

impl Standardizer {

    pub fn fit(features: &DenseMatrix) -> Standardizer {
        let stats = (0..features.ncols()).map(|j| {
            let mut sum = 0.0;
            for i in 0..features.nrows() {
                sum += features[(i, j)];
            }
            let mean = sum / features.nrows() as f64;
            let mut sse = 0.0;
            for i in 0..features.nrows() {
                let val = features[(i, j)];
                sse += (val - mean) * (val - mean);
            }
            let sd = (sse / (features.nrows() - 1) as f64).sqrt();
            Stats { mean, sd }
        }).collect();
        Standardizer { stats }
    }

    pub fn transform_matrix(&self, features: &mut DenseMatrix) {
        for j in 0..features.ncols() {
            let stats = &self.stats[j];
            if stats.sd > 0.0 {
                for i in 0..features.nrows() {
                    let val = features[(i, j)];
                    features[(i, j)] = (val - stats.mean) / stats.sd;
                }
            }
        }
    }

    pub fn transform_vector(&self, features: &mut DenseVector) {
        for j in 0..features.len() {
            let stats = &self.stats[j];
            if stats.sd > 0.0 {
                let val = features[j];
                features[j] = (val - stats.mean) / stats.sd;
            }
        }
    }
}

pub (crate) fn get_weights(includes_bias: bool, params: &DenseVector, covariance: &DenseMatrix) -> Vec<Parameter> {
    let offset = if includes_bias { 1 } else { 0 };
    params.as_slice()[offset..].iter()
    .zip(covariance.diagonal().as_slice()[offset..].iter())
    .map(|(&value, &variance)| Parameter { value, error: variance.sqrt() })
    .collect()
}

pub (crate) fn get_bias(includes_bias: bool, params: &DenseVector, covariance: &DenseMatrix) -> Option<Parameter> {
    if includes_bias {
        let value = params[0];
        let error = covariance[(0, 0)];
        Some( Parameter { value, error } )
    } else {
        None
    }
}
