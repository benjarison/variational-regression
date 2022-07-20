use std::f64::consts::PI;
use nalgebra::{Cholesky, DVector, DMatrix};
use special::Gamma;
use serde::{Serialize, Deserialize};

use crate::distribution::{GammaDistribution, BernoulliDistribution};
use crate::error::RegressionError;
use crate::math::{LN_2PI, logistic};
use crate::util::{design_matrix, design_vector, trace_of_product};

type DenseVector = DVector<f64>;
type DenseMatrix = DMatrix<f64>;


///
/// Specifies configurable hyperparameters for training a 
/// variational linear regression model
/// 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Prior distribution over the precision of the model weights
    pub weight_precision_prior: GammaDistribution,
    /// Whether or not to include a bias term
    pub bias: bool,
    /// Maximum number of training iterations
    pub max_iter: usize,
    /// Convergence criteria threshold
    pub tolerance: f64,
    /// Indicates whether or not to print training info
    pub verbose: bool
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            weight_precision_prior: GammaDistribution::new(1e-4, 1e-4).unwrap(),
            bias: true,
            max_iter: 1000, 
            tolerance: 1e-4,
            verbose: true
        }
    }
}

///
/// Represents a logistic regression model trained via variational inference
/// 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalLogisticRegression {
    /// Learned model weights
    weights: DenseVector,
    /// Feature covariance matrix
    covariance: DenseMatrix,
    /// Whether or not the model uses a bias term
    pub bias: bool
}

impl VariationalLogisticRegression {

    ///
    /// Trains the model on the provided data
    /// 
    /// # Arguments
    /// 
    /// `features` - The feature values (in row-major orientation)
    /// `labels` - The vector of corresponding labels
    /// `config` - The training configuration
    /// 
    pub fn train(
        features: &Vec<Vec<f64>>,
        labels: &Vec<bool>,
        config: &TrainConfig
    ) -> Result<VariationalLogisticRegression, RegressionError> {
        // precompute required values
        let mut problem = Problem::new(features, labels, config);
        // optimize the variational lower bound until convergence
        for iter in 0..config.max_iter {
            q_theta(&mut problem)?; // model parameters
            q_alpha(&mut problem)?; // weight precisions
            update_zeta(&mut problem)?; // noise precision
            let new_bound = lower_bound(&problem)?;
            if config.verbose {
                println!("Iteration {}, Lower Bound = {}", iter + 1, new_bound);
            }
            if (new_bound - problem.bound) / problem.bound.abs() <= config.tolerance {
                return Ok(VariationalLogisticRegression {
                    weights: problem.theta, 
                    covariance: problem.s,
                    bias: config.bias
                })
            } else {
                problem.bound = new_bound;
            }
        }
        // admit defeat
        Err(RegressionError::ConvergenceFailure(config.max_iter))
    }

    ///
    /// Computes the predictive distribution for the provided features
    /// 
    /// # Arguments
    /// 
    /// `features` - The vector of feature values
    /// 
    pub fn predict(&self, features: &Vec<f64>) -> Result<BernoulliDistribution, RegressionError> {
        let x = design_vector(features, self.bias);
        let mu = x.dot(&self.weights);
        let s = (&self.covariance * &x).dot(&x);
        let k = 1.0 / (1.0 + (PI * s) / 8.0).sqrt();
        let p = logistic(k * mu);
        BernoulliDistribution::new(p)
    }

    ///
    /// Provides the trained model weights
    /// 
    pub fn weights(&self) -> &Vec<f64> {
        self.weights.data.as_vec()
    }
}

// Defines the regression problem
struct Problem {
    pub x: DenseMatrix,
    pub y: DenseVector,
    pub sxy: DenseVector,
    pub theta: DenseVector,
    pub s: DenseMatrix,
    pub alpha: Vec<GammaDistribution>,
    pub zeta: DenseVector,
    pub wpp: GammaDistribution,
    pub n: usize,
    pub d: usize,
    pub bound: f64
}

impl Problem {
    fn new(
        features: &Vec<Vec<f64>>,
        labels: &Vec<bool>,
        config: &TrainConfig
    ) -> Problem {
        let x = design_matrix(features, config.bias);
        let n = x.nrows();
        let d = x.ncols();
        let y = DenseVector::from_iterator(
            labels.len(), 
            labels.into_iter().map(|&lab| if lab {1.0} else {0.0})
        );
        let sxy = compute_sxy(&x, &y, d);
        let wpp = config.weight_precision_prior;
        let alpha = vec![wpp; x.ncols()];
        let zeta = DenseVector::from_element(n, 1.0);
        let bound = f64::NEG_INFINITY;
        let theta = DenseVector::zeros(d);
        let s = DenseMatrix::zeros(d, d);
        Problem {x, y, sxy, theta, s, alpha, zeta, wpp, n, d, bound}
    }
}

fn compute_sxy(x: &DenseMatrix, y: &DenseVector, d: usize) -> DenseVector {
    let mut sxy = DenseVector::zeros(d);
    for i in 0..y.len() {
        let t = y[i] - 0.5;
        sxy += x.row(i).transpose() * t;
    }
    sxy
}

fn lambda(val: f64) -> f64 {
    (1.0 / (2.0 * val)) * (logistic(val) - 0.5)
}

// Factorized distribution for parameter weights
fn q_theta(prob: &mut Problem) -> Result<(), RegressionError> {
    let a = DenseVector::from(prob.alpha.iter().map(|alpha| alpha.mean()).collect::<Vec<f64>>());
    let mut s_inv = DenseMatrix::from_diagonal(&a);
    let w = prob.zeta.map(|z| lambda(z) * 2.0);
    s_inv += prob.x.tr_mul(&scale(&prob.x, &w));
    prob.s = Cholesky::new(s_inv)
        .ok_or(RegressionError::CholeskyFailure)?
        .inverse();
    prob.theta = &prob.s * &prob.sxy;
    Ok(())
}

// Factorized distribution for weight precisions
fn q_alpha(prob: &mut Problem) -> Result<(), RegressionError> {
    for i in 0..prob.d {
        let inv_scale = prob.wpp.rate() + 0.5 * (prob.theta[i] * prob.theta[i] + prob.s[(i, i)]);
        prob.alpha[i] = GammaDistribution::new(prob.wpp.shape() + 0.5, inv_scale)?;
    }
    Ok(())
}

fn update_zeta(prob: &mut Problem) -> Result<(), RegressionError> {
    let a = &prob.s + (&prob.theta * prob.theta.transpose());
    let iter = prob.x.row_iter().map(|xi| {
        (&xi * &a).dot(&xi).sqrt()
    });
    prob.zeta = DenseVector::from_iterator(prob.n, iter);
    Ok(())
}

// Variational lower bound given current model parameters
fn lower_bound(prob: &Problem) -> Result<f64, RegressionError> {
    Ok(expect_ln_p_y(prob)? +
    expect_ln_p_theta(prob)? +
    expect_ln_p_alpha(prob)? -
    expect_ln_q_theta(prob)? -
    expect_ln_q_alpha(prob)?)
}

// Expected log probability of labels conditioned on parameter weights
fn expect_ln_p_y(prob: &Problem) -> Result<f64, RegressionError> {
    let part1 = prob.zeta.map(lambda);
    let part2 = prob.zeta.map(|z| logistic(z).ln()).sum();
    let part3 = &prob.x * &prob.theta;
    let part4 = (&part3.transpose() * prob.y.map(|y| y - 0.5)).sum();
    let part5 = prob.zeta.sum() / 2.0;
    let part6 = (part3.map(|v| v * v).transpose() * &part1).sum();
    let part7 = trace_of_product(&scale(&(&prob.x * &prob.s), &part1), &prob.x.transpose());
    let part8 = part1.component_mul(&prob.zeta.map(|z| z * z)).sum();
    Ok(part2 + part4 - part5 - part6 - part7 + part8)
}

// Expceted log probability of parameter weights conditioned on their precisions
fn expect_ln_p_theta(prob: &Problem) -> Result<f64, RegressionError> {
    let init = (prob.theta.len() as f64 * -0.5) * LN_2PI;
    prob.alpha.iter().enumerate().try_fold(init, |sum, (i, a)| {
        let am = a.mean();
        let part1 = Gamma::digamma(a.shape()) - a.rate().ln();
        let part2 = (prob.theta[i] * prob.theta[i] + prob.s[(i, i)]) * am;
        Ok(sum + 0.5 * (part1 - part2))
    })
}

// Expceted log probability of the parameter weight precisions
fn expect_ln_p_alpha(prob: &Problem) -> Result<f64, RegressionError> {
    prob.alpha.iter().try_fold(0.0, |sum, a| {
        let am = a.mean();
        let term1 = prob.wpp.shape() * prob.wpp.rate().ln();
        let term2 = (prob.wpp.shape() - 1.0) * (Gamma::digamma(a.shape()) - a.rate().ln());
        let term3 = (prob.wpp.rate() * am) + Gamma::ln_gamma(prob.wpp.shape()).0;
        Ok(sum + term1 + term2 - term3)
    })
}

// Expected entropy of the parameter weights
fn expect_ln_q_theta(prob: &Problem) -> Result<f64, RegressionError> {
    let m = prob.s.shape().0;
    let chol = Cholesky::new(prob.s.clone()).unwrap().l();
    let mut ln_det = 0.0;
    for i in 0..prob.s.ncols() {
        ln_det += chol[(i, i)].ln();
    }
    ln_det *= 2.0;
    Ok(-(0.5 * ln_det + (m as f64 / 2.0) * (1.0 + LN_2PI)))
}

// Expected entropy of the parameter precisions
fn expect_ln_q_alpha(prob: &Problem) -> Result<f64, RegressionError> {
    prob.alpha.iter().try_fold(0.0, |sum, a| {
        let part1 = Gamma::ln_gamma(a.shape()).0;
        let part2 = (a.shape() - 1.0) * Gamma::digamma(a.shape());
        let part3 = a.shape() - a.rate().ln();
        Ok(sum - (part1 - part2 + part3))
    })
}

fn scale(matrix: &DenseMatrix, vector: &DenseVector) -> DenseMatrix {
    let mut scaled = matrix.clone();
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            scaled[(i, j)] *= vector[i];
        }
    }
    return scaled;
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    const FEATURES: [[f64; 4]; 10] = [
        [-0.2, -0.9, -0.5, 0.3],
        [0.6, 0.3, 0.3, -0.4],
        [0.9, -0.4, -0.5, -0.6],
        [-0.7, 0.8, 0.3, -0.3],
        [-0.5, -0.7, -0.1, 0.8],
        [0.5, 0.5, 0.0, 0.1],
        [0.1, -0.0, 0.0, -0.2],
        [0.4, 0.0, 0.2, 0.0],
        [-0.2, 0.9, -0.1, -0.9],
        [0.1, 0.4, -0.5, 0.9],
    ];

    const LABELS: [bool; 10] = [
        true, false, true, false, true, false, true, false, true, false
    ];

    #[test]
    fn test_train() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = TrainConfig::default();
        let model = VariationalLogisticRegression::train(&x, &y, &config).unwrap();
        assert_approx_eq!(model.weights()[0], 0.0043520654824470515);
        assert_approx_eq!(model.weights()[1], -0.10946450049722892);
        assert_approx_eq!(model.weights()[2], -1.6472155373009127);
        assert_approx_eq!(model.weights()[3], -1.215877178138718);
        assert_approx_eq!(model.weights()[4], -0.7679465673373882);
    }

    #[test]
    fn test_predict() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = TrainConfig::default();
        let model = VariationalLogisticRegression::train(&x, &y, &config).unwrap();
        let p = model.predict(&vec![0.3, 0.8, -0.1, -0.3]).unwrap().mean();
        assert_approx_eq!(p, 0.2700659446128214);
    }
}
