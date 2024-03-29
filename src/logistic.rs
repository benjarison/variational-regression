use std::f64::consts::PI;
use nalgebra::{Cholesky, DVector, DMatrix};
use special::Gamma;
use serde::{Serialize, Deserialize};

use crate::{BinaryLabels, Features, design_vector, Standardizer, VariationalRegression, get_weights, get_bias};
use crate::distribution::{GammaDistribution, BernoulliDistribution, ScalarDistribution};
use crate::error::RegressionError;
use crate::math::{LN_2PI, logistic, trace_of_product, scale_rows};

type DenseVector = DVector<f64>;
type DenseMatrix = DMatrix<f64>;


///
/// Specifies configurable options for training a 
/// variational logistic regression model
/// 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticTrainConfig {
    /// Prior distribution over the precision of the model weights
    pub weight_precision_prior: GammaDistribution,
    /// Whether or not to include a bias term
    pub use_bias: bool,
    /// Whether or not to standardize the features
    pub standardize: bool,
    /// Maximum number of training iterations
    pub max_iter: usize,
    /// Convergence criteria threshold
    pub tolerance: f64,
    /// Indicates whether or not to print training info
    pub verbose: bool
}

impl Default for LogisticTrainConfig {
    fn default() -> Self {
        LogisticTrainConfig {
            weight_precision_prior: GammaDistribution::vague(),
            use_bias: true,
            standardize: true,
            max_iter: 1000, 
            tolerance: 1e-4,
            verbose: true
        }
    }
}

///
/// Represents a logistic regression model trained via variational inference
/// 
#[derive(Clone, Serialize, Deserialize)]
pub struct VariationalLogisticRegression {
    /// Learned model parameters
    params: DenseVector,
    /// Covariance matrix
    covariance: DenseMatrix,
    /// Whether the model was trained with a bias term or not
    includes_bias: bool,
    /// Optional feature standardizer
    standardizer: Option<Standardizer>,
    /// Variational lower bound
    pub bound: f64
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
        features: impl Features,
        labels: impl BinaryLabels,
        config: &LogisticTrainConfig
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
                    params: problem.theta, 
                    covariance: problem.s,
                    includes_bias: config.use_bias,
                    standardizer: problem.standardizer,
                    bound: new_bound
                })
            } else {
                problem.bound = new_bound;
            }
        }
        // admit defeat
        Err(RegressionError::ConvergenceFailure(config.max_iter))
    }
}

impl VariationalRegression<BernoulliDistribution> for VariationalLogisticRegression {

    fn predict(&self, features: &[f64]) -> Result<BernoulliDistribution, RegressionError> {
        let mut x = design_vector(features, self.includes_bias);
        if let Some(std) = &self.standardizer {
            std.transform_vector(&mut x);
        }
        let mu = x.dot(&self.params);
        let s = (&self.covariance * &x).dot(&x);
        let k = 1.0 / (1.0 + (PI * s) / 8.0).sqrt();
        let p = logistic(k * mu);
        BernoulliDistribution::new(p)
    }

    fn weights(&self) -> &[f64] {
        get_weights(self.includes_bias, &self.params)
    }

    fn bias(&self) -> Option<f64> {
        get_bias(self.includes_bias, &self.params)
    }
}

// Defines the regression problem
struct Problem {
    pub x: DenseMatrix, // features
    pub y: DenseVector, // labels
    pub xtr: DenseVector, // t(x) * (y - 0.5)
    pub theta: DenseVector, // parameters (bias & weights)
    pub s: DenseMatrix, // covariance
    pub alpha: Vec<GammaDistribution>, // parameter precisions
    pub zeta: DenseVector, // variational parameters
    pub bpp: Option<GammaDistribution>, // bias prior precision
    pub wpp: GammaDistribution, // weight prior precision
    pub n: usize, // number of training examples
    pub d: usize, // feature dimensionality (inclusing bias)
    pub bound: f64, // variational lower bound
    pub standardizer: Option<Standardizer> // feature standardizer
}

impl Problem {

    fn new(
        features: impl Features,
        labels: impl BinaryLabels,
        config: &LogisticTrainConfig
    ) -> Problem {
        let mut x = features.into_matrix(config.use_bias);
        let standardizer = if config.standardize {
            Some(Standardizer::fit(&x))
        } else {
            None
        };
        if let Some(std) = &standardizer {
            std.transform_matrix(&mut x);
        }
        let n = x.nrows();
        let d = x.ncols();
        let y = labels.into_vector();
        let xtr = x.tr_mul(&y.map(|v| v - 0.5));
        let wpp = config.weight_precision_prior;
        let bpp = if config.use_bias {
            Some(GammaDistribution::vague())
        } else {
            None
        };
        let mut alpha = vec![wpp; x.ncols()];
        if let Some(pp) = bpp {
            alpha[0] = pp;
        }
        let zeta = DenseVector::from_element(n, 1.0);
        let bound = f64::NEG_INFINITY;
        let theta = DenseVector::zeros(d);
        let s = DenseMatrix::zeros(d, d);
        Problem { x, y, xtr, theta, s, alpha, zeta, bpp, wpp, n, d, bound, standardizer }
    }

    fn param_precision_prior(&self, ind: usize) -> GammaDistribution {
        match (ind, self.bpp) {
            (0, Some(bpp)) => bpp,
            _ => self.wpp
        }
    }
}

fn lambda(val: f64) -> f64 {
    (1.0 / (2.0 * val)) * (logistic(val) - 0.5)
}

// Factorized distribution for parameter values
fn q_theta(prob: &mut Problem) -> Result<(), RegressionError> {
    let a = DenseVector::from(prob.alpha.iter().map(|alpha| alpha.mean()).collect::<Vec<f64>>());
    let mut s_inv = DenseMatrix::from_diagonal(&a);
    let lambdas = prob.zeta.map(|z| lambda(z));
    s_inv += prob.x.tr_mul(&scale_rows(&prob.x, &lambdas)) * 2.0;
    prob.s = Cholesky::new(s_inv)
        .ok_or(RegressionError::CholeskyFailure)?
        .inverse();
    prob.theta = &prob.s * &prob.xtr;
    Ok(())
}

// Factorized distribution for parameter precisions
fn q_alpha(prob: &mut Problem) -> Result<(), RegressionError> {
    for i in 0..prob.d {
        let pp = prob.param_precision_prior(i);
        let inv_scale = pp.rate + 0.5 * (prob.theta[i] * prob.theta[i] + prob.s[(i, i)]);
        prob.alpha[i] = GammaDistribution::new(pp.shape + 0.5, inv_scale)?;
    }
    Ok(())
}

// Update zeta values
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

// E[ln p(y|theta)]
fn expect_ln_p_y(prob: &Problem) -> Result<f64, RegressionError> {
    let part1 = prob.zeta.map(lambda);
    let part2 = prob.zeta.map(|z| logistic(z).ln()).sum();
    let part3 = &prob.x * &prob.theta;
    let part4 = (&part3.transpose() * prob.y.map(|y| y - 0.5)).sum();
    let part5 = prob.zeta.sum() / 2.0;
    let part6 = (part3.map(|v| v * v).transpose() * &part1).sum();
    let part7 = trace_of_product(&scale_rows(&(&prob.x * &prob.s), &part1), &prob.x.transpose());
    let part8 = part1.component_mul(&prob.zeta.map(|z| z * z)).sum();
    Ok(part2 + part4 - part5 - part6 - part7 + part8)
}

// E[ln p(theta|alpha)]
fn expect_ln_p_theta(prob: &Problem) -> Result<f64, RegressionError> {
    let init = (prob.theta.len() as f64 * -0.5) * LN_2PI;
    prob.alpha.iter().enumerate().try_fold(init, |sum, (i, a)| {
        let am = a.mean();
        let part1 = Gamma::digamma(a.shape) - a.rate.ln();
        let part2 = (prob.theta[i] * prob.theta[i] + prob.s[(i, i)]) * am;
        Ok(sum + 0.5 * (part1 - part2))
    })
}

// E[ln p(alpha)]
fn expect_ln_p_alpha(prob: &Problem) -> Result<f64, RegressionError> {
    prob.alpha.iter().enumerate().try_fold(0.0, |sum, (i, a)| {
        let am = a.mean();
        let pp = prob.param_precision_prior(i);
        let term1 = pp.shape * pp.rate.ln();
        let term2 = (pp.shape - 1.0) * (Gamma::digamma(a.shape) - a.rate.ln());
        let term3 = (pp.rate * am) + Gamma::ln_gamma(prob.wpp.shape).0;
        Ok(sum + term1 + term2 - term3)
    })
}

// E[ln q(theta)]
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

// E[ln q(alpha)]
fn expect_ln_q_alpha(prob: &Problem) -> Result<f64, RegressionError> {
    prob.alpha.iter().try_fold(0.0, |sum, a| {
        let part1 = Gamma::ln_gamma(a.shape).0;
        let part2 = (a.shape - 1.0) * Gamma::digamma(a.shape);
        let part3 = a.shape - a.rate.ln();
        Ok(sum - (part1 - part2 + part3))
    })
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
    fn test_train_with_bias_with_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LogisticTrainConfig::default();
        let model = VariationalLogisticRegression::train(&x, &y, &config).unwrap();
        assert_approx_eq!(model.bias().unwrap(), 0.00951244801510034);
        assert_approx_eq!(model.weights()[0], -0.19303165213334386);
        assert_approx_eq!(model.weights()[1], -1.2534945326354745);
        assert_approx_eq!(model.weights()[2], -0.6963518106208433);
        assert_approx_eq!(model.weights()[3], -0.8508100398896856);
    }

    #[test]
    fn test_train_with_bias_no_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LogisticTrainConfig {
            standardize: false,
            ..Default::default()
        };
        let model = VariationalLogisticRegression::train(&x, &y, &config).unwrap();
        assert_approx_eq!(model.bias().unwrap(), 0.0043520654824470515);
        assert_approx_eq!(model.weights()[0], -0.10946450049722892);
        assert_approx_eq!(model.weights()[1], -1.6472155373009127);
        assert_approx_eq!(model.weights()[2], -1.215877178138718);
        assert_approx_eq!(model.weights()[3], -0.7679465673373882);
    }

    #[test]
    fn test_train_no_bias_with_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LogisticTrainConfig {
            use_bias: false,
            ..Default::default()
        };
        let model = VariationalLogisticRegression::train(&x, &y, &config).unwrap();
        assert_approx_eq!(model.weights()[0], -0.22479264662358672);
        assert_approx_eq!(model.weights()[1], -1.194338553263914);
        assert_approx_eq!(model.weights()[2], -0.6763443319536045);
        assert_approx_eq!(model.weights()[3], -0.793934474799946);
    }

    #[test]
    fn test_train_no_bias_no_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LogisticTrainConfig {
            use_bias: false,
            standardize: false,
            ..Default::default()
        };
        let model = VariationalLogisticRegression::train(&x, &y, &config).unwrap();
        assert_approx_eq!(model.weights()[0], -0.11478846445757208);
        assert_approx_eq!(model.weights()[1], -1.6111314555274376);
        assert_approx_eq!(model.weights()[2], -1.0489256680896761);
        assert_approx_eq!(model.weights()[3], -0.6788653466293544);
    }

    #[test]
    fn test_predict_with_bias_with_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LogisticTrainConfig::default();
        let model = VariationalLogisticRegression::train(&x, &y, &config).unwrap();
        let p = model.predict(&vec![0.3, 0.8, -0.1, -0.3]).unwrap().mean();
        assert_approx_eq!(p, 0.27380317759208006);
    }

    #[test]
    fn test_predict_with_bias_no_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LogisticTrainConfig {
            standardize: false,
            ..Default::default()
        };
        let model = VariationalLogisticRegression::train(&x, &y, &config).unwrap();
        let p = model.predict(&vec![0.3, 0.8, -0.1, -0.3]).unwrap().mean();
        assert_approx_eq!(p, 0.2956358962602995);
    }

    #[test]
    fn test_predict_no_bias_with_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LogisticTrainConfig {
            use_bias: false,
            ..Default::default()
        };
        let model = VariationalLogisticRegression::train(&x, &y, &config).unwrap();
        let p = model.predict(&vec![0.3, 0.8, -0.1, -0.3]).unwrap().mean();
        assert_approx_eq!(p, 0.275642768184428);
    }

    #[test]
    fn test_predict_no_bias_no_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LogisticTrainConfig {
            use_bias: false,
            standardize: false,
            ..Default::default()
        };
        let model = VariationalLogisticRegression::train(&x, &y, &config).unwrap();
        let p = model.predict(&vec![0.3, 0.8, -0.1, -0.3]).unwrap().mean();
        assert_approx_eq!(p, 0.29090997574190514);
    }
}
