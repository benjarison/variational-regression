use nalgebra::{Cholesky, DVector, DMatrix};
use special::Gamma;
use serde::{Serialize, Deserialize};

use crate::{RealLabels, Features, design_vector, Standardizer, VariationalRegression, get_weights, get_bias};
use crate::error::RegressionError;
use crate::distribution::{GammaDistribution, GaussianDistribution, ScalarDistribution};
use crate::math::LN_2PI;

type DenseVector = DVector<f64>;
type DenseMatrix = DMatrix<f64>;

///
/// Specifies configurable options for training a 
/// variational linear regression model
/// 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearTrainConfig {
    /// Prior distribution over the precision of the model weights
    pub weight_precision_prior: GammaDistribution,
    /// Prior distribution over the precision of the noise term
    pub noise_precision_prior: GammaDistribution,
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

impl Default for LinearTrainConfig {
    fn default() -> Self {
        LinearTrainConfig {
            weight_precision_prior: GammaDistribution::vague(),
            noise_precision_prior: GammaDistribution::new(1.0, 1e-4).unwrap(),
            use_bias: true,
            standardize: true,
            max_iter: 1000, 
            tolerance: 1e-4,
            verbose: true
        }
    }
}

///
/// Represents a linear regression model trained via variational inference
/// 
#[derive(Clone, Serialize, Deserialize)]
pub struct VariationalLinearRegression {
    /// Learned model weights
    params: DenseVector,
    /// Covariance matrix
    covariance: DenseMatrix,
    /// Whether the model was trained with a bias term or not
    includes_bias: bool,
    /// Optional feature standardizer
    standardizer: Option<Standardizer>,
    /// Noise precision distribution
    pub noise_precision: GammaDistribution,
    /// Variational lower bound
    pub bound: f64
}

impl VariationalLinearRegression {

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
        labels: impl RealLabels,
        config: &LinearTrainConfig
    ) -> Result<VariationalLinearRegression, RegressionError> {
        // precompute required values
        let mut problem = Problem::new(features, labels, config);
        // optimize the variational lower bound until convergence
        for iter in 0..config.max_iter {
            q_theta(&mut problem)?; // model parameters
            q_alpha(&mut problem)?; // weight precisions
            q_beta(&mut problem)?; // noise precision
            let new_bound = lower_bound(&problem)?;
            if config.verbose {
                println!("Iteration {}, Lower Bound = {}", iter + 1, new_bound);
            }
            if (new_bound - problem.bound) / problem.bound.abs() <= config.tolerance {
                return Ok(VariationalLinearRegression {
                    params: problem.theta, 
                    covariance: problem.s, 
                    includes_bias: config.use_bias,
                    standardizer: problem.standardizer,
                    noise_precision: problem.beta, 
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

impl VariationalRegression<GaussianDistribution> for VariationalLinearRegression {

    fn predict(&self, features: &[f64]) -> Result<GaussianDistribution, RegressionError> {
        let mut x = design_vector(features, self.includes_bias);
        if let Some(std) = &self.standardizer {
            std.transform_vector(&mut x);
        }
        let npm = self.noise_precision.mean();
        let pred_mean = x.dot(&self.params);
        let pred_var = (1.0 / npm) + (&self.covariance * &x).dot(&x);
        GaussianDistribution::new(pred_mean, pred_var)
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
    pub xtx: DenseMatrix, // t(x) * x
    pub xty: DenseVector, // t(x) * y
    pub yty: f64, // t(y) * y
    pub theta: DenseVector, // parameters (bias & weights)
    pub s: DenseMatrix, // covariance
    pub alpha: Vec<GammaDistribution>, // parameter precisions
    pub beta: GammaDistribution, // noise precision
    pub bpp: Option<GammaDistribution>, // bias prior precision
    pub wpp: GammaDistribution, // weight prior precision
    pub npp: GammaDistribution, // noise prior precision
    pub n: usize, // number of training examples
    pub d: usize, // feature dimensionality (including bias)
    pub bound: f64, // variational lower bound
    pub standardizer: Option<Standardizer> // feature standardizer
}

impl Problem {

    fn new(
        features: impl Features,
        labels: impl RealLabels,
        config: &LinearTrainConfig
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
        let xtx = x.tr_mul(&x);
        let xty = x.tr_mul(&y);
        let yty = y.dot(&y);
        let bpp = if config.use_bias {
            Some(GammaDistribution::vague())
        } else {
            None
        };
        let wpp = config.weight_precision_prior;
        let npp = config.noise_precision_prior;
        let mut alpha = vec![wpp; x.ncols()];
        if let Some(pp) = bpp {
            alpha[0] = pp;
        }
        let beta = npp;
        let bound = f64::NEG_INFINITY;
        let theta = DenseVector::zeros(d);
        let s = DenseMatrix::zeros(d, d);
        Problem { xtx, xty, yty, theta, s, alpha, beta, bpp, wpp, npp, n, d, bound, standardizer }
    }

    fn param_precision_prior(&self, ind: usize) -> GammaDistribution {
        match (ind, self.bpp) {
            (0, Some(bpp)) => bpp,
            _ => self.wpp
        }
    }
}

// Factorized distribution for parameter values
fn q_theta(prob: &mut Problem) -> Result<(), RegressionError> {
    let mut s_inv = &prob.xtx * prob.beta.mean();
    for i in 0..prob.d {
        let a = prob.alpha[i].mean();
        s_inv[(i, i)] += a;
    }
    prob.s = Cholesky::new(s_inv)
        .ok_or(RegressionError::CholeskyFailure)?
        .inverse();
    prob.theta = (&prob.s * prob.beta.mean()) * &prob.xty;
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

// Factorized distribution for noise precision
fn q_beta(prob: &mut Problem) -> Result<(), RegressionError> {
    let shape = prob.npp.shape + (prob.n as f64 / 2.0);
    let t = (&prob.xtx * (&prob.theta * prob.theta.transpose() + &prob.s)).trace();
    let inv_scale = prob.npp.rate + 0.5 * (prob.yty - 2.0 * prob.theta.dot(&prob.xty) + t);
    prob.beta = GammaDistribution::new(shape, inv_scale)?;
    Ok(())
}

// Variational lower bound given current model parameters
fn lower_bound(prob: &Problem) -> Result<f64, RegressionError> {
    Ok(expect_ln_p_y(prob)? +
    expect_ln_p_theta(prob)? +
    expect_ln_p_alpha(prob)? +
    expect_ln_p_beta(prob)? -
    expect_ln_q_theta(prob)? -
    expect_ln_q_alpha(prob)? -
    expect_ln_q_beta(prob)?)
}

// E[ln p(y|theta)]
fn expect_ln_p_y(prob: &Problem) -> Result<f64, RegressionError> {
    let bm = prob.beta.mean();
    let tc = &prob.theta * prob.theta.transpose();
    let part1 = prob.xty.len() as f64 * 0.5;
    let part2 = Gamma::digamma(prob.beta.shape) - prob.beta.rate.ln() - LN_2PI;
    let part3 = (bm * 0.5) * prob.yty;
    let part4 = bm * prob.theta.dot(&prob.xty);
    let part5 = (bm * 0.5) * (&prob.xtx * (tc + &prob.s)).trace();
    Ok(part1 * part2 - part3 + part4 - part5)
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
        let term3 = (pp.rate * am) + Gamma::ln_gamma(pp.shape).0;
        Ok(sum + term1 + term2 - term3)
    })
}

// E[ln p(beta)]
fn expect_ln_p_beta(prob: &Problem) -> Result<f64, RegressionError> {
    let part1 = prob.npp.shape * prob.npp.rate.ln();
    let part2 = (prob.npp.shape - 1.0) * (Gamma::digamma(prob.beta.shape) - prob.beta.rate.ln());
    let part3 = (prob.npp.rate * prob.beta.mean()) + Gamma::ln_gamma(prob.npp.shape).0;
    Ok(part1 + part2 - part3)
}

// E[ln q(theta)]
fn expect_ln_q_theta(prob: &Problem) -> Result<f64, RegressionError> {
    let m = prob.s.shape().0;
    let chol = Cholesky::new(prob.s.clone())
    .ok_or(RegressionError::CholeskyFailure)?
    .l();
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

// E[ln q(beta)]
fn expect_ln_q_beta(prob: &Problem) -> Result<f64, RegressionError> {
    Ok(-(Gamma::ln_gamma(prob.beta.shape).0 - 
    (prob.beta.shape - 1.0) * Gamma::digamma(prob.beta.shape) - 
    prob.beta.rate.ln() + 
    prob.beta.shape))
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

    const LABELS: [f64; 10] = [
        -0.4, 0.1, -0.8, 0.5, 0.6, -0.2, 0.0, 0.7, -0.3, 0.2
    ];

    #[test]
    fn test_train_with_bias_with_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LinearTrainConfig {
            noise_precision_prior: GammaDistribution { shape: 1.0001, rate: 1e-4 },
            ..Default::default()
        };
        let model = VariationalLinearRegression::train(&x, &y, &config).unwrap();
        assert_approx_eq!(model.bias().unwrap(), 0.009795973392064526);
        assert_approx_eq!(model.weights()[0], -0.053736076572620695);
        assert_approx_eq!(model.weights()[1], 0.002348926942734912);
        assert_approx_eq!(model.weights()[2], 0.36479166380848826);
        assert_approx_eq!(model.weights()[3], 0.2995772527448547);
    }

    #[test]
    fn test_train_with_bias_no_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LinearTrainConfig {
            standardize: false,
            noise_precision_prior: GammaDistribution { shape: 1.0001, rate: 1e-4 },
            ..Default::default()
        };
        let model = VariationalLinearRegression::train(&x, &y, &config).unwrap();
        assert_approx_eq!(model.bias().unwrap(), 0.14022283613177447);
        assert_approx_eq!(model.weights()[0], -0.08826080780896867);
        assert_approx_eq!(model.weights()[1], 0.003684347234472394);
        assert_approx_eq!(model.weights()[2], 1.1209335465339734);
        assert_approx_eq!(model.weights()[3], 0.5137103057008632);
    }

    #[test]
    fn test_train_no_bias_with_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LinearTrainConfig {
            use_bias: false,
            noise_precision_prior: GammaDistribution { shape: 1.0001, rate: 1e-4 },
            ..Default::default()
        };
        let model = VariationalLinearRegression::train(&x, &y, &config).unwrap();
        assert_approx_eq!(model.weights()[0], -0.0536007042304908);
        assert_approx_eq!(model.weights()[1], 0.0024537840396777044);
        assert_approx_eq!(model.weights()[2], 0.3649008472250164);
        assert_approx_eq!(model.weights()[3], 0.2997887456881104);
    }

    #[test]
    fn test_train_no_bias_no_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LinearTrainConfig {
            use_bias: false,
            standardize: false,
            noise_precision_prior: GammaDistribution { shape: 1.0001, rate: 1e-4 },
            ..Default::default()
        };
        let model = VariationalLinearRegression::train(&x, &y, &config).unwrap();
        assert_approx_eq!(model.weights()[0], -0.0362564312306051);
        assert_approx_eq!(model.weights()[1], 0.021598779423334057);
        assert_approx_eq!(model.weights()[2], 0.9458928058270641);
        assert_approx_eq!(model.weights()[3], 0.4751696529319309);
    }

    #[test]
    fn test_predict_with_bias_with_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LinearTrainConfig {
            noise_precision_prior: GammaDistribution { shape: 1.0001, rate: 1e-4 },
            ..Default::default()
        };
        let model = VariationalLinearRegression::train(&x, &y, &config).unwrap();
        let p = model.predict(&vec![0.3, 0.8, -0.1, -0.3]).unwrap();
        assert_approx_eq!(p.mean(), -0.1601830957057508);
        assert_approx_eq!(p.variance(), 0.0421041223659715);
    }

    #[test]
    fn test_predict_with_bias_no_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LinearTrainConfig {
            standardize: false,
            noise_precision_prior: GammaDistribution { shape: 1.0001, rate: 1e-4 },
            ..Default::default()
        };
        let model = VariationalLinearRegression::train(&x, &y, &config).unwrap();
        let p = model.predict(&vec![0.3, 0.8, -0.1, -0.3]).unwrap();
        assert_approx_eq!(p.mean(), -0.1495143747869945);
        assert_approx_eq!(p.variance(), 0.047374206616233275);
    }

    #[test]
    fn test_predict_no_bias_with_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LinearTrainConfig {
            use_bias: false,
            noise_precision_prior: GammaDistribution { shape: 1.0001, rate: 1e-4 },
            ..Default::default()
        };
        let model = VariationalLinearRegression::train(&x, &y, &config).unwrap();
        let p = model.predict(&vec![0.3, 0.8, -0.1, -0.3]).unwrap();
        assert_approx_eq!(p.mean(), -0.16990565682335487);
        assert_approx_eq!(p.variance(), 0.0409272332865222);
    }

    #[test]
    fn test_predict_no_bias_no_standardize() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = LinearTrainConfig {
            use_bias: false,
            standardize: false,
            noise_precision_prior: GammaDistribution { shape: 1.0001, rate: 1e-4 },
            ..Default::default()
        };
        let model = VariationalLinearRegression::train(&x, &y, &config).unwrap();
        let p = model.predict(&vec![0.3, 0.8, -0.1, -0.3]).unwrap();
        assert_approx_eq!(p.mean(), -0.2307380822928);
        assert_approx_eq!(p.variance(), 0.07177809358927849);
    }
}
