use nalgebra::{Cholesky, DVector, DMatrix};
use crate::error::RegressionError;
use special::Gamma;
use crate::distribution::{GammaDistribution, GaussianDistribution};
use crate::math::LN_TWO_PI;
use crate::config::TrainConfig;
use serde::{Serialize, Deserialize};

type DenseVector = DVector<f64>;
type DenseMatrix = DMatrix<f64>;


///
/// Represents a linear regression model trained via variational inference
/// 
#[derive(Serialize, Deserialize)]
pub struct VariationalLinearRegression {
    /// Learned model weights
    pub weights: DenseVector,
    /// Feature covariance matrix
    pub covariance: DenseMatrix,
    /// Noise precision distribution
    pub noise_precision: GammaDistribution
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
        features: Vec<Vec<f64>>,
        labels: Vec<f64>,
        config: TrainConfig
    ) -> Result<VariationalLinearRegression, RegressionError> {
        // precompute required values
        let mut problem = Problem::new(features, labels, &config);
        // optimize the variational lower bound until convergence
        for iter in 0..config.max_iter {
            q_theta(&mut problem)?; // model parameters
            q_alpha(&mut problem)?; // weight precisions
            q_beta(&mut problem)?; // noise precision
            let new_bound = lower_bound(&problem)?;
            println!("Iteration {}, Lower Bound = {}", iter + 1, new_bound);
            if (new_bound - problem.bound) / problem.bound.abs() <= config.tolerance {
                return Ok(VariationalLinearRegression {
                    weights: problem.theta, covariance: problem.s, noise_precision: problem.beta
                })
            } else {
                problem.bound = new_bound;
            }
        }
        // admit defeat
        let message = format!("Model failed to converge in {} iterations", config.max_iter);
        Err(RegressionError::from(message))
    }

    ///
    /// Computes the predictive distribution for the provided features
    /// 
    /// # Arguments
    /// 
    /// `features` - The vector of feature values
    /// 
    pub fn predict(&self, features: Vec<f64>) -> Result<GaussianDistribution, RegressionError> {
        let x = DenseVector::from_vec(features).insert_row(0, 1.0);
        let npm = self.noise_precision.mean();
        let pred_mean = x.dot(&self.weights);
        let pred_var = (1.0 / npm) + (&self.covariance * &x).dot(&x);
        GaussianDistribution::new(pred_mean, pred_var)
    }
}

struct Problem {
    pub xtx: DenseMatrix,
    pub xty: DenseVector,
    pub yty: f64,
    pub theta: DenseVector,
    pub s: DenseMatrix,
    pub alpha: Vec<GammaDistribution>,
    pub beta: GammaDistribution,
    pub wpp: GammaDistribution,
    pub npp: GammaDistribution,
    pub n: usize,
    pub d: usize,
    pub bound: f64
}

impl Problem {
    fn new(
        features: Vec<Vec<f64>>,
        labels: Vec<f64>,
        config: &TrainConfig
    ) -> Problem {
        let n = features.len();
        let d = features[0].len() + 1;
        let x = DenseMatrix::from_row_slice(
            n, d - 1, features.into_iter().flatten().collect::<Vec<f64>>().as_slice()
        ).insert_column(0, 1.0);
        let y = DenseVector::from_vec(labels);
        let xtx = x.tr_mul(&x);
        let xty = x.tr_mul(&y);
        let yty = y.dot(&y);
        let wpp = config.weight_precision_prior;
        let npp = config.noise_precision_prior;
        let alpha = vec![wpp; x.ncols()];
        let beta = npp;
        let bound = f64::NEG_INFINITY;
        let theta = DenseVector::zeros(d);
        let s = DenseMatrix::zeros(d, d);
        Problem {xtx, xty, yty, theta, s, alpha, beta, wpp, npp, n, d, bound}
    }
}

fn q_theta(prob: &mut Problem) -> Result<(), RegressionError> {
    let mut s_inv = &prob.xtx * prob.beta.mean();
    for i in 0..prob.d {
        let a = prob.alpha[i].mean();
        s_inv[(i, i)] += a;
    }
    prob.s = Cholesky::new(s_inv)
        .ok_or(RegressionError::from("cholesky error"))?
        .inverse();
    prob.theta = (&prob.s * prob.beta.mean()) * &prob.xty;
    Ok(())
}

fn q_alpha(prob: &mut Problem) -> Result<(), RegressionError> {
    for i in 0..prob.d {
        let inv_scale = prob.wpp.rate() + 0.5 * (prob.theta[i] * prob.theta[i] + prob.s[(i, i)]);
        prob.alpha[i] = GammaDistribution::new(prob.wpp.shape() + 0.5, inv_scale)?;
    }
    Ok(())
}

fn q_beta(prob: &mut Problem) -> Result<(), RegressionError> {
    let shape = prob.npp.shape() + (prob.n as f64 / 2.0);
    let t = (&prob.xtx * (&prob.theta * prob.theta.transpose() + &prob.s)).trace();
    let inv_scale = prob.npp.rate() + 0.5 * (prob.yty - 2.0 * prob.theta.dot(&prob.xty) + t);
    prob.beta = GammaDistribution::new(shape, inv_scale)?;
    Ok(())
}

fn lower_bound(prob: &Problem) -> Result<f64, RegressionError> {
    Ok(expect_ln_p_y(prob)? +
    expect_ln_p_theta(prob)? +
    expect_ln_p_alpha(prob)? +
    expect_ln_p_beta(prob)? -
    expect_ln_q_theta(prob)? -
    expect_ln_q_alpha(prob)? -
    expect_ln_q_beta(prob)?)
}

fn expect_ln_p_y(prob: &Problem) -> Result<f64, RegressionError> {
    let bm = prob.beta.mean();
    let tc = &prob.theta * prob.theta.transpose();
    let part1 = prob.xty.len() as f64 * 0.5;
    let part2 = Gamma::digamma(prob.beta.shape()) - prob.beta.rate().ln() - LN_TWO_PI;
    let part3 = (bm * 0.5) * prob.yty;
    let part4 = bm * prob.theta.dot(&prob.xty);
    let part5 = (bm * 0.5) * (&prob.xtx * (tc + &prob.s)).trace();
    Ok(part1 * part2 - part3 + part4 - part5)
}

fn expect_ln_p_theta(prob: &Problem) -> Result<f64, RegressionError> {
    let init = (prob.theta.len() as f64 * -0.5) * LN_TWO_PI;
    prob.alpha.iter().enumerate().try_fold(init, |sum, (i, a)| {
        let am = a.mean();
        let part1 = Gamma::digamma(a.shape()) - a.rate().ln();
        let part2 = (prob.theta[i] * prob.theta[i] + prob.s[(i, i)]) * am;
        Ok(sum + 0.5 * (part1 - part2))
    })
}

fn expect_ln_p_alpha(prob: &Problem) -> Result<f64, RegressionError> {
    prob.alpha.iter().try_fold(0.0, |sum, a| {
        let am = a.mean();
        let term1 = prob.wpp.shape() * prob.wpp.rate().ln();
        let term2 = (prob.wpp.shape() - 1.0) * (Gamma::digamma(a.shape()) - a.rate().ln());
        let term3 = (prob.wpp.rate() * am) + Gamma::ln_gamma(prob.wpp.shape()).0;
        Ok(sum + term1 + term2 - term3)
    })
}

fn expect_ln_p_beta(prob: &Problem) -> Result<f64, RegressionError> {
    let part1 = prob.npp.shape() * prob.npp.rate().ln();
    let part2 = (prob.npp.shape() - 1.0) * (Gamma::digamma(prob.beta.shape()) - prob.beta.rate().ln());
    let part3 = (prob.npp.rate() * prob.beta.mean()) + Gamma::ln_gamma(prob.npp.shape()).0;
    Ok(part1 + part2 - part3)
}

fn expect_ln_q_theta(prob: &Problem) -> Result<f64, RegressionError> {
    let m = prob.s.shape().0;
    let chol = Cholesky::new(prob.s.clone()).unwrap().l();
    let mut ln_det = 0.0;
    for i in 0..prob.s.ncols() {
        ln_det += chol[(i, i)].ln();
    }
    ln_det *= 2.0;
    Ok(-(0.5 * ln_det + (m as f64 / 2.0) * (1.0 + LN_TWO_PI)))
}

fn expect_ln_q_alpha(prob: &Problem) -> Result<f64, RegressionError> {
    prob.alpha.iter().try_fold(0.0, |sum, a| {
        let part1 = Gamma::ln_gamma(a.shape()).0;
        let part2 = (a.shape() - 1.0) * Gamma::digamma(a.shape());
        let part3 = a.shape() - a.rate().ln();
        Ok(sum - (part1 - part2 + part3))
    })
}

fn expect_ln_q_beta(prob: &Problem) -> Result<f64, RegressionError> {
    Ok(-(Gamma::ln_gamma(prob.beta.shape()).0 - 
    (prob.beta.shape() - 1.0) * Gamma::digamma(prob.beta.shape()) - 
    prob.beta.rate().ln() + 
    prob.beta.shape()))
}

