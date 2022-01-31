use nalgebra::{Cholesky, DVector, DMatrix};
use crate::error::RegressionError;
use special::Gamma;
use crate::distribution::{GammaDistribution, GaussianDistribution};
use crate::math::LN_2PI;
use crate::config::TrainConfig;
use serde::{Serialize, Deserialize};

type DenseVector = DVector<f64>;
type DenseMatrix = DMatrix<f64>;


///
/// Represents a linear regression model trained via variational inference
/// 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalLinearRegression {
    /// Learned model weights
    weights: DenseVector,
    /// Feature covariance matrix
    covariance: DenseMatrix,
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

    pub fn weights(&self) -> &Vec<f64> {
        self.weights.data.as_vec()
    }
}

// Defines the regression problem
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

// Factorized distribution for parameter weights
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

// Factorized distribution for weight precisions
fn q_alpha(prob: &mut Problem) -> Result<(), RegressionError> {
    for i in 0..prob.d {
        let inv_scale = prob.wpp.rate() + 0.5 * (prob.theta[i] * prob.theta[i] + prob.s[(i, i)]);
        prob.alpha[i] = GammaDistribution::new(prob.wpp.shape() + 0.5, inv_scale)?;
    }
    Ok(())
}

// Factorized distribution for noise precision
fn q_beta(prob: &mut Problem) -> Result<(), RegressionError> {
    let shape = prob.npp.shape() + (prob.n as f64 / 2.0);
    let t = (&prob.xtx * (&prob.theta * prob.theta.transpose() + &prob.s)).trace();
    let inv_scale = prob.npp.rate() + 0.5 * (prob.yty - 2.0 * prob.theta.dot(&prob.xty) + t);
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

// Expected log probability of labels conditioned on parameter weights
fn expect_ln_p_y(prob: &Problem) -> Result<f64, RegressionError> {
    let bm = prob.beta.mean();
    let tc = &prob.theta * prob.theta.transpose();
    let part1 = prob.xty.len() as f64 * 0.5;
    let part2 = Gamma::digamma(prob.beta.shape()) - prob.beta.rate().ln() - LN_2PI;
    let part3 = (bm * 0.5) * prob.yty;
    let part4 = bm * prob.theta.dot(&prob.xty);
    let part5 = (bm * 0.5) * (&prob.xtx * (tc + &prob.s)).trace();
    Ok(part1 * part2 - part3 + part4 - part5)
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

// Expected log probability of the noise precision
fn expect_ln_p_beta(prob: &Problem) -> Result<f64, RegressionError> {
    let part1 = prob.npp.shape() * prob.npp.rate().ln();
    let part2 = (prob.npp.shape() - 1.0) * (Gamma::digamma(prob.beta.shape()) - prob.beta.rate().ln());
    let part3 = (prob.npp.rate() * prob.beta.mean()) + Gamma::ln_gamma(prob.npp.shape()).0;
    Ok(part1 + part2 - part3)
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

// Expected entropy of the noise precision
fn expect_ln_q_beta(prob: &Problem) -> Result<f64, RegressionError> {
    Ok(-(Gamma::ln_gamma(prob.beta.shape()).0 - 
    (prob.beta.shape() - 1.0) * Gamma::digamma(prob.beta.shape()) - 
    prob.beta.rate().ln() + 
    prob.beta.shape()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainConfig;
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
    fn test_train() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = TrainConfig::default();
        let model = VariationalLinearRegression::train(x, y, config).unwrap();
        assert_approx_eq!(model.weights()[0], 0.10288069123755168);
        assert_approx_eq!(model.weights()[1], -0.11323826185472685);
        assert_approx_eq!(model.weights()[2], 0.024388910019891005);
        assert_approx_eq!(model.weights()[3], 0.9838454182808158);
        assert_approx_eq!(model.weights()[4], 0.45727622723291955);
    }

    #[test]
    fn test_predict() {
        let x = Vec::from(FEATURES.map(Vec::from));
        let y = Vec::from(LABELS);
        let config = TrainConfig::default();
        let model = VariationalLinearRegression::train(x, y, config).unwrap();
        let p = model.predict(vec![0.3, 0.8, -0.1, -0.3]).unwrap();
        assert_approx_eq!(p.mean(), -0.14714706930091104);
        assert_approx_eq!(p.variance(), 0.08453399119704738);
    }
}