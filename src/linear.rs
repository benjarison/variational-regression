use nalgebra::{Cholesky, DVector, DMatrix};
use crate::error::RegressionError;
use special::Gamma;
use crate::distribution::{GammaDistribution, GaussianDistribution};
use crate::math::LN_TWO_PI;
use crate::config::TrainConfig;
use serde::{Serialize, Deserialize};

type DenseVector = DVector<f64>;
type DenseMatrix = DMatrix<f64>;


#[derive(Serialize, Deserialize)]
pub struct VariationalLinearRegression {
    pub weights: DenseVector,
    pub covariance: DenseMatrix,
    pub noise_precision: GammaDistribution
}

impl VariationalLinearRegression {

    pub fn train(
        features: Vec<Vec<f64>>,
        labels: Vec<f64>,
        config: TrainConfig
    ) -> Result<VariationalLinearRegression, RegressionError> {

        let n = features.len();
        let d = features[0].len();
        let x = DenseMatrix::from_row_slice(
            n, d, features.into_iter().flatten().collect::<Vec<f64>>().as_slice()
        ).insert_column(0, 1.0);
        let y = DenseVector::from_vec(labels);
        let xtx = x.tr_mul(&x);
        let xty = x.tr_mul(&y);
        let yty = y.dot(&y);
        let mut alpha = vec![config.weight_prior; x.ncols()];
        let mut beta = config.noise_prior;
        let mut bound = f64::NEG_INFINITY;

        for iter in 0..config.max_iter {
            let (theta, s) = q_theta(&xtx, &xty, &alpha, &beta)?;
            alpha = q_alpha(&s, &theta, &config.weight_prior)?;
            beta = q_beta(&xtx, &xty, yty, &theta, &s, n, &config.noise_prior)?;
            let new_bound = lower_bound(&xtx, &xty, yty, &theta, &s, &alpha, &beta, &config.weight_prior, &config.noise_prior)?;
            println!("Iteration {}, Lower Bound = {}", iter + 1, new_bound);
            if (new_bound - bound) / bound.abs() <= config.tolerance {
                return Ok(VariationalLinearRegression {
                    weights: theta, covariance: s, noise_precision: beta
                })
            } else {
                bound = new_bound;
            }
        }

        let message = format!("Model failed to converge in {} iterations", config.max_iter);
        Err(RegressionError::from(message))
    }

    pub fn predict(&self, features: Vec<f64>) -> Result<GaussianDistribution, RegressionError> {
        let x = DenseVector::from_vec(features).insert_row(0, 1.0);
        let npm = self.noise_precision.mean();
        let pred_mean = x.dot(&self.weights);
        let pred_var = (1.0 / npm) + (&self.covariance * &x).dot(&x);
        GaussianDistribution::new(pred_mean, pred_var)
    }
}

fn q_theta(
    xtx: &DenseMatrix,
    xty: &DenseVector,
    alpha: &Vec<GammaDistribution>,
    beta: &GammaDistribution
) -> Result<(DenseVector, DenseMatrix), RegressionError> {

    let mut s_inv = xtx * beta.mean();
    for i in 0..alpha.len() {
        let a = alpha[i].mean();
        s_inv[(i, i)] += a;
    }
    let s = Cholesky::new(s_inv)
        .ok_or(RegressionError::from("cholesky error"))?
        .inverse();
    let theta = (&s * beta.mean()) * xty;
    Ok((theta, s))
}

fn q_alpha(
    s: &DenseMatrix,
    theta: &DenseVector,
    weight_prior: &GammaDistribution,
) -> Result<Vec<GammaDistribution>, RegressionError> {
    (0..theta.len()).map(|i| {
        let inv_scale = weight_prior.rate() + 0.5 * (theta[i] * theta[i] + s[(i, i)]);
        GammaDistribution::new(weight_prior.shape() + 0.5, inv_scale)
    }).collect::<Result<Vec<GammaDistribution>, RegressionError>>()
}

fn q_beta(
    xtx: &DenseMatrix,
    xty: &DenseVector,
    yty: f64,
    theta: &DenseVector,
    s: &DenseMatrix,
    n: usize,
    noise_prior: &GammaDistribution
) -> Result<GammaDistribution, RegressionError> {
    let shape = noise_prior.shape() + (n as f64 / 2.0);
    let t = (xtx * (theta * theta.transpose() + s)).trace();
    let inv_scale = noise_prior.rate() + 0.5 * (yty - 2.0 * theta.dot(xty) + t);
    GammaDistribution::new(shape, inv_scale)
}

fn lower_bound(
    xtx: &DenseMatrix, 
    xty: &DenseVector, 
    yty: f64,
    theta: &DenseVector,
    s: &DenseMatrix,
    alpha: &Vec<GammaDistribution>,
    beta: &GammaDistribution,
    weight_prior: &GammaDistribution,
    noise_prior: &GammaDistribution
) -> Result<f64, RegressionError> {
    Ok(expect_ln_p_y(xtx, xty, yty, &s, beta, theta)? +
    expect_ln_p_theta(&s, alpha, theta)? +
    expect_ln_p_alpha(alpha, weight_prior)? +
    expect_ln_p_beta(beta, noise_prior)? -
    expect_ln_q_theta(s)? -
    expect_ln_q_alpha(alpha)? -
    expect_ln_q_beta(beta)?)
}

fn expect_ln_p_y(
    xtx: &DenseMatrix,
    xty: &DenseVector,
    yty: f64,
    s: &DenseMatrix,
    beta: &GammaDistribution,
    theta: &DenseVector
) -> Result<f64, RegressionError> {
    let bm = beta.mean();
    let tc = theta * theta.transpose();
    let part1 = xty.len() as f64 * 0.5;
    let part2 = Gamma::digamma(beta.shape()) - beta.rate().ln() - LN_TWO_PI;
    let part3 = (bm * 0.5) * yty;
    let part4 = bm * theta.dot(xty);
    let part5 = (bm * 0.5) * (xtx * (tc + s)).trace();
    Ok(part1 * part2 - part3 + part4 - part5)
}

fn expect_ln_p_theta(
    s: &DenseMatrix,
    alpha: &Vec<GammaDistribution>,
    theta: &DenseVector
) -> Result<f64, RegressionError> {
    let init = (theta.len() as f64 * -0.5) * LN_TWO_PI;
    alpha.iter().enumerate().try_fold(init, |sum, (i, a)| {
        let am = a.mean();
        let part1 = Gamma::digamma(a.shape()) - a.rate().ln();
        let part2 = (theta[i] * theta[i] + s[(i, i)]) * am;
        Ok(sum + 0.5 * (part1 - part2))
    })
}

fn expect_ln_p_alpha(
    alpha: &Vec<GammaDistribution>,
    weight_prior: &GammaDistribution
) -> Result<f64, RegressionError> {
    alpha.iter().try_fold(0.0, |sum, a| {
        let am = a.mean();
        let term1 = weight_prior.shape() * weight_prior.rate().ln();
        let term2 = (weight_prior.shape() - 1.0) * (Gamma::digamma(a.shape()) - a.rate().ln());
        let term3 = (weight_prior.rate() * am) + Gamma::ln_gamma(weight_prior.shape()).0;
        Ok(sum + term1 + term2 - term3)
    })
}

fn expect_ln_p_beta(beta: &GammaDistribution, noise_prior: &GammaDistribution) -> Result<f64, RegressionError> {
    let part1 = noise_prior.shape() * noise_prior.rate().ln();
    let part2 = (noise_prior.shape() - 1.0) * (Gamma::digamma(beta.shape()) - beta.rate().ln());
    let part3 = (noise_prior.rate() * beta.mean()) + Gamma::ln_gamma(noise_prior.shape()).0;
    Ok(part1 + part2 - part3)
}

fn expect_ln_q_theta(s: &DenseMatrix) -> Result<f64, RegressionError> {
    let m = s.shape().0;
    let chol = Cholesky::new(s.clone()).unwrap().l();
    let mut ln_det = 0.0;
    for i in 0..s.ncols() {
        ln_det += chol[(i, i)].ln();
    }
    ln_det *= 2.0;
    Ok(-(0.5 * ln_det + (m as f64 / 2.0) * (1.0 + LN_TWO_PI)))
}

fn expect_ln_q_alpha(alpha: &Vec<GammaDistribution>) -> Result<f64, RegressionError> {
    alpha.iter().try_fold(0.0, |sum, a| {
        let part1 = Gamma::ln_gamma(a.shape()).0;
        let part2 = (a.shape() - 1.0) * Gamma::digamma(a.shape());
        let part3 = a.shape() - a.rate().ln();
        Ok(sum - (part1 - part2 + part3))
    })
}

fn expect_ln_q_beta(beta: &GammaDistribution) -> Result<f64, RegressionError> {
    Ok(-(Gamma::ln_gamma(beta.shape()).0 - 
    (beta.shape() - 1.0) * Gamma::digamma(beta.shape()) - 
    beta.rate().ln() + 
    beta.shape()))
}

