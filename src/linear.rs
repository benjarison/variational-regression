use nalgebra::Cholesky;
use statrs::distribution::Gamma;
use statrs::statistics::Distribution;
use crate::error::RegressionError;
use statrs::function::gamma::{digamma, ln_gamma};
use crate::math::LN_TWO_PI;
use crate::config::TrainConfig;
use crate::{DenseVector, DenseMatrix};

pub struct VariationalLinearRegression {
    pub weights: DenseVector,
    pub covariance: DenseMatrix,
    pub noise_precision: Gamma
}

impl VariationalLinearRegression {

    pub fn train(
        features: DenseMatrix,
        labels: DenseVector,
        config: TrainConfig
    ) -> Result<VariationalLinearRegression, RegressionError> {

        let x = features.insert_column(0, 1.0);
        let y = labels;
        let xtx = x.tr_mul(&x);
        let xty = x.tr_mul(&y);
        let yty = y.dot(&y);
        let mut alpha = vec![config.weight_prior; x.ncols()];
        let mut beta = config.noise_prior;
        let mut bound = f64::NEG_INFINITY;

        for iter in 0..config.max_iter {
            let (theta, s) = q_theta(&xtx, &xty, &alpha, &beta)?;
            alpha = q_alpha(&s, &theta, &config.weight_prior)?;
            beta = q_beta(&xtx, &xty, yty, &theta, &s, &config.noise_prior)?;
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


}

fn q_theta(
    xtx: &DenseMatrix,
    xty: &DenseVector,
    alpha: &Vec<Gamma>,
    beta: &Gamma
) -> Result<(DenseVector, DenseMatrix), RegressionError> {

    let b = beta.mean().ok_or(RegressionError::from("noise precision error"))?;
    let mut s_inv = xtx * b;
    for i in 0..alpha.len() {
        let a = alpha[i]
            .mean()
            .ok_or(RegressionError::from("weight precision error"))?;
        s_inv[(i, i)] += a;
    }
    let s = Cholesky::new(s_inv)
        .ok_or(RegressionError::from("cholesky error"))?
        .inverse();
    let theta = (&s * b) * xty;
    Ok((theta, s))
}

fn q_alpha(
    s: &DenseMatrix,
    theta: &DenseVector,
    weight_prior: &Gamma,
) -> Result<Vec<Gamma>, RegressionError> {
    (0..theta.len()).map(|i| {
        let inv_scale = weight_prior.rate() + 0.5 * (theta[i] * theta[i] + s[(i, i)]);
        Gamma::new(weight_prior.shape() + 0.5, inv_scale).map_err(RegressionError::from)
    }).collect::<Result<Vec<Gamma>, RegressionError>>()
}

fn q_beta(
    xtx: &DenseMatrix,
    xty: &DenseVector,
    yty: f64,
    theta: &DenseVector,
    s: &DenseMatrix,
    noise_prior: &Gamma
) -> Result<Gamma, RegressionError> {
    let shape = noise_prior.shape() + (xty.len() as f64 / 2.0);
    let t = (xtx * (theta * theta.transpose() + s)).trace();
    let inv_scale = noise_prior.rate() + 0.5 * (yty - 2.0 * theta.dot(xty) + t);
    Gamma::new(shape, inv_scale).map_err(RegressionError::from)
}

fn lower_bound(
    xtx: &DenseMatrix, 
    xty: &DenseVector, 
    yty: f64,
    theta: &DenseVector,
    s: &DenseMatrix,
    alpha: &Vec<Gamma>,
    beta: &Gamma,
    weight_prior: &Gamma,
    noise_prior: &Gamma
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
    beta: &Gamma,
    theta: &DenseVector
) -> Result<f64, RegressionError> {
    let bm = beta.mean().ok_or(RegressionError::from("invalid gamma mean"))?;
    let tc = theta * theta.transpose();
    let part1 = xty.len() as f64 * 0.5;
    let part2 = digamma(beta.shape()) - beta.rate().ln() - LN_TWO_PI;
    let part3 = (bm * 0.5) * yty;
    let part4 = bm * theta.dot(xty);
    let part5 = (bm * 0.5) * (xtx * (tc + s)).trace();
    Ok(part1 * part2 - part3 + part4 - part5)
}

fn expect_ln_p_theta(
    s: &DenseMatrix,
    alpha: &Vec<Gamma>,
    theta: &DenseVector
) -> Result<f64, RegressionError> {
    let init = (theta.len() as f64 * -0.5) * LN_TWO_PI;
    alpha.iter().enumerate().try_fold(init, |sum, (i, a)| {
        let am = alpha[i].mean().ok_or(RegressionError::from("invalid gamma mean"))?;
        let part1 = digamma(alpha[i].shape()) - alpha[i].rate().ln();
        let part2 = (theta[i] * theta[i] + s[(i, i)]) * am;
        Ok(sum + 0.5 * (part1 - part2))
    })
}

fn expect_ln_p_alpha(
    alpha: &Vec<Gamma>,
    weight_prior: &Gamma
) -> Result<f64, RegressionError> {
    alpha.iter().try_fold(0.0, |sum, a| {
        let a_mean = a.mean().ok_or(RegressionError::from("invalid gamma mean"))?;
        let term1 = weight_prior.shape() * weight_prior.rate().ln();
        let term2 = (weight_prior.shape() - 1.0) * (digamma(a.shape()) - a.rate().ln());
        let term3 = (weight_prior.rate() * a_mean) - ln_gamma(weight_prior.shape());
        Ok(sum + term1 + term2 - term3)
    })
}

fn expect_ln_p_beta(beta: &Gamma, noise_prior: &Gamma) -> Result<f64, RegressionError> {
    let part1 = noise_prior.shape() * noise_prior.rate().ln();
    let part2 = (noise_prior.shape() - 1.0) * (digamma(beta.shape()) - beta.rate().ln());
    let part3 = (noise_prior.rate() * beta.mean().unwrap()) + ln_gamma(noise_prior.shape());
    Ok(part1 + part2 - part3)
}

fn expect_ln_q_theta(s: &DenseMatrix) -> Result<f64, RegressionError> {
    let m = s.shape().0;
    let chol = Cholesky::new(s.clone()).unwrap();
    Ok(-0.5 * chol.determinant().ln() + (m as f64 / 2.0) * (1.0 + LN_TWO_PI))
}

fn expect_ln_q_alpha(alpha: &Vec<Gamma>) -> Result<f64, RegressionError> {
    alpha.iter().try_fold(0.0, |sum, a| {
        let part1 = ln_gamma(a.shape());
        let part2 = (a.shape() - 1.0) * digamma(a.shape());
        let part3 = a.shape() - a.rate().ln();
        Ok(sum - (part1 - part2 + part3))
    })
}

fn expect_ln_q_beta(beta: &Gamma) -> Result<f64, RegressionError> {
    Ok(-ln_gamma(beta.shape()) - 
    (beta.shape() - 1.0) * digamma(beta.shape()) - 
    beta.rate().ln() + 
    beta.shape())
}
