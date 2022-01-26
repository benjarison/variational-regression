use nalgebra::{DVector, Cholesky};
use nalgebra::DMatrix;
use nalgebra::SquareMatrix;
use statrs::distribution::Gamma;
use std::error::Error;
use statrs::statistics::Distribution;
use crate::error::RegressionError;
use statrs::function::gamma::{digamma, ln_gamma};
use std::f64::consts::PI;
use crate::math::LN_TWO_PI;

pub type DenseVector = DVector<f64>;
pub type DenseMatrix = DMatrix<f64>;

pub struct VariationalLinearRegression {
    pub intercept: f64,
    pub weights: DenseVector,
    pub covariance: DenseMatrix,
    pub noise_precision: Gamma
}

impl VariationalLinearRegression {

    pub fn train(
        features: DenseMatrix,
        labels: DenseVector,
        weight_prior: Gamma,
        noise_prior: Gamma,
        tolerance: f64,
        max_iter: usize
    ) -> Result<VariationalLinearRegression, RegressionError> {

        let x = features.insert_column(0, 1.0);
        let y = labels;
        let xtx = x.tr_mul(&x);
        let xty = x.tr_mul(&y);
        let yty = y.dot(&y);
        let mut alpha = vec![weight_prior; x.ncols()];
        let mut beta = noise_prior;

        let mut iter = 0;
        for iter in 0..max_iter {
            let (theta, s) = q_theta(&xtx, &xty, &alpha, &beta)?;
            alpha = q_alpha(&s, &theta, &weight_prior)?;
            beta = q_beta(&xtx, &xty, yty, &theta, &s, &noise_prior)?;

        }


        Err(RegressionError::from(""))
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

fn expect_log_p_y(
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

fn expect_log_p_theta(
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

fn expect_log_p_alpha(
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