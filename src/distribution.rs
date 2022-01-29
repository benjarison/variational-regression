use crate::error::RegressionError;
use serde::{Serialize, Deserialize};


#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GammaDistribution {
    shape: f64,
    rate: f64
}

impl GammaDistribution {

    pub fn new(shape: f64, rate: f64) -> Result<GammaDistribution, RegressionError> {
        if shape <= 0.0 {
            Err(RegressionError::from("Shape parameter must be positive"))
        } else if rate <= 0.0 {
            Err(RegressionError::from("Rate parameter must be positive"))
        } else {
            Ok(GammaDistribution {shape, rate})
        }
    }

    #[inline]
    pub fn shape(&self) -> f64 {
        self.shape
    }

    #[inline]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        self.shape / self.rate
    }
}


#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GaussianDistribution {
    mean: f64,
    variance: f64
}

impl GaussianDistribution {

    pub fn new(mean: f64, variance: f64) -> Result<GaussianDistribution, RegressionError> {
        if variance <= 0.0 {
            Err(RegressionError::from("Variance must be positive"))
        }  else {
            Ok(GaussianDistribution {mean, variance})
        }
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    #[inline]
    pub fn variance(&self) -> f64 {
        self.variance
    }
}


