use crate::error::{RegressionError::InvalidDistribution, RegressionError};
use serde::{Serialize, Deserialize};

pub trait ScalarDistribution {

    ///
    /// Computes the mean of the distribution
    /// 
    fn mean(&self) -> f64;

    ///
    /// Computes the variance of the distribution
    /// 
    fn variance(&self) -> f64;

    ///
    /// Computes the standard deviation of the distribution
    /// 
    fn std_dev(&self) -> f64;
}

///
/// Represents a Gamma distribution
/// 
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GammaDistribution {
    /// Shape parameter
    pub shape: f64,
    /// Rate (inverse scale) parameter
    pub rate: f64
}

impl GammaDistribution {

    ///
    /// Constructs a new Gamma distribution from the provided parameters
    /// 
    /// # Arguments
    /// 
    /// `shape` - The shape parameter
    /// `rate` - The rate parameter
    /// 
    pub fn new(shape: f64, rate: f64) -> Result<GammaDistribution, RegressionError> {
        if shape <= 0.0 {
            Err(InvalidDistribution(format!("Shape parameter must be positive (found {})", shape)))
        } else if rate <= 0.0 {
            Err(InvalidDistribution(format!("Rate parameter must be positive (found {})", rate)))
        } else {
            Ok(GammaDistribution { shape, rate })
        }
    }
}

impl ScalarDistribution for GammaDistribution {

    #[inline]
    fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    #[inline]
    fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }

    #[inline]
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

///
/// Represents a Gaussian (Normal) distribution
/// 
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GaussianDistribution {
    /// Mean parameter
    pub mean: f64,
    /// Variance parameter
    pub variance: f64
}

impl GaussianDistribution {

    ///
    /// Constructs a new Gaussian distribution from the provided parameters
    /// 
    /// # Arguments
    /// 
    /// `mean` - The mean parameter
    /// `variance` - The variance parameter
    /// 
    pub fn new(mean: f64, variance: f64) -> Result<GaussianDistribution, RegressionError> {
        if variance <= 0.0 {
            Err(InvalidDistribution(format!("Variance must be positive (found {})", variance)))
        }  else {
            Ok(GaussianDistribution { mean, variance })
        }
    }

}

impl ScalarDistribution for GaussianDistribution {

    #[inline]
    fn mean(&self) -> f64 {
        self.mean
    }

    #[inline]
    fn variance(&self) -> f64 {
        self.variance
    }

    #[inline]
    fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }
}

///
/// Represents a Bernoulli distribution
/// 
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BernoulliDistribution {
    /// Probability of positive event
    pub p: f64
}

impl BernoulliDistribution {
    
    ///
    /// Constructs a new Bernoulli distribution from the provided parameter
    /// 
    /// # Arguments
    /// 
    /// `p` - The probability of 'success'
    /// 
    pub fn new(p: f64) -> Result<BernoulliDistribution, RegressionError> {
        if p >= 0.0 && p <= 1.0 {
            Ok(BernoulliDistribution { p })
        } else {
            Err(InvalidDistribution(format!("Invalid parameter 'p': {}", p)))
        }
    }
}

impl ScalarDistribution for BernoulliDistribution {

    #[inline]
    fn mean(&self) -> f64 {
        self.p
    }

    #[inline]
    fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }

    #[inline]
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}
