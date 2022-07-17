use crate::error::{RegressionError::InvalidDistribution, RegressionError};
use serde::{Serialize, Deserialize};


///
/// Represents a Gamma distribution
/// 
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GammaDistribution {
    /// Shape parameter
    shape: f64,
    /// Rate (inverse scale) parameter
    rate: f64
}

impl GammaDistribution {

    ///
    /// Constructs a new distribution from the provided parameters
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
            Ok(GammaDistribution {shape, rate})
        }
    }

    ///
    /// Provides the shape parameter
    /// 
    #[inline]
    pub fn shape(&self) -> f64 {
        self.shape
    }

    ///
    /// Provides the rate parameter
    /// 
    #[inline]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    ///
    /// Computes the mean of the distribution
    /// 
    #[inline]
    pub fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    ///
    /// Computes the variance of the distribution
    /// 
    #[inline]
    pub fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }
}


///
/// Represents a Gaussian (Normal) distribution
/// 
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GaussianDistribution {
    /// Mean parameter
    mean: f64,
    /// Variance parameter
    variance: f64
}

impl GaussianDistribution {

    ///
    /// Constructs a new distribution from the provided parameters
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
            Ok(GaussianDistribution {mean, variance})
        }
    }

    ///
    /// Provides the mean parameter
    /// 
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    ///
    /// Provides the variance parameter
    /// 
    #[inline]
    pub fn variance(&self) -> f64 {
        self.variance
    }
}


///
/// Represents a Bernoulli distribution
/// 
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BernoulliDistribution {
    /// Positive probability
    p: f64
}

impl BernoulliDistribution {
    
    ///
    /// Constructs a new distribution from the provided parameter
    /// 
    /// # Arguments
    /// 
    /// `p` - The probability of 'success'
    /// 
    pub fn new(p: f64) -> Result<BernoulliDistribution, RegressionError> {
        if p >= 0.0 && p <= 1.0 {
            Ok(BernoulliDistribution {p})
        } else {
            Err(InvalidDistribution(format!("Invalid parameter 'p': {}", p)))
        }
    }

    ///
    /// Computes the mean of the distribution
    /// 
    #[inline]
    pub fn mean(&self) -> f64 {
        self.p
    }

    ///
    /// Computes the variance of the distribution
    /// 
    #[inline]
    pub fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }
}
