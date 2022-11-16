use crate::error::{RegressionError::InvalidDistribution, RegressionError};
use serde::{Serialize, Deserialize};

///
/// Represents a generic scalar distribution
/// 
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
    /// `p` - The probability of a positive event
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

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_gamma() {
        let a = GammaDistribution::new(1.0, 2.0).unwrap();
        assert!(a.shape == 1.0);
        assert!(a.rate == 2.0);
        assert_approx_eq!(a.mean(), 0.5);
        assert_approx_eq!(a.variance(), 0.25);
        assert_approx_eq!(a.std_dev(), 0.5);

        assert!(GammaDistribution::new(0.0, 1.0).is_err());
    }

    #[test]
    fn test_gaussian() {
        let a = GaussianDistribution::new(1.0, 4.0).unwrap();
        assert!(a.mean == 1.0);
        assert!(a.variance == 4.0);
        assert_approx_eq!(a.mean(), 1.0);
        assert_approx_eq!(a.variance(), 4.0);
        assert_approx_eq!(a.std_dev(), 2.0);

        assert!(GaussianDistribution::new(0.0, 0.0).is_err());
    }

    #[test]
    fn test_bernoulli() {
        let a = BernoulliDistribution::new(0.4).unwrap();
        assert!(a.p == 0.4);
        assert_approx_eq!(a.mean(), 0.4);
        assert_approx_eq!(a.variance(), 0.24);
        assert_approx_eq!(a.std_dev(), 0.24f64.sqrt());

        assert!(BernoulliDistribution::new(2.0).is_err());
    }
}
