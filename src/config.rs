use crate::distribution::GammaDistribution;

///
/// Specifies configurable hyperparameters for training
/// 
pub struct TrainConfig {
    /// Prior distribution over the precision of the model weights
    pub weight_precision_prior: GammaDistribution,
    /// Prior distribution over the precision of the noise term
    pub noise_precision_prior: GammaDistribution,
    /// Maximum number of training iterations
    pub max_iter: usize,
    /// Convergence criteria threshold
    pub tolerance: f64,
    /// Indicates whether or not to print training info
    pub verbose: bool
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            weight_precision_prior: GammaDistribution::new(1e-4, 1e-4).unwrap(),
            noise_precision_prior: GammaDistribution::new(1e-4, 1e-4).unwrap(),
            max_iter: 1000, 
            tolerance: 1e-4,
            verbose: true
        }
    }
}
