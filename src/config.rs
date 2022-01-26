use statrs::distribution::Gamma;

pub struct TrainConfig {
    pub weight_prior: Gamma,
    pub noise_prior: Gamma,
    pub max_iter: usize,
    pub tolerance: f64
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            weight_prior: Gamma::new(1e-4, 1e-4).unwrap(),
            noise_prior: Gamma::new(1.01, 1e-4).unwrap(),
            max_iter: 1000, 
            tolerance: 1e-4
        }
    }
}
