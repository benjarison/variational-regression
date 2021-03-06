///
/// Cache value of ln(2PI)
/// 
pub const LN_2PI: f64 = 1.8378770664093453;

///
/// Logistic sigmoid function
/// 
/// # Arguments
/// 
/// `x` - the logit value
/// 
#[inline]
pub fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
