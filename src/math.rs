use nalgebra::DMatrix;

type DenseMatrix = DMatrix<f64>;

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

///
/// Efficiently computes trace(a * b) by only forming
/// the diagonal ridge of (a * b)
/// 
/// # Arguments
/// 
/// `a` - left hand side matrix
/// `b` - right hand side matrix
/// 
pub fn trace_of_product(a: &DenseMatrix, b: &DenseMatrix) -> f64 {
    (0..a.nrows())
    .fold(0.0, |sum, i| {
        sum + (a.row(i) * b.column(i))[(0, 0)]
    })
}
