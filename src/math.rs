use nalgebra::{DMatrix, DVector};

type DenseVector = DVector<f64>;
type DenseMatrix = DMatrix<f64>;

// Cache value of ln(2PI)
pub const LN_2PI: f64 = 1.8378770664093453;

// Logistic sigmoid function
#[inline]
pub fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Efficiently computes trace(a * b) by only forming
// the diagonal ridge of (a * b)
pub fn trace_of_product(a: &DenseMatrix, b: &DenseMatrix) -> f64 {
    (0..a.nrows())
    .fold(0.0, |sum, i| {
        sum + (a.row(i) * b.column(i))[(0, 0)]
    })
}

// Scales each row of the matrix by the corresponding vector element
pub fn scale_rows(matrix: &DenseMatrix, vector: &DenseVector) -> DenseMatrix {
    let mut scaled = matrix.clone();
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            scaled[(i, j)] *= vector[i];
        }
    }
    return scaled;
}
