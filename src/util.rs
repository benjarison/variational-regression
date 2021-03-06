use nalgebra::{DVector, DMatrix};

type DenseVector = DVector<f64>;
type DenseMatrix = DMatrix<f64>;

pub fn design_matrix(features: &Vec<Vec<f64>>, bias: bool) -> DenseMatrix {
    let offset = if bias { 1 } else { 0 };
    let n = features.len();
    let d = features[0].len() + offset;
    let mut x = DenseMatrix::zeros(n, d);
    for i in 0..n {
        if bias {
            x[(i, 0)] = 1.0;
        }
        for j in offset..d {
            x[(i, j)] = features[i][j - offset];
        }
    }
    return x;
}

pub fn design_vector(features: &Vec<f64>, bias: bool) -> DenseVector {
    let offset = if bias { 1 } else { 0 };
    let d = features.len() + offset;
    let mut x = DenseVector::zeros(d);
    if bias {
        x[0] = 1.0;
    }
    for j in offset..d {
        x[j] = features[j - offset];
    }
    return x;
}

pub fn trace_of_product(a: &DenseMatrix, b: &DenseMatrix) -> f64 {
    let mut trace = 0.0;
    for i in 0..a.nrows() {
        trace += (a.row(i) * b.column(i))[(0, 0)];
    }
    return trace;
}
