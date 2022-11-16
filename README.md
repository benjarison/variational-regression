# variational-regression

Regression models trained using variational inference

## Motivation

This library provides implementations of Bayesian regression models that learn parameter values by optimizing a variational lower bound on the log likelihood. The benefit of using variational inference is that training is efficient (no sampling required), and we have a well defined objective function to optimize.

## Linear Regression

The model is specified as follows:

p(<b>y</b> | <b>&theta;</b>) = &prod; N(y<sub>i</sub> | <b>&theta;</b><sup>T</sup><b>x</b><sub>i</sub>, &beta;<sup>-1</sup>)

p(<b>&theta;</b> | <b>&alpha;</b>) = &prod; N(&theta;<sub>j</sub> | 0, &alpha;<sub>j</sub><sup>-1</sup>)

p(<b>&alpha;</b>) = &prod; Gam(&alpha;<sub>j</sub> | a<sub>0</sub>, b<sub>0</sub>)

p(&beta;) = Gam(&beta; | c<sub>0</sub>, d<sub>0</sub>)

with the given notation:

<b>y</b> = labels

<b>x</b> = features

<b>&theta;</b> = model weights

<b>&alpha;</b> = weight precision (inverse variance)

&beta; = noise precision (inverse variance)

## Logistic Regression

The model is specified as follows:

p(<b>y</b> | <b>&theta;</b>) = &prod; &sigma;(<b>&theta;</b><sup>T</sup><b>x</b><sub>i</sub>)<sup>y<sub>i</sub></sup> {1 - &sigma;(<b>&theta;</b><sup>T</sup><b>x</b><sub>i</sub>)}<sup>1-y<sub>i</sub></sup>

p(<b>&theta;</b> | <b>&alpha;</b>) = &prod; N(&theta;<sub>j</sub> | 0, &alpha;<sub>j</sub><sup>-1</sup>)

p(<b>&alpha;</b>) = &prod; Gam(&alpha;<sub>j</sub> | a<sub>0</sub>, b<sub>0</sub>)

with the given notation:

<b>y</b> = labels

<b>x</b> = features

<b>&theta;</b> = model weights

<b>&alpha;</b> = weight precision (inverse variance)

&sigma; = logistic sigmoid function

## Reference

The models implemented here are heavily based on those presented in Chapter 10 of "Pattern Recognition and Machine Learning" (Bishop, 2006). However a key difference is that here, each model weight has its own distribution for precision, which facilitates the construction of models with greater sparsity.

## Examples

```rust
use variational_regression::*;

fn main() -> Result<(), RegressionError> {

    // construct features
    let features = vec![
        vec![-0.2, -0.9, -0.5, 0.3],
        vec![0.6, 0.3, 0.3, -0.4],
        vec![0.9, -0.4, -0.5, -0.6],
        vec![-0.7, 0.8, 0.3, -0.3],
        vec![-0.5, -0.7, -0.1, 0.8],
        vec![0.5, 0.5, 0.0, 0.1],
        vec![0.1, -0.0, 0.0, -0.2],
        vec![0.4, 0.0, 0.2, 0.0],
        vec![-0.2, 0.9, -0.1, -0.9],
        vec![0.1, 0.4, -0.5, 0.9],
    ];
    
    // construct labels
    let labels = vec![-0.4, 0.1, -0.8, 0.5, 0.6, -0.2, 0.0, 0.7, -0.3, 0.2];
    
    // configure and train model
    let config = LinearConfig::default();
    let model = VariationalLinearRegression::train(&features, &labels, &config)?;
    
    // inspect model weights
    for (ind, weight) in model.weights().iter().enumerate() {
        println!("Weight {}: {}", ind + 1, weight);
    }

    // inspect noise variance
    println!("Noise Variance: {}", 1.0 / model.noise_precision.mean());
    
    // get predictive distribution
    let prediction = model.predict(&vec![0.1, -0.5, 0.3, 0.9])?;
    println!("Predictive mean: {}", prediction.mean());
    
    Ok(())
}
```

```rust
use variational_regression::*;

fn main() -> Result<(), RegressionError> {

    // construct features
    let features = vec![
        vec![-0.2, -0.9, -0.5, 0.3],
        vec![0.6, 0.3, 0.3, -0.4],
        vec![0.9, -0.4, -0.5, -0.6],
        vec![-0.7, 0.8, 0.3, -0.3],
        vec![-0.5, -0.7, -0.1, 0.8],
        vec![0.5, 0.5, 0.0, 0.1],
        vec![0.1, -0.0, 0.0, -0.2],
        vec![0.4, 0.0, 0.2, 0.0],
        vec![-0.2, 0.9, -0.1, -0.9],
        vec![0.1, 0.4, -0.5, 0.9],
    ];
    
    // construct labels
    let labels = vec![true, false, true, false, true, false, true, false, true, false];
    
    // configure and train model
    let config = LogisticConfig::default();
    let model = VariationalLogisticRegression::train(&features, &labels, &config)?;
    
    // inspect model weights
    for (ind, weight) in model.weights().iter().enumerate() {
        println!("Weight {}: {}", ind + 1, weight);
    }

    // get predictive distribution
    let prediction = model.predict(&vec![0.1, -0.5, 0.3, 0.9])?;
    println!("Predictive mean: {}", prediction.mean());
    
    Ok(())
}
```