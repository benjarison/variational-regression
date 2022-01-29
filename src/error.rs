use std::error::Error;
use std::fmt::{Display, Formatter, Debug};
use std::convert::From;

#[derive(Debug)]
pub struct RegressionError {
    pub message: String
}

impl Display for RegressionError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for RegressionError {}

impl From<&str> for RegressionError {
    fn from(msg: &str) -> RegressionError {
        RegressionError {
            message: String::from(msg)
        }
    }
}

impl From<String> for RegressionError {
    fn from(msg: String) -> RegressionError {
        RegressionError {message: msg}
    }
}
