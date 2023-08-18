use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn compress(mut xs: Vec<i64>) -> PyResult<Vec<i64>> {
    Ok(xs)
}

/// A Python module implemented in Rust.
#[pymodule]
fn compressor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    Ok(())
}