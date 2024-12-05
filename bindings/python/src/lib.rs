mod kmeans;
use kmeans::Kmeans;

use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _toymlrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    // Create the clustering submodule
    let clustering_module = PyModule::new(m.py(), "clustering")?;
    let _ = clustering_module.add_class::<Kmeans>();
    m.add_submodule(&clustering_module)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("toymlrs.clustering", clustering_module)?;

    Ok(())
}
