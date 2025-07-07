use pyo3::prelude::*;

pub mod graph;
pub mod my_rand;
pub mod greedy;
pub mod aco2;
pub mod utils;
pub mod config;
pub mod neighborhood;
pub mod annealing;
pub mod trace;
pub mod distorsion_heuristics;
pub mod counters;
pub mod compressed_graph;
pub mod vns;
use pyo3::wrap_pyfunction;                // <-- bien importer la macro

#[pyfunction]
fn hello() -> PyResult<()> {
    println!("hello");
    Ok(())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn mylib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}