use pyo3::prelude::*;

pub mod graph;
pub mod my_rand;

pub mod solver;
pub mod utils;
pub mod config;
pub mod neighborhood;
pub mod trace;
pub mod distorsion_heuristics;
pub mod counters;
use rand::RngCore;
use rand::SeedableRng;

use crate::distorsion_heuristics::constants;
use crate::graph::compressed_graph::CompressedGraph;
use crate::graph::graph_core::GraphCore;
use crate::graph::graph_generator::GraphRng;
use crate::graph::MatGraph;
use crate::graph::RootedTree;
use crate::my_rand::Prng;
use crate::neighborhood::NSVal;
use crate::neighborhood::NeighborhoodStrategies;
use crate::solver::vns::VNS;
use crate::solver::MultiBFSTree;
use crate::solver::Solver;
use crate::solver::VNSWithStart;
use crate::utils::TarjanSolver;


#[pyclass]
struct RustVectorF64(Vec<f64>);

#[pymethods]
impl RustVectorF64 {
    fn to_list(&self) -> Vec<f64> {self.0.clone()}
    #[staticmethod]
    fn from_list(l: Vec<f64>) -> Self {RustVectorF64(l)}
}

#[pyclass]
struct RustVectorU32(Vec<u32>);

#[pymethods]
impl RustVectorU32 {
    fn to_list(&self) -> Vec<u32> {self.0.clone()}
    #[staticmethod]
    fn from_list(l: Vec<u32>) -> Self {RustVectorU32(l)}

}


#[pymethods]
impl CompressedGraph {
    #[staticmethod]
    fn from_edges_py(edges: Vec<[usize; 2]>) -> CompressedGraph {
        Self::from_edges_only(&edges)
    }

    fn vertex_count_py(&self) -> usize {
        self.vertex_count()
    }

    fn get_dist_matrix_py(&self) -> RustVectorU32 {
        RustVectorU32(self.get_dist_matrix())
    }

    fn distorsion_py(&self, parent_dist_matrix: &RustVectorU32) -> f64 {
        self.distorsion(&mut vec![u32::MAX; self.n*self.n], &parent_dist_matrix.0)
    }

    fn wiener_slow_py(&self) -> u64 {
        self.wiener(&mut vec![u32::MAX; self.n*self.n])
    }

    fn get_edge_betweeness_centrality_py(&self) -> RustVectorF64 {
        RustVectorF64(self.get_edge_betweeness_centrality())
    }

    fn bfs_furthest_vertex_py(&self, u: usize) -> (usize, Vec<usize>) {
        let mut path = vec![];
        let v = self.bfs_further_vertex(u, true, &mut path);
        (v, path)
    }

    fn bfs_connected_components_py(&self) -> (i32, Vec<i32>) {
        self.bfs_connected_components()
    }

    fn select_largest_connected_component_py(&self) -> Self {
        self.clone().connectify()
    }

    fn random_subtree_py(&self, prng: &mut RustPrng) -> RootedTree {
        self.random_subtree(&mut prng.0)
    }

    #[staticmethod]
    fn random_graph_py(n: usize, m_added: usize, prng: &mut RustPrng) -> CompressedGraph {
        Self::random_graph(n, m_added, &mut prng.0)
    }    

    fn write_dot_py(&self, fname: &str) {
        self.to_dot(fname);
    }

    fn renumber_py(&self, permutation: Vec<usize>) -> CompressedGraph {
        self.renumber(&permutation)
    }

    fn get_edges_py(&self) -> Vec<[usize; 2]> {
        self.get_edges()
    }
}

#[pymethods]
impl RootedTree {
    #[staticmethod]
    fn from_graph_py(g: &CompressedGraph, root: usize) -> RootedTree {
        RootedTree::from_graph(g, root)
    }

    fn to_graph_py(&self, g: &CompressedGraph) -> CompressedGraph {
        self.to_graph(g)
    }

    fn wiener_fast_py(&mut self) -> u64 {
        self.distance_sum()
    }

    fn stretch_py(&mut self, g: &CompressedGraph) -> f64 {
        let mut ts = TarjanSolver::new(g.n, g);
        self.stretch(g, &mut ts)
    }

    fn stretch_ebc_py(&mut self, g: &CompressedGraph, ebc: &RustVectorF64) -> f64 {
        let mut ts = TarjanSolver::new(g.n, g);
        self.stretch_ebc(g, &mut ts, &ebc.0)
    }

    fn distorsion_py(&mut self, g: &CompressedGraph, dm: &RustVectorU32) -> f64 {
        self.distorsion(g, &dm.0)
    }
}


#[pyclass]
#[derive(Clone)]
struct RustPrng(Prng);

#[pymethods]
impl RustPrng {
    fn clone_py(&self) -> Self {
        self.clone()
    }

    #[staticmethod]
    fn from_seed_u64_py(seed: u64) -> Self {
        RustPrng(Prng::seed_from_u64(seed))
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
}

#[derive(Debug, Clone, Copy)]
#[pyclass]
pub enum PyNeighborhoodStrategies {
    EdgeSwap(),
    EdgeSubtreeRelocation(),
    CriticalPathSubtreeRelocation(),
    CriticalPathSubtreeVNS(),
    SpiderSubtreeBFS((usize, usize)),
    SpiderSubtreeSwap((usize, usize)),
    SubtreeSubtreeBFS((usize, usize)),
    SubtreeSubtreeSwap((usize, usize))
}

impl PyNeighborhoodStrategies {
    fn to_normal(self) -> NeighborhoodStrategies {
        match self {
            PyNeighborhoodStrategies::CriticalPathSubtreeRelocation() => NeighborhoodStrategies::CriticalPathSubtreeRelocation,
            PyNeighborhoodStrategies::EdgeSubtreeRelocation() => NeighborhoodStrategies::EdgeSubtreeRelocation,
            PyNeighborhoodStrategies::CriticalPathSubtreeVNS() => NeighborhoodStrategies::CriticalPathSubtreeVNS,
            PyNeighborhoodStrategies::EdgeSwap() => NeighborhoodStrategies::EdgeSwap,
            PyNeighborhoodStrategies::SpiderSubtreeSwap((a, b)) => NeighborhoodStrategies::SpiderSubtreeSwap(NSVal::N(a, b)),
            PyNeighborhoodStrategies::SpiderSubtreeBFS((a, b)) => NeighborhoodStrategies::SpiderSubtreeVNS(NSVal::N(a, b)),
            PyNeighborhoodStrategies::SubtreeSubtreeSwap((a, b)) => NeighborhoodStrategies::SpiderSubtreeSwap(NSVal::N(a, b)),
            PyNeighborhoodStrategies::SubtreeSubtreeBFS((a, b)) => NeighborhoodStrategies::SubtreeSubtreeVNS(NSVal::N(a, b))
        }
    }
}



#[pyclass]
struct PySolver;

#[pymethods]
impl PySolver {
    #[staticmethod]
    fn vns_all_default_py(g: CompressedGraph, seed: u64, time_limit: f64) -> RootedTree {
       VNSWithStart::<CompressedGraph, MultiBFSTree, 2, 0>::auto_solve_no_ebcdm(g, seed, time_limit)
    }

    #[staticmethod]
    fn vns_custom_mode(g: CompressedGraph, seed: u64, time_limit: f64, 
        mode: Vec<PyNeighborhoodStrategies>, sample_sizes: Vec<usize>) -> RootedTree {

        let (cc, _ccv) = g.bfs_connected_components();
        assert_eq!(cc, 1);

        let mut vns: VNS<CompressedGraph> = VNS::new_custom_mode(g.clone(), seed, vec![], vec![], mode.iter().map(|v| {v.to_normal()}).collect(), sample_sizes);
        vns.recompute_distorsion = false;
        //vns.verbose = cfg!(feature="verbose");
        let base_tree = MultiBFSTree::<50>::auto_solve_no_ebcdm(g.clone(), seed + 15, time_limit);

        vns.gvns2(base_tree, 100000, time_limit)
    }

    #[staticmethod]
    fn vns_custom_mode_custom_start(g: CompressedGraph, seed: u64, time_limit: f64, 
        mode: Vec<PyNeighborhoodStrategies>, sample_sizes: Vec<usize>, start: RootedTree) -> RootedTree {
            
        let (cc, _ccv) = g.bfs_connected_components();
        assert_eq!(cc, 1);

        let mut vns: VNS<CompressedGraph> = VNS::new_custom_mode(g.clone(), seed, vec![], vec![], mode.iter().map(|v| {v.to_normal()}).collect(), sample_sizes);
        vns.recompute_distorsion = false;
        //vns.verbose = cfg!(feature="verbose");

        vns.gvns2(start, 100000, time_limit)
        }
}



/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rust_vnsdisto(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompressedGraph>()?;
    m.add_class::<RootedTree>()?;
    m.add_class::<RustVectorF64>()?;
    m.add_class::<RustVectorU32>()?;
    m.add_class::<RustPrng>()?;
    m.add_class::<PySolver>()?;
    m.add_class::<PyNeighborhoodStrategies>()?;

    Ok(())
}