use crate::{graph::{graph_core::GraphCore, graph_generator::{GraphData, GraphRng}, RootedTree}, vns::VNS};


pub trait Solver {
    fn auto_parameters_solve(gdt: &GraphData, seed: u64, time_limit: f64) -> RootedTree;
}

impl<T: GraphCore+GraphRng> Solver for VNS<T> {
    fn auto_parameters_solve(gdt: &GraphData, seed: u64, time_limit: f64) -> RootedTree {
        let (g, ebc, dm) = gdt.graph_ebc_dist_matrix::<T>();
        assert!(g.is_connected());
        let mut vns: VNS<T> = VNS::new(g, seed, ebc, dm, 2);
        vns.gvns_random_start_timeout_no_distorsion(time_limit).0
    }
}


