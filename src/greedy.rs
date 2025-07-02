use core::f64;


use crate::{graph::MatGraph, graph_core::GraphCore, graph_generator::GraphRng};

#[derive(PartialEq, PartialOrd)]
struct ComparableFloat(f64);


impl Eq for ComparableFloat {}

impl Ord for ComparableFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

pub fn greedy_ebc_delete_no_recompute(g: &MatGraph, ebc: &Vec<f64>, dm: &Vec<u32>) -> (f64, MatGraph) {
    let mut tree = g.clone();
    let mut edges = g.get_edges();
    edges.sort_by_key(|&[u, v]| {ComparableFloat(ebc[u + g.n * v])});

    for &edge in edges.iter() {
        if tree.is_connected_without(edge) {
            tree.remove_edge_slow(edge);
        }
    }

    (tree.distorsion(&mut vec![u32::MAX; g.n*g.n], dm), tree)

}

