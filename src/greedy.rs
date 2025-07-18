use core::f64;
use std::collections::VecDeque;


use crate::{distorsion_heuristics::Num, graph::{graph_core::GraphCore, graph_generator::GraphRng, MatGraph, RootedTree}};

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

pub fn greedy_bfs<T: GraphCore>(g: &T) -> (u64, RootedTree) {
    let n = g.vertex_count();
    let max_degree_node = (0..n).max_by_key(|u| {g.get_neighboor_count_unchecked(*u)}).unwrap();

    let mut tree = RootedTree::new(n, max_degree_node);
    let mut visited = vec![false; n];
    visited[max_degree_node] = true;
    let mut queue = VecDeque::new();
    
    for &u in g.get_neighbors(max_degree_node) {
        queue.push_back((u, max_degree_node));
        visited[u] = true;
    }

    while !queue.is_empty() {
        let (u, parent) = queue.pop_front().unwrap();
        tree.add_child(parent, u);

        for &v in g.get_neighbors(u) {
            if !visited[v] {
                visited[v] = true;
                queue.push_back((v, u));
            }
        }
    }

    tree.update_leaves();
    (tree.new_disto_approx4(), tree)

}

