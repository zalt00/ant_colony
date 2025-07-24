use std::collections::VecDeque;



use crate::graph::{graph_core::GraphCore, graph_generator::GraphRng, MatGraph, RootedTree};

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

fn greedy_bfs_from_node<T: GraphCore>(g: &T, ustart: usize, queue: &mut VecDeque<(usize, usize)>) -> RootedTree {
    let n = g.vertex_count();
    let mut tree = RootedTree::new(n, ustart);
    let mut visited = vec![false; n];
    visited[ustart] = true;
    
    for &u in g.get_neighbors(ustart) {
        queue.push_back((u, ustart));
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
    tree

}

pub fn greedy_bfs<T: GraphCore>(g: &T) -> RootedTree {
    let n = g.vertex_count();
    let max_degree_node = (0..n).max_by_key(|u| {g.get_neighboor_count_unchecked(*u)}).unwrap();
    let mut queue = VecDeque::new();
    greedy_bfs_from_node(g, max_degree_node, &mut queue)

}

pub fn random_greedy_bfs<T: GraphCore>(g: &T, rd_val: u64) -> RootedTree {
    let n = g.vertex_count();
    let max_degree_node = (rd_val % n as u64) as usize;
    let mut queue = VecDeque::new();
    greedy_bfs_from_node(g, max_degree_node, &mut queue)

}


pub fn multiple_greedy_bfs<T: GraphCore>(g: &T, mut k: usize) -> (u64, RootedTree) {
    let n = g.vertex_count();

    k = k.min(n);

    let mut node_order: Vec<usize> = (0..n).collect();
    node_order.sort_by_key(|u | {g.get_neighboor_count_unchecked(*u) as isize});

    let mut best_tree = RootedTree::new(n, 0);
    let mut best_disto = u64::MAX;
    let mut queue = VecDeque::new();
    for &max_degree_node in node_order[node_order.len() - k..].iter() {
        let mut tree = greedy_bfs_from_node(g, max_degree_node, &mut queue);

        let disto = tree.distance_sum();
        if disto < best_disto {
            best_disto = disto;
            best_tree = tree;
        }
    }
    //println!("greedy bfs end");

    (best_disto, best_tree)

}

pub fn starting_node_test_greedy_bfs<T: GraphCore>(g: &T) {
    let n = g.vertex_count();
    let dm = g.get_dist_matrix();
    //let max_degree_node = (0..n).max_by_key(|u| {g.get_neighboor_count_unchecked(*u)}).unwrap();
    let mut node_order: Vec<usize> = (0..n).collect();
    node_order.sort_by_key(|u | {g.get_neighboor_count_unchecked(*u)});
    for max_degree_node in node_order {
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

        println!("disto: {}  {}", tree.distorsion(g, &dm), g.get_neighboor_count_unchecked(max_degree_node))
    }

}






