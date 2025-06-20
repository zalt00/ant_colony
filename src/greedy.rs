use rustworkx_core::petgraph::{self, graph::EdgeIndex};

use crate::graph::Graph;


pub fn greedy_algo(g: &Graph) -> f64 {

    let t_edges = g.get_edges_tpl();

    let mut t = petgraph::graph::UnGraph::<u32, ()>::from_edges(t_edges.iter());

    'outer: loop {
        let edges = rustworkx_core::centrality::edge_betweenness_centrality(&t, true, 50);
        let indices: Vec<EdgeIndex> = t.edge_indices().collect();
        let mut emaped: Vec<(f64, EdgeIndex)> = edges.iter().enumerate().map(|(i, v)|
            {(v.unwrap(), indices[i])}).collect();
        emaped.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for i in 0..emaped.len() {
            let (_, ei) = emaped[i];
            let endpoints = t.edge_endpoints(ei).unwrap();
            t.remove_edge(ei);
            if petgraph::algo::connected_components(&t) > 1 {
                t.add_edge(endpoints.0, endpoints.1, ());
            } else {
                continue 'outer;
            }
        }

        let mut t2 = Graph::from_petgraph(&t);
        assert_eq!(t2.n -1, t.edge_count());

        return t2.distorsion(&mut t2.get_dist_matrix(), &g.get_dist_matrix());
    }



}



