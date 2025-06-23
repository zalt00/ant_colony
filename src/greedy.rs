use core::f64;

use rand::RngCore;
use rand_xoshiro::Xoshiro256PlusPlus;
use rustworkx_core::petgraph::{self, graph::EdgeIndex};

use crate::graph::{Graph, Par, ACO};


pub fn greedy_algo(g: &Graph) -> (f64, Graph) {

    let t_edges = g.get_edges_tpl();

    let mut t = petgraph::graph::UnGraph::<u32, ()>::from_edges(t_edges.iter());

    'outer: loop {
        let edges = rustworkx_core::centrality::edge_betweenness_centrality(&t, true, 500000);
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

        return (t2.distorsion(&mut t2.get_dist_matrix(), &g.get_dist_matrix()), t2)
    }

}


impl ACO {

    pub fn greedy_stretch(&mut self) -> f64 {
        let ori = self.init_origin();
        let mut ens_sommets: Vec<usize> = vec![ori];


        for _march_advance in 0..(self.n-1) {
            let mut cur_min_stretch = f64::INFINITY;
            let mut cur_min_ei = 42;
            for ei in 0..self.possible_edges.len() {
                let edge = self.possible_edges[ei];
                let (new_vertex, parent) = if self.covered_vertices[edge[0]] {(edge[1], edge[0])} else {(edge[0], edge[1])};

                self.tree.ajout_sommet_arbre_recalculer_dist(&mut self.tree_dist_matrix, &mut ens_sommets, new_vertex, parent);
                self.covered_vertices[new_vertex] = true;
                //println!("{:?}", ens_sommets);
                let stretch = self.tree.stretch_moyen(&self.g, &mut self.tree_dist_matrix, &ens_sommets, &self.covered_vertices);
                //println!("{}", stretch);
                self.covered_vertices[new_vertex] = false;
                self.tree.annuler_recalcul(&mut self.tree_dist_matrix, &mut ens_sommets, new_vertex);

                if stretch < cur_min_stretch {
                    cur_min_stretch = stretch;
                    cur_min_ei = ei;
                }

            }
            let edge: [usize; 2] = self.possible_edges[cur_min_ei];

            let (new_vertex, parent) = if self.covered_vertices[edge[0]] {(edge[1], edge[0])} else {(edge[0], edge[1])};

            self.tree.ajout_sommet_arbre_recalculer_dist(&mut self.tree_dist_matrix, &mut ens_sommets, new_vertex, parent);
            self.covered_vertices[new_vertex] = true;
            self.update_possible_edges(cur_min_ei);
            self.tree.add_edge_unckecked(edge[0], edge[1]);

        }

        self.tree.distorsion(&mut self.tree_dist_matrix, &self.dist_matrix)

    }

    pub fn random_tree(&mut self, prng: &mut Xoshiro256PlusPlus) {
        let ori = self.init_origin();
        let mut ens_sommets: Vec<usize> = vec![ori];

        for _march_advance in 0..(self.n-1) {

            let ei = (prng.next_u64() % self.possible_edges.len() as u64) as usize;
            let edge = self.possible_edges[ei];
            let (new_vertex, parent) = if self.covered_vertices[edge[0]] {(edge[1], edge[0])} else {(edge[0], edge[1])};

            self.tree.ajout_sommet_arbre_recalculer_dist(&mut self.tree_dist_matrix, &mut ens_sommets, new_vertex, parent);
            //println!("{:?}", ens_sommets);


            
            self.update_possible_edges(ei);
            self.covered_vertices[new_vertex] = true;

            self.tree.add_edge_unckecked(edge[0], edge[1]);

        }
        //self.tree.distorsion(&mut self.tree_dist_matrix, &self.dist_matrix)

    }




}


// pub fn greedy_stretch(g: Graph) -> f64 {

//     let mut aco_dummy = ACO::new_dummy(g);
//     todo!()
//     aco_dummy.init_origin();

    // let ens_sommet = vec![];
    // let sommet_dedans = vec![false; aco_dummy];
    // aco_dummy.tree.ajout_sommet_arbre_recalculer_dist(aco_dummy.tr, ens_sommets, u, parent);
    // loop {
    //     for ei in aco_dummy.possible_edges() {



    //     }
    // }




    // let t_edges = g.get_edges_tpl();

    // let mut t = petgraph::graph::UnGraph::<u32, ()>::from_edges(t_edges.iter());

    // 'outer: loop {
    //     let edges = rustworkx_core::centrality::edge_betweenness_centrality(&t, true, 50);
    //     let indices: Vec<EdgeIndex> = t.edge_indices().collect();
    //     let mut emaped: Vec<(f64, EdgeIndex)> = edges.iter().enumerate().map(|(i, v)|
    //         {(v.unwrap(), indices[i])}).collect();
    //     emaped.sort_by(|a, b| a.partial_cmp(b).unwrap());

    //     for i in 0..emaped.len() {
    //         let (_, ei) = emaped[i];
    //         let endpoints = t.edge_endpoints(ei).unwrap();
    //         t.remove_edge(ei);
    //         if petgraph::algo::connected_components(&t) > 1 {
    //             t.add_edge(endpoints.0, endpoints.1, ());
    //         } else {
    //             continue 'outer;
    //         }
    //     }

    //     let mut t2 = Graph::from_petgraph(&t);
    //     assert_eq!(t2.n -1, t.edge_count());

    //     return t2.distorsion(&mut t2.get_dist_matrix(), &g.get_dist_matrix());
    // }

// }






