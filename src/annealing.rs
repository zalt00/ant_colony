use core::f64;
use std::time::Instant;

use rand::{RngCore, SeedableRng};

use crate::utils::TarjanSolver;
use crate::trace::TraceData;
use crate::neighborhood::NeighborhoodStrategies;
use crate::my_rand::{my_rand, Prng};
use crate::graph_generator::GraphRng;
use crate::graph_core::GraphCore;
use crate::graph::RootedTree;
use crate::distorsion_heuristics::constants;






pub struct SA<T: GraphCore+GraphRng> {
    n: usize,
    g: T,
    tree_buf: T,
    tarjan_solver: TarjanSolver,
    edges: Vec<[usize; 2]>,
    prng: Prng,
    edge_betweeness_centrality: Vec<f64>,

    k: usize, // current neighborhood
    neighborhood_strategies: &'static [NeighborhoodStrategies],

    temperature: f64,
    _coef: f64,


    dist_matrix: Vec<u32>
}   

impl<T: GraphCore+GraphRng> SA<T> {
    pub fn new(g: T, seed_u64: u64, edge_betweeness_centrality: Vec<f64>, dist_matrix: Vec<u32>) -> SA<T> {
        use NeighborhoodStrategies::*;
        static NEIGHBORHOOD_STRATEGIES: [NeighborhoodStrategies; 3] = [CriticalPathSubtreeRelocation, EdgeSubtreeRelocation, EdgeSwap];

        let prng = Prng::seed_from_u64(seed_u64);
        let edges = g.get_edges();
        let n = g.vertex_count();
        let tarjan_solver = TarjanSolver::new(n, &g);
        let tree_buf = g.clone_empty();
        SA { n, g, tree_buf,
            tarjan_solver, edges, prng, edge_betweeness_centrality,
            k: 0, neighborhood_strategies: &NEIGHBORHOOD_STRATEGIES, temperature: 1.0, _coef: 1.,
            dist_matrix }


    }

    pub fn init_strategy(&mut self, x: &mut RootedTree, i: usize) {
        use NeighborhoodStrategies::*;
        match self.neighborhood_strategies[i] {
            EdgeSwap => {
                x.update_parents();  // todo cette chose n'a pas a exister
            },
            _ => ()
        }
    }

    pub fn get_neighbor(&mut self, x: &mut RootedTree, i: usize) -> RootedTree {
        // /!\ appeler init_strategy avant

        use NeighborhoodStrategies::*;
        match self.neighborhood_strategies[i] {
            EdgeSwap => {
                let mut y = x.clone();
                //self.tarjan_solver.launch(&y, &self.g);
                while !y.edge_swap_random(&mut self.prng, &self.edges) {};
                y
            },
            EdgeSubtreeRelocation => {
                while !x.subtree_swap_with_random_edge(&mut self.prng, &self.edges, &self.g, &mut self.tree_buf) {};
                let root = (self.prng.next_u64() % self.n as u64) as usize;
                RootedTree::from_graph(&self.tree_buf, root)
            },
            CriticalPathSubtreeRelocation => {
                x.subtree_swap_with_random_critical_path(&mut self.prng, &self.g, &mut self.tree_buf);
                let root = (self.prng.next_u64() % self.n as u64) as usize;
                RootedTree::from_graph(&self.tree_buf, root)
            }
        }
    }


    pub fn launch(&mut self, time_limit: f64) -> (f64, Vec<TraceData>) {
        let mut best_disto_approx = constants::INF;
        let mut best_approx_tree = RootedTree::new(self.n, 0);
        let mut cur_disto_approx = constants::INF;

        let now = Instant::now();

        let step = 1.0/2.0;

        let mut cur_approx_tree = self.g.random_subtree(&mut self.prng);

        let mut elapsed;
        let mut trace: Vec<TraceData> = Vec::new();

        let mut iter_id = 0;

        while {elapsed = now.elapsed(); elapsed}.as_secs_f64() < time_limit {
            let dist_previous = best_disto_approx;
            while self.temperature > step / 2.0 {
                self.init_strategy(&mut cur_approx_tree, self.k);
                let y = self.get_neighbor(&mut cur_approx_tree, self.k);
                //y.recompute_depths_rec(y.root, 0);
                let ydist = y.heuristic(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);

                if ydist < best_disto_approx {
                    best_approx_tree = y.clone();
                    best_disto_approx = ydist;
                    println!("ydist: {}, k: {}", ydist, self.k);
                }

                if ydist < cur_disto_approx {
                    cur_approx_tree = y;
                    cur_disto_approx = ydist;
                } else {
                    let r = my_rand(&mut self.prng);
                    let proba = 0.0;// (-self.temperature.recip() * self.coef).exp();
                    //println!("proba: {} temperature; {}", proba, self.temperature);
                    if r <= proba {
                        //println!("uphill {} {}", cur_disto_approx, ydist);
                        cur_approx_tree = y;
                        cur_disto_approx = ydist;
                    }
                }

                self.temperature -= step;

            }

            if best_disto_approx < dist_previous {
                self.k = 0;
            } else {
                self.k = (self.k + 1).min(self.neighborhood_strategies.len() - 1);
            }
            iter_id += 1;

            self.temperature = 1.0;
            cur_approx_tree = best_approx_tree.clone();
            cur_disto_approx = best_disto_approx;

            //println!("elapsed: {}", elapsed.as_secs_f64());

        }
        let d = best_approx_tree.distorsion::<T>(&self.g, &self.dist_matrix);
        trace.push(TraceData::new(d, iter_id, elapsed.as_secs_f64()));
        (d, trace)
    }

    pub fn beuh(&mut self, time_limit: f64) -> (f64, Vec<TraceData>) {
        let mut best_disto_approx = constants::INF;
        let mut best_approx_tree = RootedTree::new(self.n, 0);
        let mut cur_disto_approx = constants::INF;

        let now = Instant::now();

        let mut cur_approx_tree = self.g.random_subtree(&mut self.prng);

        let mut elapsed;
        let mut trace: Vec<TraceData> = Vec::new();

        let mut iter_id = 0;

        while {elapsed = now.elapsed(); elapsed}.as_secs_f64() < time_limit {
            let dist_previous = best_disto_approx;
            for _ in 0..2 {
                self.init_strategy(&mut cur_approx_tree, self.k);
                let y = self.get_neighbor(&mut cur_approx_tree, self.k);
                //y.recompute_depths_rec(y.root, 0);
                let ydist = y.heuristic(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);

                if ydist < best_disto_approx {
                    best_approx_tree = y.clone();
                    best_disto_approx = ydist;
                }

                if ydist < cur_disto_approx {
                    cur_approx_tree = y;
                    cur_disto_approx = ydist;
                }

            }

            if best_disto_approx < dist_previous {
                self.k = 0;
                //println!("ydist: {}, k: {}", best_disto_approx, self.k);

            } else {
                self.k = (self.k + 1).min(self.neighborhood_strategies.len() - 1);
            }
            iter_id += 1;

            //println!("elapsed: {}", elapsed.as_secs_f64());

        }
        let d = best_approx_tree.distorsion::<T>(&self.g, &self.dist_matrix);
        trace.push(TraceData::new(d, iter_id, elapsed.as_secs_f64()));
        (d, trace)
    }


}













