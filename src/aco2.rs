
use core::f64;
use std::time::Instant;

use rand::{RngCore, SeedableRng};
use crate::my_rand::Prng;

use crate::neighborhood::VNS;
use crate::trace::TraceData;
use crate::utils::{SegmentTree, TarjanSolver};
use crate::{graph::{Graph, RootedTree}, my_rand::my_rand};


pub struct ACO2 {
    n: usize,
    g: Graph,
    tree: RootedTree,

    tau_matrix: Vec<f64>,
    tau_sg: SegmentTree,

    adj_set: Vec<bool>,
    covered_vertices: Vec<bool>,

    tarjan_solver: TarjanSolver,

    edge_to_index: Vec<usize>,

    k: usize,
    c: f64,
    evap: f64,
    min_tau: f64,
    max_tau: f64,
    tau_init: f64,

    prng: Prng,

    edges: Vec<[usize; 2]>,

    edge_betweeness_centrality: Vec<f64>,

    base_tree: Option<RootedTree>,

    dist_matrix: Vec<u32>,
    pub trace: Vec<f64>,

    pub vnd_hybrid: bool
}

impl ACO2 {
    pub fn new(g: Graph, k: usize, c: f64, evap: f64, min_tau: f64, max_tau: f64, tau_init: f64,
            seed_u64: u64, base_tree: Option<RootedTree>, edge_betweeness_centrality: Vec<f64>,
        dist_matrix: Vec<u32>) -> ACO2 {
        let n = g.n;
        let tau_matrix = vec![tau_init; n*n];
        let edges = g.get_edges();
        let tau_sg = SegmentTree::new(edges.len());
        let adj_set = vec![false; edges.len()];

        let covered_vertices = vec![false; n];

        let tarjan_solver = TarjanSolver::new(n);

        let mut edge_to_index = vec![usize::MAX; n*n];
        for (ei, &[u, v]) in edges.iter().enumerate() {
            edge_to_index[u + n * v] = ei;
            edge_to_index[v + n * u] = ei;
        }

        let prng = Prng::seed_from_u64(seed_u64);

        //let edge_betweeness_centrality = g.get_edge_betweeness_centrality();

        let tree = RootedTree::new(n, 0);

        ACO2 { n, g, tree, tau_matrix, tau_sg, adj_set,
            covered_vertices, tarjan_solver, edge_to_index, k,
            c, evap, min_tau, max_tau, tau_init, prng, edges,
            edge_betweeness_centrality, base_tree, dist_matrix, trace: vec![], vnd_hybrid: false }
    }


    pub fn reset_state(&mut self) -> usize {
        let r = (self.prng.next_u64() % self.n as u64) as usize;
        self.tree.reset(r);

        //self.tau_sg.0.fill(0.0);

        self.covered_vertices.fill(false);
        self.covered_vertices[r] = true;

        self.adj_set.fill(false);
        for &v in self.g.get_neighbors(r) {
            let ei = self.edge_to_index[r + self.n * v];
            self.adj_set[ei] = true;
        }

        self.tau_sg.reset();

        for &[u, v] in &self.edges {
            debug_assert!(u < v);
            let emati = u + self.n * v;
            let ei = self.edge_to_index[emati];
            if self.adj_set[ei] {
                self.tau_sg.update(ei, self.tau_matrix[emati]);
            }
        }
        r
    }

    pub fn get_edge(&mut self, rho: f64) -> usize {
        let s = self.tau_sg.global_sum();

        self.tau_sg.smallest_above(rho * s)
    }

    pub fn update_adjset(&mut self, u: usize) {
        for &v in self.g.get_neighbors(u) {
            let (i, j) = (u.min(v), u.max(v));
            let ei =  self.edge_to_index[i + self.n * j];

            if self.adj_set[ei] {
                self.tau_sg.update(ei, 0.0);
                self.adj_set[ei] = false;
            } else {
                self.tau_sg.update(ei, self.tau_matrix[i + self.n * j]);
                self.adj_set[ei] = true;
            }
        }

    }

    pub fn update_tau(&mut self, cur_best_tree: &RootedTree) {
        // evaporation
        for &[u, v] in &self.edges {
            let val = (self.tau_matrix[u + self.n * v] * (1.0 - self.evap))
                .clamp(self.min_tau, self.max_tau);
            self.tau_matrix[u + self.n * v] = val;
            self.tau_matrix[v + self.n * u] = val;
        }

        // renforcement
        for u in 0..self.n {
            for &v in cur_best_tree.get_children(u) {
                self.tau_matrix[u + self.n * v] += self.c;
                self.tau_matrix[v + self.n * u] += self.c;
            }
        }
    }

    pub fn update_tau_hybrid(&mut self, cur_best_tree: &RootedTree, iter_best_tree: &RootedTree, w: f64) {
        // evaporation
        for &[u, v] in &self.edges {
            let val = (self.tau_matrix[u + self.n * v] * (1.0 - self.evap))
                .clamp(self.min_tau, self.max_tau);
            self.tau_matrix[u + self.n * v] = val;
            self.tau_matrix[v + self.n * u] = val;
        }

        // renforcement
        for u in 0..self.n {
            for &v in cur_best_tree.get_children(u) {
                self.tau_matrix[u + self.n * v] += self.c * w;
                self.tau_matrix[v + self.n * u] += self.c * w;
            }
        }

        for u in 0..self.n {
            for &v in iter_best_tree.get_children(u) {
                self.tau_matrix[u + self.n * v] += self.c * (1.-w);
                self.tau_matrix[v + self.n * u] += self.c * (1.-w);
            }
        }

    }



    pub fn launch(&mut self, iter_count: usize, w: f64, time_limit: f64, recheck_every: f64) -> (f64, Vec<TraceData>) {

        debug_assert!(self.tau_init <= self.max_tau);

        let mut _cool = 0;
        let mut _pas_cool = 0;

        let has_time_limit = time_limit >= 0.0;

        let now = if has_time_limit {
            Some(Instant::now())
        } else {
            None
        };

        let mut trace = Vec::new();

        let mut tot_best_tree = RootedTree::new(self.n, 0);
        let mut tot_real_disto = f64::INFINITY;

        let mut cur_best_tree; 
        let mut cur_best_disto;

        let mut vns = if self.vnd_hybrid {
            Some(VNS::new(self.g.clone(), 1213, self.edge_betweeness_centrality.clone(), self.dist_matrix.clone(), 0))
        } else {
            None
        };

        if let Some(t) = &self.base_tree {
            cur_best_tree = t.clone();
            cur_best_disto = cur_best_tree
                .disto_approx(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality,
                &self.dist_matrix);
        } else {
            cur_best_tree = RootedTree::new(self.n, 0);
            cur_best_disto = f64::INFINITY;
        }

        let mut recheck_count = 0;

        for _iter_id in 1..=iter_count {

            if has_time_limit {
                let now = now.unwrap();
                let elapsed = now.elapsed();
                trace.push(TraceData::new(tot_real_disto, _iter_id, elapsed.as_secs_f64()));
                if elapsed.as_secs_f64() >= time_limit {
                    break
                }

                if elapsed.as_secs_f64() >= recheck_every * (recheck_count + 1) as f64 {
                    recheck_count += 1;

                    let disto = cur_best_tree.distorsion(&self.dist_matrix);
                    if disto < tot_real_disto {
                        tot_real_disto = disto;
                        tot_best_tree = cur_best_tree.clone();
                    } else {
                        cur_best_tree = tot_best_tree.clone();
                        cur_best_disto = tot_best_tree
                        .disto_approx(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality,
                        &self.dist_matrix);
                    }

                    println!("tot real best disto: {}", tot_real_disto);
                }

            }

            
            let mut iter_best_disto = f64::INFINITY;
            let mut iter_best_tree = RootedTree::new(self.n, 0);

            for _ant_id in 1..=self.k {
                let _r = self.reset_state();

                for _ in 0..(self.n-2) {
                    let rho = my_rand(&mut self.prng);
                    let ei = self.get_edge(rho);

                    let e = self.edges[ei];

                    // assert_ne!(self.covered_vertices[e[0]], self.covered_vertices[e[1]]);

                    let (u, parent) =
                        if self.covered_vertices[e[0]] {(e[1], e[0])} else {(e[0], e[1])};
                    
                    if self.tree.depths[parent] > 10000 {
                        println!("{}", self.adj_set[ei]);
                        
                        println!("proba: {}", self.tau_sg.get(ei));
                        println!("{}", self.tau_sg.global_sum());
                        println!("{}", rho);
                        println!("{} {}", parent, u);
                        println!("{}, {}", ei, self.edges.len());
                        println!("{:?}", &self.tau_sg.get_leaves()[ei-50..ei+50]);
                        panic!()
                    }



                    self.update_adjset(u);
                    self.covered_vertices[u] = true;


                    self.tree.add_child(parent, u);

                }
                let rho = my_rand(&mut self.prng);
                let ei = self.get_edge(rho);
                let e = self.edges[ei];
                assert_ne!(self.covered_vertices[e[0]], self.covered_vertices[e[1]]);
                let (u, parent) =
                    if self.covered_vertices[e[0]] {(e[1], e[0])} else {(e[0], e[1])};
                self.covered_vertices[u] = true;
                self.tree.add_child(parent, u);

                debug_assert!(self.covered_vertices.iter().all(|x| *x));

                

                let disto_approx = if self.vnd_hybrid {
                    let d = self.tree.disto_approx(&self.g,
                        &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality,
                    &self.dist_matrix);

                    let (tree2, d2) = vns.as_mut().unwrap().vnd(self.tree.clone(), d);
                    self.tree = tree2;
                    d2
                } else {
                    self.tree.disto_approx(&self.g,
                        &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality,
                    &self.dist_matrix)
                };
                
                // let _vraie_disto = self.tree.distorsion(&self.dist_matrix);
                // let _vraie_disto_best = cur_best_tree.distorsion(&self.dist_matrix);

                // if (_vraie_disto < _vraie_disto_best && disto_approx > cur_best_disto)
                //     || (_vraie_disto > _vraie_disto_best && disto_approx < cur_best_disto) {
                //         _pas_cool += 1;

                //         println!("{} {} {}", _pas_cool, _cool, _pas_cool as f64 / (_pas_cool as f64 + _cool as f64) * 100.0);
                //     }
                // else {
                //     _cool += 1;
                // }
                if disto_approx < iter_best_disto {
                    iter_best_disto = disto_approx;
                    iter_best_tree = self.tree.clone();
                }

                if disto_approx < cur_best_disto {
                    cur_best_disto = disto_approx;
                    cur_best_tree = self.tree.clone();
                }
                self.trace.push(cur_best_disto);


            }
            // truc qui marchait un peu sinon: reappliquer l'evap entre les deux
            self.update_tau_hybrid(&cur_best_tree, &iter_best_tree, w);

        }

        (cur_best_tree.distorsion(&self.dist_matrix), trace)
    }
}






