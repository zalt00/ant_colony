use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::{graph::Graph, utils::Par};


pub struct ACO {
    pub(crate) n: usize,
    pub(crate) g: Graph,
    pub(crate) dist_matrix: Vec<u32>,
    pub(crate) tree: Graph,
    pub(crate) tree_dist_matrix: Vec<u32>,
    pub tau_matrix: Vec<f64>,
    k: usize,
    c: Par<f64>,
    evap: Par<f64>,
    max_tau: Par<f64>,
    interval_tau: Par<f64>,

    pub(crate) possible_edges: Vec<[usize; 2]>,
    pub(crate) covered_vertices: Vec<bool>,

    pub(crate) prng: Xoshiro256PlusPlus,

    edges: Vec<[usize; 2]>,

    pub(crate) edge_betweeness_centrality: Vec<f64>,
    pub(crate) trace: Vec<f64>,

    pub base_tree: Graph,
    pub base_dist: f64
}

impl ACO {
    pub fn new(g: Graph, k: usize, _alpha: Par<f64>, _beta: Par<f64>, c: Par<f64>, evap: Par<f64>,
    max_tau: Par<f64>, interval_tau: Par<f64>, edge_betweeness_centrality: Vec<f64>, dist_matrix: Vec<u32>) -> ACO {
        let n = g.n;
        //let dist_matrix = g.get_dist_matrix();
        let mut tree_dist_matrix = vec![u32::MAX; n*n];
        for i in 0..n {
            tree_dist_matrix[i + n*i] = 0;
        }

        //let edge_betweeness_centrality = g.get_edge_betweeness_centrality();

        let mut edges = vec![];

        for i in 0..(n-1) {
            for &j in g.get_neighbors(i) {
                if j > i {
                    edges.push([i, j])
                }
            }
        }

        let mut tau_matrix = vec![-1.0; n * n];
        for [i, j] in edges.iter() {
            tau_matrix[*i + n * *j] = max_tau.val;
            tau_matrix[*j + n * *i] = max_tau.val;
        }

        ACO { n, g, dist_matrix, tree: Graph::new_empty(n), tree_dist_matrix,
            tau_matrix, k, c, evap,
            max_tau, interval_tau,
            possible_edges: vec![], covered_vertices: vec![false; n],
            prng: Xoshiro256PlusPlus::seed_from_u64(1245),
            edges,
            edge_betweeness_centrality,
            trace: vec![],
            base_tree: Graph::new_empty(n),
            base_dist: f64::INFINITY
        }
    }

    pub fn new_dummy(g: Graph) -> ACO {
        let n = g.n;
        let dist_matrix = g.get_dist_matrix();
        let mut tree_dist_matrix = vec![u32::MAX; n*n];
        for i in 0..n {
            tree_dist_matrix[i + n*i] = 0;
        }
        let edge_betweeness_centrality = g.get_edge_betweeness_centrality();

        let mut edges = vec![];

        for i in 0..(n-1) {
            for &j in g.get_neighbors(i) {
                if j > i {
                    edges.push([i, j])
                }
            }
        }

        ACO { n, g, dist_matrix, tree: Graph::new_empty(n), tree_dist_matrix,
            tau_matrix: vec![], k: 0, c: Par::new_free(0.0), evap: Par::new_free(0.0),
            max_tau: Par::new_free(0.0), interval_tau: Par::new_free(0.0),
            possible_edges: vec![], covered_vertices: vec![false; n],
            prng: Xoshiro256PlusPlus::seed_from_u64(1245),
            edges,
            edge_betweeness_centrality,
            trace: vec![],
            base_tree: Graph::new_empty(n),
            base_dist: f64::INFINITY
        }
    }

    pub fn get_tau_tab_info(&self) -> (f64, f64) {
        let mut m = 0.0;
        let mut c = 0;
        let mut cm = 0.0_f64;

        for i in 0..(self.n * self.n) {
            if self.tau_matrix[i] > -0.5 {
                m += self.tau_matrix[i];
                cm = cm.max(self.tau_matrix[i]);
                c += 1;
            }
        }

        (m / c as f64, cm)
    }

    pub fn cost(&mut self) -> f64 {
        self.tree.distorsion(&mut self.tree_dist_matrix, &self.dist_matrix)
    }

    pub fn gain_of_edge(&self, i: usize, j: usize) -> f64 {
        //println!("{}", self.edge_betweeness_centrality[i + self.n * j]);
        self.tau_matrix[i + self.n * j] 
            //* self.edge_betweeness_centrality[i + self.n * j].sqrt()//.powf(self.beta.val)
    }
    
    pub fn update_possible_edges0(&mut self, edge_to_remove: usize) -> [usize; 2] {

        let edge = self.possible_edges.swap_remove(edge_to_remove);

        debug_assert!(self.covered_vertices[edge[0]] != self.covered_vertices[edge[1]]);

        let new_vertex: usize = if self.covered_vertices[edge[0]] {edge[1]} else {edge[0]};
        self.covered_vertices[new_vertex] = true;
        let mut i = 0;
        while i < self.possible_edges.len() {
            // en fait on fait pas ca, on fait un xor entre possibles edges et toutes les aretes du voisinage
            let e2: &mut [usize; 2] = self.possible_edges.get_mut(i).unwrap();
            if e2[0] == new_vertex || e2[1] == new_vertex {
                self.possible_edges.swap_remove(i);
            } else {
                i+=1;
            }
        }

        for &v in self.g.get_neighbors(new_vertex) {
            if !self.covered_vertices[v] {
                self.possible_edges.push([new_vertex, v])
            }
        }


        edge
    }

    pub fn update_possible_edges(&mut self, edge_to_remove: usize) -> [usize; 2] {
        let edge = self.possible_edges.swap_remove(edge_to_remove);

        debug_assert!(self.covered_vertices[edge[0]] != self.covered_vertices[edge[1]]);

        let new_vertex: usize = if self.covered_vertices[edge[0]] {edge[1]} else {edge[0]};
        self.update_possible_edges1(new_vertex);    
        self.update_possible_edges2(new_vertex);
        edge
    }

    pub fn update_possible_edges1(&mut self, new_vertex: usize) {

        self.covered_vertices[new_vertex] = true;
        let mut i = 0;
        while i < self.possible_edges.len() {
            let e2 = self.possible_edges[i];
            if e2[0] == new_vertex || e2[1] == new_vertex {
                self.possible_edges.swap_remove(i);
            } else {
                i+=1;
            }
        }
    }
    pub fn update_possible_edges2(&mut self, new_vertex: usize) {

        for &v in self.g.get_neighbors(new_vertex) {
            if !self.covered_vertices[v] {
                self.possible_edges.push([new_vertex, v])
            }
        }
    }

    

    pub fn get_chosen_edge(&self, r: f64) -> usize {
                //let now: Instant = Instant::now();

        // optimisable avec un segment tree
        //println!("{}", self.possible_edges.len());

        let mut s = 0.0;
        for e in self.possible_edges.iter() {
            s += self.gain_of_edge(e[0], e[1]);
        }

        let mut cs = 0.0;
        for (i, e) in self.possible_edges.iter().enumerate() {
            cs += self.gain_of_edge(e[0], e[1]);
            if cs >= r * s {
                //println!("g: {} {}", self.gain_of_edge(e[0], e[1]) / s * self.possible_edges.len() as f64, self.tau_matrix[e[0] + self.n*e[1]]);
                //unsafe{CHEPA += now.elapsed().as_secs_f64();}

                return i;
            }
        }
        panic!()
    }

    pub fn init_origin(&mut self) -> usize{
        self.covered_vertices.fill(false);
        self.possible_edges.clear();
        let origin = (self.prng.next_u64() % self.n as u64) as usize;

        self.covered_vertices[origin] = true;

        for v in self.g.get_neighbors(origin) {
            self.possible_edges.push([origin, *v])
        }
        origin

    }

    pub fn launch(&mut self, iter_count: usize) -> (f64, Graph) {

        let mut glob_best_disto = self.base_dist;
        let mut cur_best_tree: Graph = self.base_tree.clone();
        let mut cur_best_disto: f64 = 

            if self.base_dist.is_finite() {
                cur_best_tree.distorsion_approx(&mut cur_best_tree.get_dist_matrix(), &self.edges, &self.edge_betweeness_centrality)*0.8
            } else {
                f64::INFINITY
            };

        let mut cur_best_tree_edges = self.base_tree.get_edges();
        
        let mut ant_edges: Vec<Vec<[usize; 2]>> = vec![vec![]; self.k];
        let mut dist_values: Vec<(f64, usize)> = vec![(0.0, 0); self.k];

        for _iter_index in 1..(iter_count+1) {

            if _iter_index % 1700 == 0 || _iter_index == iter_count {
                for [i, j] in self.edges.iter() {
                    self.tau_matrix[*i + self.n * *j] = self.max_tau.val;
                    self.tau_matrix[*j + self.n * *i] = self.max_tau.val;
                }
                let disto = cur_best_tree.distorsion(&mut cur_best_tree.get_dist_matrix(), &self.dist_matrix);
                if disto < glob_best_disto {
                    glob_best_disto = disto;
                }
                cur_best_disto = f64::INFINITY;
                cur_best_tree = Graph::new_empty(self.n);
                cur_best_tree_edges = vec![];

            }

            for ant in 0..self.k {
                self.init_origin();
                self.tree.clear();
                for _march_advance in 0..(self.n-1) {
                    let r = self.prng.next_u64() as f64 / u64::MAX as f64;
                    let ei: usize = self.get_chosen_edge(r);

                    let edge: [usize; 2] = self.update_possible_edges(ei);

                    self.tree.add_edge_unckecked(edge[0], edge[1]);

                    ant_edges[ant].push(edge);
                }
                let disto_approx= self.tree.distorsion_approx(&mut self.tree_dist_matrix, &self.edges, &self.edge_betweeness_centrality)*0.8;
                //let disto_approx = self.tree.distorsion(&mut self.tree_dist_matrix, &self.dist_matrix);

                //println!("{} {} {}", disto, disto_approx, disto / disto_approx);
                // println!("disto: {}", disto);
                dist_values[ant] = (disto_approx, ant);

                if disto_approx < cur_best_disto {
                    cur_best_disto = disto_approx;
                    cur_best_tree = self.tree.clone();
                    cur_best_tree_edges = cur_best_tree.get_edges();
                }

            }
            for edge in self.edges.iter() {
                self.tau_matrix[edge[0] + self.n * edge[1]] *= (1.0 - self.evap.val).powi(self.k as i32);
                self.tau_matrix[edge[1] + self.n * edge[0]] *= (1.0 - self.evap.val).powi(self.k as i32);
            }

            dist_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // for ant in 0..self.k {
            //     for edge in ant_edges[ant].drain(..) {
            //         let dtau = self.c.val / dist_values[ant].powi(20);

            //         self.tau_matrix[edge[0] + self.n * edge[1]] += dtau;
            //         self.tau_matrix[edge[1] + self.n * edge[0]] += dtau;
            //     }
            // }
            //println!("{:?}", dist_values);
            // let best_ant = dist_values[0].1;
            // let best_disto = dist_values[0].0;
            // for edge in ant_edges[best_ant].iter() {
            //     let dtau = self.c.val / best_disto.powi(3);

            //     self.tau_matrix[edge[0] + self.n * edge[1]] += dtau;
            //     self.tau_matrix[edge[1] + self.n * edge[0]] += dtau;
            // }

            for edge in cur_best_tree_edges.iter() {
                let dtau = self.c.val / 20.0;

                self.tau_matrix[edge[0] + self.n * edge[1]] += dtau;
                self.tau_matrix[edge[1] + self.n * edge[0]] += dtau;
            }



            for edge in self.edges.iter() {
                let v = self.tau_matrix[edge[0] + self.n * edge[1]]
                    .clamp(self.max_tau.val - self.interval_tau.val, self.max_tau.val);
                self.tau_matrix[edge[0] + self.n * edge[1]] = v;
                self.tau_matrix[edge[1] + self.n * edge[0]] = v;
            }
            self.trace.push(cur_best_disto);
            //println!("{:?}", self.tau_matrix);
        }

        (glob_best_disto, cur_best_tree)

    }
    



}

