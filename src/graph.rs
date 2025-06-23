use core::f64;
use std::{fmt::Debug, time::Instant, u32};

use rustworkx_core::petgraph;
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rustworkx_core::petgraph::visit::{EdgeIndexable, NodeIndexable};

use crate::my_rand::irwin_hall;



pub const N: usize = 10000;

pub fn repr<T: Debug>(mat: &Vec<T>) -> String {
    let mut s= String::new();
    let n = (mat.len() as f64).sqrt() as usize;
    for i in 0..n {
        s.push_str(format!("{:?}\n", &mat[i * n..(i+1) * n]).as_str());
    }

    s
}



#[derive(Debug, Clone)]
pub struct Graph {
    pub(crate) n: usize,
    pub(crate) adj_tab: Vec<usize>
}

pub static mut CHEPA: f64 = 0.0;

impl Graph {
    pub fn new_empty(n: usize) -> Graph {
        Graph { n, adj_tab: vec![0; n*(n+1)] }
    }

    pub fn clear(&mut self) {
        for i in 0..self.n {
            self.adj_tab[i * self.n] = 0;
        }

    }

    pub fn from_petgraph(pg: &petgraph::graph::UnGraph<u32, ()>) -> Graph {
        let mut g = Graph::new_empty(pg.node_count());

        for e in pg.raw_edges() {
            let s = e.source();
            let t = e.target();

            let is = petgraph::visit::NodeIndexable::to_index(&pg, s);
            let it = petgraph::visit::NodeIndexable::to_index(&pg, t);

            g.add_edge_unckecked(is, it);

        }
        g
    }

    pub fn get_edge_betweeness_centrality(&self) -> Vec<f64> {
        use rustworkx_core::petgraph;
        use rustworkx_core::centrality::edge_betweenness_centrality;

        let g = petgraph::graph::UnGraph::<usize, ()>::from_edges(self.get_edges_tpl());
        
        let output = edge_betweenness_centrality(&g, false, 50);
        let mut mat = vec![0.0; self.n*self.n];

        for i in g.edge_indices() {
            let endpoints = g.edge_endpoints(i).unwrap();
            let bt = output[EdgeIndexable::to_index(&g, i)].unwrap();
            let u = NodeIndexable::to_index(&g, endpoints.0);
            let v = NodeIndexable::to_index(&g, endpoints.1);

            mat[u + self.n * v] = bt;
            mat[v + self.n * u] = bt;
        }

        mat
    }


    pub fn get_neighboor_count_unchecked(&self, i: usize) -> usize {
        self.adj_tab[i * self.n]
    }

    pub fn get_edges(&self) -> Vec<[usize; 2]> {
        let mut edges = vec![];

        for i in 0..(self.n-1) {
            for &j in self.get_neighbors(i) {
                if j > i {
                    edges.push([i, j])
                }
            }
        }
        edges
    }

    pub fn get_edges_tpl(&self) -> Vec<(u32, u32)> {
        let mut edges = vec![];

        for i in 0..(self.n-1) {
            for &j in self.get_neighbors(i) {
                if j > i {
                    edges.push((i as u32, j as u32))
                }
            }
        }
        edges
    }

    pub fn incr_neighboor_count_unchecked(&mut self, i: usize) {
        self.adj_tab[i * self.n] += 1;
    }

    pub fn decr_neighboor_count_unchecked(&mut self, i: usize) {
        self.adj_tab[i * self.n] -= 1;
    }

    pub fn add_edge_unckecked(&mut self, i: usize, j: usize) {
        let li = self.get_neighboor_count_unchecked(i);
        self.adj_tab[i * self.n + 1 + li as usize] = j;
        self.incr_neighboor_count_unchecked(i);

        let lj = self.get_neighboor_count_unchecked(j);
        self.adj_tab[j * self.n + 1 + lj as usize] = i;
        self.incr_neighboor_count_unchecked(j);
    }

    pub fn remove_edge_last_added_unckecked(&mut self, i: usize, j: usize) {
        self.decr_neighboor_count_unchecked(i);
        self.decr_neighboor_count_unchecked(j);
    }

    pub fn get_neighbors(&self, i: usize) -> &[usize] {
        &self.adj_tab[i * self.n + 1..i * self.n + self.get_neighboor_count_unchecked(i) + 1]
    }

    pub fn from_string(s: String) -> Option<(f64, Graph)> {

        let data: Vec<&str> = s.split("|").collect();
        let disto: f64 = data[0].parse().ok()?;
        let n: usize = data[1].parse().ok()?;

        let mut g = Graph::new_empty(n);

        for line in data[2].trim().split(";") {
            let l: Vec<&str> = line.split(",").collect();
            let i: usize = l[0].parse().ok()?;
            let j: usize = l[1].parse().ok()?;
            g.add_edge_unckecked(i, j); 
        }
        Some((disto, g))
    }

    pub fn bfs(&self, u: usize, dist_matrix: &mut Vec<u32>) {
        static mut QUEUE: [(usize, u32); N] = [(0, 0); N];
        
        if dist_matrix[u + self.n * u] != u32::MAX {
            return
        }

        unsafe{QUEUE[0] = (u, 0)};

        let mut visited = vec![false; self.n];
        visited[u] = true;

        let mut i = 0;
        let mut j = 1;

        while i < j {
            unsafe{
                let (v, d) = QUEUE[i];
                dist_matrix[u + self.n * v] = d;
                dist_matrix[v + self.n * u] = d;

                i += 1;

                for &nv in self.get_neighbors(v) {
                    if !visited[nv] {
                        visited[nv] = true;
                        QUEUE[j] = (nv, d + 1);
                        j += 1;
                    }
                }
            }
        }
    }

    pub fn get_dist_matrix(&self) -> Vec<u32> {
        let mut dist_matrix = vec![u32::MAX; self.n * self.n];

        for u in 0..self.n {
            self.bfs(u, &mut dist_matrix);
        }

        dist_matrix
    }

    pub fn update_dist_matrix(&self, dist_matrix: &mut Vec<u32>) {
        dist_matrix.fill(u32::MAX);
        for u in 0..self.n {
            self.bfs(u, dist_matrix);
        }
    }

    pub fn distorsion(&mut self, dist_matrix: &mut Vec<u32>, parent_dist_matrix: &Vec<u32>) -> f64 {
        self.update_dist_matrix(dist_matrix);   
        let mut s = 0.0;
        for i in 0..(self.n * self.n) {
            if parent_dist_matrix[i] > 0 {
                s += dist_matrix[i] as f64 / parent_dist_matrix[i] as f64;
            }
        }
        s / self.n as f64 / (self.n-1) as f64
    }

    pub fn distorsion_approx0(&mut self, dist_matrix: &mut Vec<u32>, edges: &Vec<[usize; 2]>, ebc: &Vec<f64>) -> f64 {
        self.update_dist_matrix(dist_matrix);   
        let mut s = 0.0;
        for e in edges.iter() {
            s += dist_matrix[e[0] + self.n * e[1]] as f64 * ebc[e[0] + self.n * e[1]];
        }

        s / self.n as f64 / (self.n-1) as f64

    }

    pub fn distorsion_approx(&mut self, dist_matrix: &mut Vec<u32>, edges: &Vec<[usize; 2]>, ebc: &Vec<f64>) -> f64 {
        self.update_dist_matrix(dist_matrix);   
        self.distorsion_approx2(dist_matrix, edges, ebc)

    }
    pub fn distorsion_approx2(&mut self, dist_matrix: &mut Vec<u32>, edges: &Vec<[usize; 2]>, ebc: &Vec<f64>) -> f64 {
        let mut s = 0.0;
        for e in edges.iter() {
            s += dist_matrix[e[0] + self.n * e[1]] as f64 * ebc[e[0] + self.n * e[1]];
        }

        s / self.n as f64 / (self.n-1) as f64

    }

    pub fn ajout_sommet_arbre_recalculer_dist(&mut self, dist_matrix: &mut Vec<u32>, ens_sommets: &mut Vec<usize>, u: usize, parent: usize) {
        for s in ens_sommets.iter() {
            let v = dist_matrix[*s + self.n * parent];
            dist_matrix[*s + self.n * u] = v + 1;
            dist_matrix[u + self.n * *s] = v + 1;
        }

        ens_sommets.push(u);
    }

    pub fn annuler_recalcul(&mut self, dist_matrix: &mut Vec<u32>, ens_sommets: &mut Vec<usize>, u: usize) {
        for s in ens_sommets.iter() {
            dist_matrix[*s + self.n * u] = u32::MAX;
            dist_matrix[u + self.n * *s] = u32::MAX;
        }

        ens_sommets.pop();
    }

    pub fn stretch_moyen(&self, parent_g: &Graph, dist_matrix: &Vec<u32>, ens_sommets: &Vec<usize>, sommet_dedans: &Vec<bool>) -> f64 {
        
        let mut s = 0;
        let mut c = 0;
        for u in ens_sommets.iter() {
            //println!("ne {:?}", parent_g.get_neighbors(*u));
            for v in parent_g.get_neighbors(*u) {
                if sommet_dedans[*v] {
                    s += dist_matrix[*u + self.n * *v];
                    c += 1;
                }
            }
        }

        s as f64 / c as f64
    }

}

#[derive(Debug, Clone, Copy)]
pub struct Par<T> {
    pub(crate) val: T,
    pub(super) bounds: [T; 2],
    pub(super) std: T
}

impl Par<f64> {
    pub fn derive(&self, prng: &mut Xoshiro256PlusPlus, p: f64) -> Par<f64> {
        let mut prev = self.clone();
        prev.val += irwin_hall(prng) * self.std / p;
        prev.val = prev.val.clamp(self.bounds[0], self.bounds[1]);

        prev
    }

    pub fn new_free(val: f64) -> Par<f64> {
        Par { val, bounds: [0.0, 10.0], std: 0.0 }
    }
}


pub struct ACO {
    pub(crate) n: usize,
    pub(crate) g: Graph,
    pub(crate) dist_matrix: Vec<u32>,
    pub(crate) tree: Graph,
    pub(crate) tree_dist_matrix: Vec<u32>,
    pub tau_matrix: Vec<f64>,
    k: usize,
    alpha: Par<f64>,
    beta: Par<f64>,
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
    pub fn new(g: Graph, k: usize, alpha: Par<f64>, beta: Par<f64>, c: Par<f64>, evap: Par<f64>,
    max_tau: Par<f64>, interval_tau: Par<f64>) -> ACO {
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

        let mut tau_matrix = vec![-1.0; n * n];
        for [i, j] in edges.iter() {
            tau_matrix[*i + n * *j] = max_tau.val;
            tau_matrix[*j + n * *i] = max_tau.val;
        }

        ACO { n, g, dist_matrix, tree: Graph::new_empty(n), tree_dist_matrix,
            tau_matrix, k, alpha, beta, c, evap,
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
            tau_matrix: vec![], k: 0, alpha: Par::new_free(0.0),
            beta: Par::new_free(0.0), c: Par::new_free(0.0), evap: Par::new_free(0.0),
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





