use std::{fmt::Debug, fs::File, io::{BufWriter, Read}, u32};

use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;


pub const N: usize = 10000;

pub struct Uf(Vec<isize>);

impl Uf {
    pub fn find(&mut self, i: usize) -> Option<isize> {
        let v = self.0.get(i)?;
        if *v < 0 {
            Some(i as isize)
        } else {
            let v2 = *v as usize;
            let c = self.find(v2)?;
            self.0[i] = c;
            Some(c)
        }
    }

    pub fn union(&mut self, i1: usize, i2: usize) -> Option<()> {
        let c1 = self.find(i1)?;
        let c2 = self.find(i2)?;
        let s1 = self.0[c1 as usize];
        let s2 = self.0[c2 as usize];
        if s1 < s2 {  // |c1| > |c2|
            self.0[c2 as usize] = c1;
            self.0[c1 as usize] = s1 + s2;
        } else {
            self.0[c1 as usize] = c2;
            self.0[c2 as usize] = s1 + s2;
        }
        Some(())
    }

    pub fn init(n: usize) -> Uf {
        Uf(vec![-1; n])
    }
}

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

impl Graph {
    pub fn new_empty(n: usize) -> Graph {
        Graph { n, adj_tab: vec![0; n*(n+1)] }
    }

    pub fn get_neighboor_count_unchecked(&self, i: usize) -> usize {
        self.adj_tab[i * self.n]
    }

    pub fn incr_neighboor_count_unchecked(&mut self, i: usize) {
        self.adj_tab[i * self.n] += 1;
    }

    pub fn add_edge_unckecked(&mut self, i: usize, j: usize) {
        let li = self.get_neighboor_count_unchecked(i);
        self.adj_tab[i * self.n + 1 + li as usize] = j;
        self.incr_neighboor_count_unchecked(i);

        let lj = self.get_neighboor_count_unchecked(j);
        self.adj_tab[j * self.n + 1 + lj as usize] = i;
        self.incr_neighboor_count_unchecked(j);
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

    pub fn distorsion(&self, dist_matrix: &mut Vec<u32>, parent_dist_matrix: &Vec<u32>) -> f64 {
        self.update_dist_matrix(dist_matrix);   
        let mut s = 0.0;
        for i in 0..(self.n * self.n) {
            if parent_dist_matrix[i] > 0 {
                s += dist_matrix[i] as f64 / parent_dist_matrix[i] as f64;
            }
        }

        s / self.n as f64 / (self.n-1) as f64
    }
}


pub struct ACO {
    n: usize,
    g: Graph,
    dist_matrix: Vec<u32>,
    tree: Graph,
    tree_dist_matrix: Vec<u32>,
    tau_matrix: Vec<f64>,
    k: usize,
    alpha: u32,
    c: f64,
    evap: f64,

    possible_edges: Vec<[usize; 2]>,
    covered_vertices: Vec<bool>,

    prng: Xoshiro256PlusPlus,

    edges: Vec<[usize; 2]>
}

impl ACO {
    pub fn new(g: Graph, k: usize, alpha: u32, c: f64, evap: f64) -> ACO {
        let n = g.n;
        let dist_matrix = g.get_dist_matrix();
        let tree_dist_matrix = vec![u32::MAX; n*n];

        let mut edges = vec![];

        for i in 0..(n-1) {
            for &j in g.get_neighbors(i) {
                if j > i {
                    edges.push([i, j])
                }
            }
        }

        ACO { n, g, dist_matrix, tree: Graph::new_empty(n), tree_dist_matrix,
            tau_matrix: vec![1.0; n * n], k, alpha, c, evap,
            possible_edges: vec![], covered_vertices: vec![false; n],
            prng: Xoshiro256PlusPlus::seed_from_u64(1245),
            edges
        }
    }

    pub fn cost(&mut self) -> f64 {
        self.tree.distorsion(&mut self.tree_dist_matrix, &self.dist_matrix)
    }

    pub fn gain_of_edge(&self, i: usize, j: usize) -> f64 {
        // pour le moment, pas d'heuristique
        self.tau_matrix[i + self.n * j].powi(self.alpha as i32)
    }
    
    pub fn update_possible_edges(&mut self, edge_to_remove: usize) -> [usize; 2] {
        let edge = self.possible_edges.swap_remove(edge_to_remove);

        debug_assert!(self.covered_vertices[edge[0]] != self.covered_vertices[edge[1]]);

        let new_vertex = if self.covered_vertices[edge[0]] {edge[1]} else {edge[0]};
        self.covered_vertices[new_vertex] = true;
        let mut i = 0;
        while i < self.possible_edges.len() {
            let e2 = self.possible_edges.get_mut(i).unwrap();
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

    pub fn get_chosen_edge(&self, r: f64) -> usize {
        let mut s = 0.0;
        for e in self.possible_edges.iter() {
            s += self.gain_of_edge(e[0], e[1]);
        }

        let mut cs = 0.0;
        for (i, e) in self.possible_edges.iter().enumerate() {
            cs += self.gain_of_edge(e[0], e[1]);
            if cs >= r * s {
                return i;
            }
        }
        panic!()
    }

    pub fn init_origin(&mut self) {
        self.covered_vertices.fill(false);
        self.possible_edges.clear();
        let origin = (self.prng.next_u64() % self.n as u64) as usize;

        self.covered_vertices[origin] = true;

        for v in self.g.get_neighbors(origin) {
            self.possible_edges.push([origin, *v])
        }

    }

    pub fn launch(&mut self, iter_count: usize) -> (f64, Graph) {
        let mut cur_best_disto = f64::INFINITY;
        let mut cur_best_tree = Graph::new_empty(self.n);
        
        let mut ant_edges: Vec<Vec<[usize; 2]>> = vec![vec![]; self.k];
        let mut dist_values = vec![0.0; self.k];

        for _iter_index in 0..iter_count {
            for ant in 0..self.k {
                self.init_origin();
                self.tree.adj_tab.fill(0);
                for _march_advance in 0..(self.n-1) {
                    let r = self.prng.next_u64() as f64 / u64::MAX as f64;
                    let ei = self.get_chosen_edge(r);

                    let edge = self.update_possible_edges(ei);

                    self.tree.add_edge_unckecked(edge[0], edge[1]);

                    ant_edges[ant].push(edge);
                }
                let disto = self.tree.distorsion(&mut self.tree_dist_matrix, &self.dist_matrix);
                // println!("disto: {}", disto);
                dist_values[ant] = disto;

                if disto < cur_best_disto {
                    cur_best_disto = disto;
                    cur_best_tree = self.tree.clone();
                }
            }
            for edge in self.edges.iter() {
                self.tau_matrix[edge[0] + self.n * edge[1]] *= 1.0 - self.evap;
                self.tau_matrix[edge[1] + self.n * edge[0]] *= 1.0 - self.evap;
            }
            for ant in 0..self.k {
                for edge in ant_edges[ant].drain(..) {
                    let dtau = self.c / dist_values[ant];

                    self.tau_matrix[edge[0] + self.n * edge[1]] += dtau;
                    self.tau_matrix[edge[1] + self.n * edge[0]] += dtau;
                }
            }

            //println!("{:?}", self.tau_matrix);
        }

        (cur_best_disto, cur_best_tree)

    }




}





