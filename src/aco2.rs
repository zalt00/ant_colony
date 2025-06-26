
use rand::{RngCore, SeedableRng};
use crate::my_rand::Prng;

use crate::{graph::{Graph, RootedTree}, my_rand::my_rand};


pub struct SegmentTree (Vec<f64>);

impl SegmentTree {
    pub fn new(n: usize) -> SegmentTree {
        let size = n * 2;
        SegmentTree(vec![0.0; size])
    }

    pub const fn size(&self) -> usize {
        self.0.len()
    }

    pub const fn leaf_count(&self) -> usize {
        self.0.len() / 2
    }

    pub const fn intern_node_count(&self) -> usize {
        self.0.len() / 2
    }

    const fn children_count(&self, vi: usize) -> u8 {
        if vi * 2 + 2 < self.size() {
            2
        } else if vi * 2 + 2 == self.size() {
            1
        } else {
            0
        }
    }

    const fn parent(&self, vi: usize) -> usize {
        (vi-1) / 2
    }

    pub fn global_sum(&self) -> f64 {
        self.0[0]
    }

    pub fn update(&mut self, v: usize, val: f64) {
        let mut vi = v + self.intern_node_count();
        let dv = val - self.0[vi];
        self.0[vi] = val;
        while vi != 0 {
            vi = self.parent(vi);
            self.0[vi] += dv;
        }
    }

    fn _smallest_above_from(&self, vi: usize, val: f64) -> usize {
        //println!("{} {}", vi, val);
        let children_count = self.children_count(vi);
        if children_count == 2 {
            //println!("g{} d{}", self.0[vi * 2 + 1], self.0[vi * 2 + 2]);

            let val_left = self.0[vi * 2 + 1];
            if val_left < val {
                self._smallest_above_from(vi * 2 + 2, val - val_left)
            } else {
                self._smallest_above_from(vi * 2 + 1, val)
            }

        } else if children_count == 1 {
            self._smallest_above_from(vi * 2 + 1, val)
        } else {
            debug_assert!(val <= self.0[vi]);
            debug_assert!(vi >= self.intern_node_count());
            vi
        }
    }

    pub fn smallest_above(&self, val: f64) -> usize {
        self._smallest_above_from(0, val) - self.intern_node_count()
    }
}

pub fn test_segment_tree() {
    let mut sg = SegmentTree::new(5);  // ordre 2, 3, 4, 0, 1

    sg.update(0, 0.2);
    sg.update(1, 0.2);
    sg.update(2, 0.2);
    sg.update(3, 0.2);
    sg.update(4, 0.2);

    assert_eq!(sg.smallest_above(0.1), 2);
    assert_eq!(sg.smallest_above(0.2), 2);
    assert_eq!(sg.smallest_above(0.3), 3);
    assert_eq!(sg.smallest_above(1.0), 1);
    assert_eq!(sg.smallest_above(0.9), 1);
    assert_eq!(sg.smallest_above(0.5), 4);



    sg.update(0, 0.0);
    sg.update(1, 0.0);
    sg.update(2, 0.6);
    sg.update(3, 0.2);
    sg.update(4, 0.2);

    assert_eq!(sg.smallest_above(0.1), 2);
    assert_eq!(sg.smallest_above(0.2), 2);
    assert_eq!(sg.smallest_above(0.3), 2);
    assert_eq!(sg.smallest_above(1.0), 4);
    assert_eq!(sg.smallest_above(0.9), 4);
    assert_eq!(sg.smallest_above(0.5), 2);


}


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
    pub fn reset(&mut self) {
        self.0.fill(-1);
    }
}

pub struct TarjanSolver {
    n: usize,
    uf: Uf,
    mark: Vec<bool>,
    ancestors: Vec<usize>,
    results: Vec<usize>
}

impl TarjanSolver {
    pub fn new(n: usize) -> TarjanSolver {
        TarjanSolver { n, uf: Uf::init(n), mark: vec![false; n], ancestors: vec![0; n], results: vec![0; n*n] }
    }

    fn reset(&mut self) {
        self.uf.reset();
        self.mark.fill(false);
    }

    fn _launch_from(&mut self, u: usize, tree: &RootedTree, g: &Graph) {
        self.ancestors[u] = u;
        for v in tree.get_children(u) {
            self._launch_from(*v, tree, g);
            self.uf.union(u, *v);
            self.ancestors[self.uf.find(u).unwrap() as usize] = u;
        }
        self.mark[u] = true;

        for v in g.get_neighbors(u) {
            if self.mark[*v] {
                let lca = self.ancestors[self.uf.find(*v).unwrap() as usize];
                self.results[u + self.n * v] = lca;
                self.results[v + self.n * u] = lca;
            }
        }
    }

    pub fn launch(&mut self, tree: &RootedTree, g: &Graph) -> &Vec<usize> {
        self.reset();
        self._launch_from(tree.get_root(), tree, g);

        &self.results
    }

    pub fn get_results(&self) -> &Vec<usize> {
        &self.results
    }


}


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
    pub trace: Vec<f64>
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
            edge_betweeness_centrality, base_tree, dist_matrix, trace: vec![] }
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

        for &[u, v] in &self.edges {
            debug_assert!(u < v);
            let emati = u + self.n * v;
            let ei = self.edge_to_index[emati];
            if self.adj_set[ei] {
                self.tau_sg.update(ei, self.tau_matrix[emati]);
            } else {
                self.tau_sg.update(ei, 0.0);
            }
        }
        r
    }

    pub fn get_edge(&mut self) -> usize {
        let s = self.tau_sg.global_sum();
        let rho = my_rand(&mut self.prng);

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



    pub fn launch(&mut self, iter_count: usize, w: f64) -> f64 {

        debug_assert!(self.tau_init <= self.max_tau);

        let mut _cool = 0;
        let mut _pas_cool = 0;


        let mut cur_best_tree; 
        let mut cur_best_disto;
        if let Some(t) = &self.base_tree {
            cur_best_tree = t.clone();
            cur_best_disto = cur_best_tree
                .disto_approx(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality);
        } else {
            cur_best_tree = RootedTree::new(self.n, 0);
            cur_best_disto = f64::INFINITY;
        }
        //println!("{:?}", self.edge_to_index);

        for _iter_id in 1..=iter_count {

            
            let mut iter_best_disto = f64::INFINITY;
            let mut iter_best_tree = RootedTree::new(self.n, 0);

            for _ant_id in 1..=self.k {
                let _r = self.reset_state();

                for _ in 0..(self.n-2) {
                    let ei = self.get_edge();

                    let e = self.edges[ei];

                    debug_assert_ne!(self.covered_vertices[e[0]], self.covered_vertices[e[1]]);

                    let (u, parent) =
                        if self.covered_vertices[e[0]] {(e[1], e[0])} else {(e[0], e[1])};
                    
                    self.update_adjset(u);
                    self.covered_vertices[u] = true;
                    self.tree.add_child(parent, u);

                }

                let ei = self.get_edge();
                let e = self.edges[ei];
                debug_assert_ne!(self.covered_vertices[e[0]], self.covered_vertices[e[1]]);
                let (u, parent) =
                    if self.covered_vertices[e[0]] {(e[1], e[0])} else {(e[0], e[1])};
                self.covered_vertices[u] = true;
                self.tree.add_child(parent, u);

                debug_assert!(self.covered_vertices.iter().all(|x| *x));

                let disto_approx = self.tree.disto_approx(&self.g,
                    &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality);
                
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

        cur_best_tree.distorsion(&self.dist_matrix)
    }
}






