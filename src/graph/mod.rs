use std::usize;
use std::{fmt::Debug, u32};

use bincode::{Decode, Encode};
use rustworkx_core::petgraph;
use serde::{Deserialize, Serialize};

pub mod graph_core;
pub mod graph_generator; 
pub mod compressed_graph;
pub mod graph_serde;

use crate::utils::CompressedVecVec;

use self::graph_core::GraphCore;
use self::graph_generator::GraphRng;

#[cfg(feature="large_graph")]
pub const N: usize = 20_000_000;
#[cfg(not(feature="large_graph"))]
pub const N: usize = 50000;

#[derive(Debug, Clone, Default)]
pub struct MatGraph {
    pub(crate) n: usize,
    pub(crate) adj_tab: Vec<usize>
}

pub static mut CHEPA: f64 = 0.0;

impl MatGraph {
    pub const fn new_really_empty() -> MatGraph {
        MatGraph { n: 0, adj_tab: vec![] }
    }

    pub fn new_empty(n: usize) -> MatGraph {
        MatGraph { n, adj_tab: vec![0; n*(n+1)] }
    }

    pub fn clear(&mut self) {
        for i in 0..self.n {
            self.adj_tab[i * self.n] = 0;
        }
    }

    pub fn from_petgraph(pg: &petgraph::graph::UnGraph<u32, ()>) -> MatGraph {
        let mut g = MatGraph::new_empty(pg.node_count());

        for e in pg.raw_edges() {
            let s = e.source();
            let t = e.target();

            let is = petgraph::visit::NodeIndexable::to_index(&pg, s);
            let it = petgraph::visit::NodeIndexable::to_index(&pg, t);

            g.add_edge_unckecked(is, it);

        }
        g
    }



    pub fn incr_neighboor_count_unchecked(&mut self, i: usize) {
        self.adj_tab[i * self.n] += 1;
    }

    pub fn decr_neighboor_count_unchecked(&mut self, i: usize) {
        self.adj_tab[i * self.n] -= 1;
    }
    
    pub fn remove_edge_last_added_unckecked(&mut self, i: usize, j: usize) {
        self.decr_neighboor_count_unchecked(i);
        self.decr_neighboor_count_unchecked(j);
    }

    pub fn remove_edge_slow(&mut self, edge: [usize; 2]) {
        let [u, v] = edge;
        let i = self.get_neighbors(u).iter().position(|x| {*x == v}).unwrap();
        let arr = self.get_neighbors_mut(u);
        arr[i] = *arr.last().unwrap();
        self.decr_neighboor_count_unchecked(u);

        let i = self.get_neighbors(v).iter().position(|x| {*x == u}).unwrap();
        let arr = self.get_neighbors_mut(v);
        arr[i] = *arr.last().unwrap();
        self.decr_neighboor_count_unchecked(v);
    }

    fn get_neighbors_mut(&mut self, i: usize) -> &mut [usize] {
        let nc = self.get_neighboor_count_unchecked(i);
        &mut self.adj_tab[i * self.n + 1..i * self.n + nc + 1]
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

    pub fn stretch_moyen(&self, parent_g: &MatGraph, dist_matrix: &Vec<u32>, ens_sommets: &Vec<usize>, sommet_dedans: &Vec<bool>) -> f64 {
        
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

impl GraphCore for MatGraph {
    fn get_neighbors(&self, i: usize) -> &[usize] {
        &self.adj_tab[i * self.n + 1..i * self.n + self.get_neighboor_count_unchecked(i) + 1]
    }
    
    fn vertex_count(&self) -> usize {
        self.n
    }
    
    fn from_edges(n:usize, edges: &Vec<[usize; 2]>) -> Self {
        let mut g = Self::new_empty(n);
        for &[u, v] in edges {
            g.add_edge_unckecked(u, v);
        }
        g
    }
    
    fn clone_empty(&self) -> Self {
        Self::new_empty(self.n)
    }
    
    
    fn add_edge_unckecked(&mut self, i: usize, j: usize) {
        let li = self.get_neighboor_count_unchecked(i);
        self.adj_tab[i * self.n + 1 + li as usize] = j;
        self.incr_neighboor_count_unchecked(i);

        let lj = self.get_neighboor_count_unchecked(j);
        self.adj_tab[j * self.n + 1 + lj as usize] = i;
        self.incr_neighboor_count_unchecked(j);
    }
    
    fn reset(&mut self) {
        self.clear();
    }
    
    fn get_neighboor_count_unchecked(&self, i: usize) -> usize {
        self.adj_tab[i * self.n]
    }
    
    fn get_edges_compressed_vecvec<X: Clone+Copy>(&self, init_value: X) -> CompressedVecVec<X> {
        let mut degrees = vec![0; self.n];
        for u in 0..self.n {
            degrees[u] = self.get_neighboor_count_unchecked(u);
        }
        CompressedVecVec::new(init_value, self.n, &degrees)
    }



}

impl GraphRng for MatGraph {}



#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct RootedTree {
    pub n: usize,
    pub parent: Vec<usize>,
    pub arity: Vec<usize>,
    pub depths: Vec<usize>,
    pub root: usize
}

impl RootedTree {
    pub const fn new_really_empty() -> RootedTree {
        RootedTree { n: 0, parent: vec![], arity: vec![], depths: vec![], root: 0}
    }

    pub fn new(n: usize, root: usize) -> RootedTree {
        let parent = vec![usize::MAX; n];

        let mut depths = vec![usize::MAX; n];
        depths[root] = 0;

        RootedTree { n, parent, arity: vec![0; n], depths, root }
    }

    pub fn reset(&mut self, root: usize) {
        self.parent.fill(usize::MAX);
        self.arity.fill(0);
        self.depths.fill(usize::MAX);
        self.depths[root] = 0;
        self.root = root;
    }

    pub fn add_child(&mut self, u: usize, v: usize) {
        // u: parent
        // v: enfant
        self.parent[v] = u;
        self.arity[u] += 1;
        self.depths[v] = self.depths[u] + 1;
    }

    pub fn change_parent(&mut self, u: usize, new_parent: usize) {
        assert!(u != self.root);
        self.arity[self.parent[u]] -= 1;
        self.arity[new_parent] += 1;
        self.parent[u] = new_parent;
    }

    pub fn recompute_arity(&mut self) {
        for u in 0..self.n {
            if u != self.root {
                self.arity[self.parent[u]] += 1;
            }
        }
    }

    pub fn from_graph<T: GraphCore>(g: &T, root: usize) -> RootedTree {
        let mut tree = Self::new(g.vertex_count(), root);

        let mut visited = vec![false; g.vertex_count()];

        fn dfs<T: GraphCore>(u: usize, g: &T, visited: &mut Vec<bool>, tree: &mut RootedTree) {
            visited[u] = true;

            for &v in g.get_neighbors(u) {
                if !visited[v] {
                    tree.add_child(u, v);
                    dfs(v, g, visited, tree);
                }
            }

        }
        dfs(root, g, &mut visited, &mut tree);

        tree
    }

    pub fn update_leaves(&mut self) {}

    pub fn precalcul_sizes(&mut self, _u: usize, tab: &mut Vec<u64>) {
        static mut QUEUE: [usize; 50_000_000] = [0; 50000000];
        let mut i = 0;
        let mut j = 0;
        for u in 0..self.n {
            if self.arity[u] == 0 {
                unsafe{QUEUE[j] = u;}
                j += 1;
            }
        }
        while i < j {
            unsafe{
                let u = QUEUE[i];
                tab[u] += 1;

                if u != self.root {

                    if self.arity[self.parent[u]] == 1 {
                        QUEUE[j] = self.parent[u];
                        j += 1;
                    } else {
                        self.arity[self.parent[u]] -= 1;
                    }

                    tab[self.parent[u]] += tab[u];
                }
                i += 1;
            }
        }
        self.arity.fill(0);
        self.recompute_arity();
    }


    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n {false}
        else {
            self.parent[u] == v || self.parent[v] == u
        }
    }

    pub fn recompute_depths(&mut self) {
        self.depths = vec![usize::MAX; self.n];
        self.depths[self.root] = 0;

        for i in 0..self.n {
            if self.arity[i] == 0 {
                self.recompute_depths_rec(i);
            }
        }
    }

    fn recompute_depths_rec(&mut self, u: usize) {
        if self.depths[u] == usize::MAX {
            self.recompute_depths_rec(self.parent[u]);
            self.depths[u] = self.depths[self.parent[u]] + 1
        }
    }

    pub fn get_children_compressed_vecvec(&mut self) -> CompressedVecVec<usize> {
        let mut children = CompressedVecVec::new(usize::MAX, self.n, &self.arity);
        for u in 0..self.n {
            if u != self.root {
                let p = self.parent[u];
                self.arity[p] -= 1;
                children.get_slice_mut(p)[self.arity[p]] = u;
            }
        }
        self.recompute_arity();
        children
    }

    pub fn reroot<T: GraphCore>(&self, template: &T, root: usize) -> RootedTree {
        RootedTree::from_graph(&self.to_graph(template), root)
    }


}


pub static mut COUNTER_APPROX: usize = 0;
pub static mut COUNTER_NONAPPROX: usize = 0;

pub fn print_counters() {
    unsafe {let v1 = COUNTER_APPROX;
    let v2 = COUNTER_NONAPPROX;
    println!("{} {}", v1, v2)}
}

impl RootedTree {

    pub const fn get_root(&self) -> usize {self.root}

    pub fn fill_graph<T: GraphCore>(&self, tree_buf: &mut T) {
        tree_buf.reset();
        for u in 0..self.n {
            if u != self.root {
                tree_buf.add_edge_unckecked(u, self.parent[u]);
            }
        }
    }

    pub fn to_graph<T: GraphCore>(&self, template: &T) -> T {
        let mut g = template.clone_empty();
        for u in 0..self.n {
            if u != self.root {
                g.add_edge_unckecked(u, self.parent[u]);
            }
        }
        g
    }

    pub fn edges<'a>(&'a self) -> EdgeIterator<'a> {
        EdgeIterator { tree: self, u: 0 }
    }

}


pub struct EdgeIterator<'a> {
    tree: &'a RootedTree,
    u: usize,
}

impl<'a> Iterator for EdgeIterator<'a> {
    type Item = [usize; 2];
    fn next(&mut self) -> Option<Self::Item> {
        if self.u == self.tree.root {
            self.u += 1;
        }
        if self.u >= self.tree.n {
            None 
        } else {

            let res = [self.u, self.tree.parent[self.u]];
            self.u += 1;
            Some(res)
        }
    }
}





