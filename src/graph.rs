use std::usize;
use std::{fmt::Debug, u32};

use rustworkx_core::petgraph;
use rustworkx_core::petgraph::visit::{EdgeIndexable, NodeIndexable};

use crate::compressed_graph::CompressedGraph;
use crate::graph_core::GraphCore;
use crate::graph_generator::GraphRng;


pub const N: usize = 50000;

pub fn repr<T: Debug>(mat: &Vec<T>) -> String {
    let mut s= String::new();
    let n = (mat.len() as f64).sqrt() as usize;
    for i in 0..n {
        s.push_str(format!("{:?}\n", &mat[i * n..(i+1) * n]).as_str());
    }

    s
}


#[derive(Debug, Clone)]
pub struct MatGraph {
    pub(crate) n: usize,
    pub(crate) adj_tab: Vec<usize>
}

pub static mut CHEPA: f64 = 0.0;

impl MatGraph {
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

    pub fn get_neighboor_count_unchecked(&self, i: usize) -> usize {
        self.adj_tab[i * self.n]
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

    pub fn from_string(s: String) -> Option<(f64, MatGraph)> {

        let data: Vec<&str> = s.split("|").collect();
        let disto: f64 = data[0].parse().ok()?;
        let n: usize = data[1].parse().ok()?;

        let mut g = MatGraph::new_empty(n);

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



    pub fn s22_slow(&self, dist_matrix: &mut Vec<u32>) -> f64 {
        self.update_dist_matrix(dist_matrix);   

        let mut s = 0;
        for i in 0..(self.n-1) {
            for j in (i+1)..self.n {
                s += dist_matrix[i + self.n * j] as u64;
            }

        }
        s as f64
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
    
    fn update_from_edges(&mut self, edges: &Vec<[usize; 2]>) {
        self.clear();
        for &[u, v] in edges {
            self.add_edge_unckecked(u, v);
        }
    }
    
    fn clone_empty(&self) -> Self {
        Self::new_empty(self.n)
    }

    fn update_dist_matrix(&self, dist_matrix: &mut Vec<u32>) {
        dist_matrix.fill(u32::MAX);
        for u in 0..self.n {
            self.bfs(u, dist_matrix);
        }
    }
    
    fn get_edge_betweeness_centrality(&self) -> Vec<f64> {
        use rustworkx_core::petgraph;
        use rustworkx_core::centrality::edge_betweenness_centrality;

        let g = petgraph::graph::UnGraph::<usize, ()>::from_edges(self.get_edges_tpl());
        
        let output = edge_betweenness_centrality(&g, false, 500000);
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



}

impl GraphRng for MatGraph {}



#[derive(Clone)]
pub struct RootedTree {
    pub(crate) n: usize,
    pub(crate) children: Vec<Vec<usize>>,
    pub(crate) root: usize,
    pub(crate) depths: Vec<usize>,
    pub(crate) parents: Vec<usize>  // todo cette chose n'a pas a exister
}

impl RootedTree {
    pub fn new(n: usize, root: usize) -> RootedTree {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(vec![])
        }

        let mut depths = vec![usize::MAX; n];
        depths[root] = 0;

        RootedTree { n, children: v, root, depths, parents: vec![] }
    }


    pub fn add_child(&mut self, parent: usize, child: usize) {
        // note: parent doit avoir ete ajoute auparavant
        self.children[parent].push(child);
        self.depths[child] = if let Some(val) = self.depths[parent].checked_add(1) {
            val
        } else {
            println!("{} {}", parent, child);
            println!("{:?}", self.depths);
            println!("\n{:?}", self.children);
            panic!()
        }
    }

    pub fn recompute_depths_rec(&mut self, u: usize, d: usize) {
        self.depths[u] = d;
        for v in self.children[u].clone() {
            self.recompute_depths_rec(v, d + 1);
        }
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

    pub fn get_children(&self, u: usize) -> &[usize] {&self.children[u]}
    pub const fn get_root(&self) -> usize {self.root}

    pub fn reset(&mut self, root: usize) {
        for v in self.children.iter_mut() {
            v.clear();
        }
        self.depths.fill(usize::MAX);
        self.depths[root] = 0;

        self.root = root;
    }

    pub fn fill_graph<T: GraphCore>(&self, tree_buf: &mut T) {
        let mut edges = Vec::with_capacity(self.n - 1);
        for (u, children) in self.children.iter().enumerate() {
            for v in children.iter() {
                edges.push([u, *v]);
            }
        }
        tree_buf.update_from_edges(&edges);
    }

    pub fn to_graph<T: GraphCore>(&self) -> T {
        let mut edges = Vec::with_capacity(self.n - 1);
        for (u, children) in self.children.iter().enumerate() {
            for v in children.iter() {
                edges.push([u, *v]);
            }
        }
        T::from_edges(self.n, &edges)
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

}


impl RootedTree {


    pub fn update_parents(&mut self) {
        if self.parents.is_empty() {
            self.parents = vec![usize::MAX; self.n];
        }

        for (u, ch) in self.children.iter().enumerate() {
            for v in ch {
                self.parents[*v] = u;
            }
        }

        self.parents[self.root] = usize::MAX;
    }



}



