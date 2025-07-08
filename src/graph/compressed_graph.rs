use crate::graph::{graph_core::GraphCore, graph_generator::GraphRng};





pub fn init_compressed_vecvec<T: Clone+Copy>(init_value: T, n: usize, degrees: &Vec<usize>) 
    -> (Vec<usize>, Vec<T>) {
        let mut idx = vec![0; n];
        for i in 1..n {
            idx[i] = idx[i-1] + degrees[i-1];
        }
        let s = idx[n-1] + degrees[n-1];
        let data = vec![init_value; s];

        (idx, data)
    }

pub fn init_compressed_vecvec_idx(n: usize, degrees: &Vec<usize>) 
    -> Vec<usize> {
        let mut idx = vec![0; n];
        for i in 1..n {
            idx[i] = idx[i-1] + degrees[i-1];
        }

        idx
    }

#[derive(Clone)]
pub struct CompressedGraph {
    pub(crate) n: usize,
    idx: Vec<usize>,
    data: Vec<usize>,
    degrees: Vec<usize>
}

impl CompressedGraph {
    pub fn new(n: usize, degrees: &Vec<usize>) -> CompressedGraph {
        let (idx, data) = init_compressed_vecvec(usize::MAX, n, &degrees);
        CompressedGraph { n, idx, data, degrees: vec![0; n] }
    }


    pub fn len(&self) -> usize {
        self.data.len()
    }

    fn update_from_edges(&mut self, edges: &Vec<[usize; 2]>) {
        // no clear
        for &[u, v] in edges {
            self.add_edge_unckecked(u, v);
        }
    }
    
}


impl GraphCore for CompressedGraph {
    fn get_neighbors(&self, i: usize) -> &[usize] {
        &self.data[self.idx[i]..self.idx[i] + self.degrees[i]]
    }

    fn vertex_count(&self) -> usize {
        self.n
    }
    
    fn from_edges(n: usize, edges: &Vec<[usize; 2]>) -> CompressedGraph {
        let mut degrees = vec![0; n];
        for &[u, v] in edges {
            degrees[u] += 1;
            degrees[v] += 1;
        }

        let mut g = Self::new(n, &degrees);
        g.update_from_edges(edges);
        g
    }
    
    fn clone_empty(&self) -> Self {
        Self::new(self.n, &self.degrees)
    }
    
    fn add_edge_unckecked(&mut self, u: usize, v: usize) {
        self.data[self.idx[u] + self.degrees[u]] = v;
        self.degrees[u] += 1;

        self.data[self.idx[v] + self.degrees[v]] = u;
        self.degrees[v] += 1;    }
    
    fn reset(&mut self) {
        self.degrees.fill(0);
    }
    
    fn get_neighboor_count_unchecked(&self, i: usize) -> usize {
        self.degrees[i]
    }
    
    fn get_edges_compressed_vecvec<X: Clone+Copy>(&self, init_value: X) -> (Vec<usize>, Vec<X>) {
        println!("wee");
        init_compressed_vecvec(init_value, self.n, &self.degrees)
    }




}

impl GraphRng for CompressedGraph {}







