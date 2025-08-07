use pyo3::pyclass;

use crate::{graph::{graph_core::GraphCore, graph_generator::GraphRng}, utils::CompressedVecVec};







#[derive(Clone, Default)]
#[pyclass]
pub struct CompressedGraph {
    pub(crate) n: usize,
    adj_array: CompressedVecVec<usize>,
    degrees: Vec<usize>  // current degrees (for add_edge_unchecked)
}

impl CompressedGraph {
    pub fn new(n: usize, degrees: &Vec<usize>) -> CompressedGraph {
        let adj_array = CompressedVecVec::new(usize::MAX, n, degrees);
        CompressedGraph { n, adj_array, degrees: vec![0; n] }
    }

    fn update_from_edges(&mut self, edges: &[[usize; 2]]) {
        // no clear
        for &[u, v] in edges {
            self.add_edge_unckecked(u, v);
        }
    }
    
}


impl GraphCore for CompressedGraph {
    fn get_neighbors(&self, i: usize) -> &[usize] {
        &self.adj_array.get_slice(i)[..self.degrees[i]]
    }

    fn vertex_count(&self) -> usize {
        self.n
    }
    
    fn from_edges(n: usize, edges: &[[usize; 2]]) -> CompressedGraph {
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
        self.adj_array.get_slice_mut(u)[self.degrees[u]] = v;
        self.degrees[u] += 1;

        self.adj_array.get_slice_mut(v)[self.degrees[v]] = u;
        self.degrees[v] += 1;    
    }
    
    fn reset(&mut self) {
        self.degrees.fill(0);
    }
    
    fn get_neighboor_count_unchecked(&self, i: usize) -> usize {
        self.degrees[i]
    }
    
    fn get_edges_compressed_vecvec<X: Clone+Copy>(&self, init_value: X) -> CompressedVecVec<X> {
        CompressedVecVec::new(init_value, self.n, &self.degrees)
    }




}

impl GraphRng for CompressedGraph {}







