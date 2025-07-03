use crate::{graph_core::GraphCore, graph_generator::GraphRng};





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

#[derive(Clone)]
pub struct CompressedGraph {
    pub(crate) n: usize,
    idx: Vec<usize>,
    data: Vec<usize>,
    degrees: Vec<usize>
}

impl CompressedGraph {
    pub fn new(n: usize, degrees: Vec<usize>) -> CompressedGraph {
        let (idx, data) = init_compressed_vecvec(usize::MAX, n, &degrees);
        CompressedGraph { n, idx, data, degrees }
    }


    pub fn len(&self) -> usize {
        self.data.len()
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

        let mut g = Self::new(n, degrees);
        g.update_from_edges(edges);
        g
    }
    
    fn update_from_edges(&mut self, edges: &Vec<[usize; 2]>) {
        let mut cur_deg = vec![0; self.n];

        for &[u, v] in edges {
            self.data[self.idx[u] + cur_deg[u]] = v;
            cur_deg[u] += 1;

            self.data[self.idx[v] + cur_deg[v]] = u;
            cur_deg[v] += 1;
        }
    }
    
    fn clone_empty(&self) -> Self {
        Self::new(self.n, self.degrees.clone())
    }
    
    fn update_dist_matrix(&self, dist_matrix: &mut Vec<u32>) {
        todo!()
    }
    
    fn get_edge_betweeness_centrality(&self) -> Vec<f64> {
        todo!()
    }
    
    fn add_edge_unckecked(&mut self, u: usize, v: usize) {
        todo!()
    }
    
    fn reset(&mut self) {
        todo!()
    }
    
    fn new_empty(n: usize) -> Self {
        todo!()
    }

}

impl GraphRng for CompressedGraph {}







