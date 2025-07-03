use crate::{compressed_graph::CompressedGraph, graph::N};

pub trait GraphCore: Clone {
    fn get_neighbors(&self, i: usize) -> &[usize];

    fn vertex_count(&self) -> usize;

    fn get_edges(&self) -> Vec<[usize; 2]> {
        let mut edges = vec![];

        for i in 0..(self.vertex_count()-1) {
            for &j in self.get_neighbors(i) {
                if j > i {
                    edges.push([i, j])
                }
            }
        }
        edges
    }

    fn get_edges_tpl(&self) -> Vec<(u32, u32)> {
        let mut edges = vec![];

        for i in 0..(self.vertex_count()-1) {
            for &j in self.get_neighbors(i) {
                if j > i {
                    edges.push((i as u32, j as u32))
                }
            }
        }
        edges
    }

    fn from_edges(n: usize, edges: &Vec<[usize; 2]>) -> Self;
    fn add_edge_unckecked(&mut self, u: usize, v: usize);
    fn reset(&mut self);
    fn new_empty(n: usize) -> Self where Self:Sized;

    fn update_from_edges(&mut self, edges: &Vec<[usize; 2]>);

    fn clone_empty(&self) -> Self;

    fn update_dist_matrix(&self, dist_matrix: &mut Vec<u32>);

    fn get_dist_matrix(&self) -> Vec<u32> {
        let mut dist_matrix = vec![u32::MAX; self.vertex_count() * self.vertex_count()];
        self.update_dist_matrix(&mut dist_matrix);
        dist_matrix
    }
    fn distorsion(&self, my_dist_matrix_buffer: &mut Vec<u32>, parent_dist_matrix: &Vec<u32>) -> f64 {
        self.update_dist_matrix(my_dist_matrix_buffer);   
        let mut s = 0.0;
        for i in 0..(self.vertex_count() * self.vertex_count()) {
            if parent_dist_matrix[i] > 0 {
                s += my_dist_matrix_buffer[i] as f64 / parent_dist_matrix[i] as f64;
            }
        }
        s / self.vertex_count() as f64 / (self.vertex_count()-1) as f64
    }

    fn get_edge_betweeness_centrality(&self) -> Vec<f64>;

    fn bfs_further_vertex(&self, u: usize, construct_path: bool, path: &mut Vec<usize>) -> usize {
        static mut QUEUE: [(usize, u32); N] = [(0, 0); N];
        
        let n = self.vertex_count();

        unsafe{QUEUE[0] = (u, 0)};

        let mut previous = vec![-1; n];
        previous[u] = -2;

        let mut current_furthest = (u, 0);

        let mut i = 0;
        let mut j = 1;

        while i < j {
            unsafe{
                let (v, d) = QUEUE[i];
                //println!("{} {}", v, d);
                if current_furthest.1 < d {
                    current_furthest = (v, d);
                }

                i += 1;

                for &nv in self.get_neighbors(v) {
                    if previous[nv] == -1 {
                        previous[nv] = v as isize;
                        QUEUE[j] = (nv, d + 1);
                        j += 1;
                    }
                }
            }
        }

        if construct_path {
            let mut v = current_furthest.0 as isize;
            while v != -2 {
                assert!(v >= 0);
                path.push(v as usize);
                v = previous[v as usize];
            }
        }

        current_furthest.0
    }

}

