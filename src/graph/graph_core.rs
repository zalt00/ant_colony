use crate::{graph::N, my_rand::Prng};

pub trait GraphCore: Clone {
    fn get_neighbors(&self, i: usize) -> &[usize];
    fn get_neighboor_count_unchecked(&self, i: usize) -> usize;
    fn vertex_count(&self) -> usize;
    fn from_edges(n: usize, edges: &Vec<[usize; 2]>) -> Self;
    fn add_edge_unckecked(&mut self, u: usize, v: usize);
    fn reset(&mut self);
    fn clone_empty(&self) -> Self;
    fn get_edges_compressed_vecvec<X: Clone+Copy>(&self, init_value: X) -> (Vec<usize>, Vec<X>);


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



    fn update_dist_matrix(&self, dist_matrix: &mut Vec<u32>) {
        dist_matrix.fill(u32::MAX);
        for u in 0..self.vertex_count() {
            self.bfs(u, dist_matrix);
        }
    }
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

    fn get_edge_betweeness_centrality(&self) -> Vec<f64> {
        use rustworkx_core::petgraph;
        use rustworkx_core::petgraph::visit::EdgeIndexable;
        use rustworkx_core::petgraph::visit::NodeIndexable;

        use rustworkx_core::centrality::edge_betweenness_centrality;

        let n = self.vertex_count();
        let g = petgraph::graph::UnGraph::<usize, ()>::from_edges(self.get_edges_tpl());
        
        let output = edge_betweenness_centrality(&g, false, 500000);
        let mut mat = vec![0.0; n*n];

        for i in g.edge_indices() {
            let endpoints = g.edge_endpoints(i).unwrap();
            let bt = output[EdgeIndexable::to_index(&g, i)].unwrap();
            let u = NodeIndexable::to_index(&g, endpoints.0);
            let v = NodeIndexable::to_index(&g, endpoints.1);

            mat[u + n * v] = bt;
            mat[v + n * u] = bt;
        }

        mat
    }
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

    fn bfs(&self, u: usize, dist_matrix: &mut Vec<u32>) {
        static mut QUEUE: [(usize, u32); N] = [(0, 0); N];
        
        if dist_matrix[u + self.vertex_count() * u] != u32::MAX {
            return
        }

        unsafe{QUEUE[0] = (u, 0)};

        let mut visited = vec![false; self.vertex_count()];
        visited[u] = true;

        let mut i = 0;
        let mut j = 1;
        let n = self.vertex_count();
        while i < j {
            unsafe{
                let (v, d) = QUEUE[i];
                dist_matrix[u + n * v] = d;
                dist_matrix[v + n * u] = d;

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



    fn clustering(&self) -> (Vec<usize>, Vec<usize>) {
        let mut c = 0;
        let mut cv = vec![usize::MAX; self.vertex_count()];
        let mut csizes = vec![0];

        let mut vertices = (0..self.vertex_count()).collect::<Vec<usize>>();
        vertices.sort_by_key(|v| {self.get_neighboor_count_unchecked(*v)});

        for v in vertices.iter().rev() {
            if cv[*v] == usize::MAX {
                cv[*v] = c;
                csizes[c] += 1;
                for u in self.get_neighbors(*v) {
                    if cv[*u] == usize::MAX {
                        cv[*u] = c;
                        csizes[c] += 1;
                    }
                }
                c += 1;
                csizes.push(0);
            }
        }

        (cv, csizes)



    }

}

