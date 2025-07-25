use crate::graph::graph_core::GraphCore;
use crate::{graph::MatGraph, utils::TarjanSolver};
use crate::graph::RootedTree;

#[cfg(not(feature = "mean_path_heuristic"))]
pub type Num = f64;

#[cfg(feature = "mean_path_heuristic")]
pub type Num = u64;


pub mod constants {
    use crate::distorsion_heuristics::Num;

    #[cfg(not(feature = "mean_path_heuristic"))]
    pub const INF: Num = f64::INFINITY;
    
    #[cfg(feature = "mean_path_heuristic")]
    pub const INF: Num = u64::MAX;
}



impl RootedTree {
    

    #[cfg(feature = "ebc_stretch_heuristic")]
    pub fn heuristic<T: GraphCore>(&mut self, g: &T, _edges: &Vec<[usize; 2]>,
            tarjan_solver: &mut TarjanSolver, ebc: &Vec<f64>, _dm: &Vec<u32>) -> Num {
        
        self.stretch_ebc(g, tarjan_solver, ebc)
    }

    #[cfg(feature = "stretch_heuristic")]
    pub fn heuristic<T: GraphCore>(&mut self, g: &T, _edges: &Vec<[usize; 2]>,
            tarjan_solver: &mut TarjanSolver, _ebc: &Vec<f64>, _dm: &Vec<u32>) -> Num {
                
        self.stretch(g, tarjan_solver)
    }

    pub fn stretch<T: GraphCore>(&mut self, g: &T, tarjan_solver: &mut TarjanSolver) -> f64 {
            
        let children = self.get_children_compressed_vecvec();

        let lca = tarjan_solver.launch(self, g, &children);
        let mut s = 0.0;
        for u in 0..self.n {
            for (i, &v) in g.get_neighbors(u).iter().enumerate() {
                let l = lca.get_slice(u)[i];
                if l < usize::MAX {
                    s += (self.depths[u] + self.depths[v] - 2*self.depths[l]) as f64;
                }
            }
        }
        s / (lca.len() as f64 / 2.0)
    }

    pub fn stretch_ebc<T: GraphCore>(&mut self, g: &T, tarjan_solver: &mut TarjanSolver, ebc: &Vec<f64>) -> f64 {
            
        let children = self.get_children_compressed_vecvec();

        let lca = tarjan_solver.launch(self, g, &children);
        let mut s = 0.0;
        for u in 0..self.n {
            for (i, &v) in g.get_neighbors(u).iter().enumerate() {
                let l = lca.get_slice(u)[i];
                if l < usize::MAX {
                    s += (self.depths[u] + self.depths[v] - 2*self.depths[l]) as f64 * ebc[u + self.n*v];
                }
            }
        }
        s / (lca.len() as f64 / 2.0)
    }

    #[cfg(not(feature = "use_heuristic"))]
    pub fn heuristic<T: GraphCore>(&mut self, g: &T, _edges: &Vec<[usize; 2]>,
            _tarjan_solver: &mut TarjanSolver, _ebc: &Vec<f64>, dm: &Vec<u32>) -> Num {

        self.distorsion::<T>(g, dm)  
    }

    #[cfg(feature = "mean_path_heuristic")]
    pub fn heuristic<T: GraphCore>(&mut self, _g: &T, _edges: &Vec<[usize; 2]>,
            _tarjan_solver: &mut TarjanSolver, _ebc: &Vec<f64>, _dm: &Vec<u32>) -> Num {
                use crate::counters;

        counters::incr(0);
        self.distance_sum()
    }

    pub fn distance_sum(&mut self) -> u64 {
        let mut size = vec![0; self.n];
        self.precalcul_sizes(self.root, &mut size);
        let mut s3 = 0;
        for u in 0..self.n {
            let su = size[u];
            s3 += su * (self.n as u64 - su);
        }
        s3
    }

    pub fn slow_disto_approx(&self, g: &MatGraph, edges: &Vec<[usize; 2]>, ebc: &Vec<f64>) -> f64 {
        let mut t: MatGraph = self.to_graph(g);

        t.distorsion_approx(&mut t.get_dist_matrix(), edges, ebc)
    }
 
    pub fn distorsion<T: GraphCore>(&self, g: &T, dm: &Vec<u32>) -> f64 {
        let t: T = self.to_graph(g);

        t.distorsion(&mut vec![u32::MAX; self.n*self.n], &dm)
    }

}