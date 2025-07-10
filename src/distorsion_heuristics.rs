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
    pub fn heuristic<T: GraphCore>(&mut self, g: &T, edges: &Vec<[usize; 2]>,
            tarjan_solver: &mut TarjanSolver, ebc: &Vec<f64>, _dm: &Vec<u32>) -> Num {

        let (cidx, children) = self.get_children_compressed_vecvec();
        let (lca_idx, lca) = tarjan_solver.launch(self, g, &cidx, &children);

        let mut s = 0.0; 

        for u in 0..self.n { 
            for (i, &v) in g.get_neighbors(u).iter().enumerate() {
                let l = lca[lca_idx[u] + i];
                if l < usize::MAX {
                    // assert!(debug_hmap.insert([u.min(v), u.max(v)]), "{} {}", lca.len(),
                    //     lca.iter().fold(0, |i, x| {if *x == usize::MAX {i + 1} else {i}}));

                    s += (self.depths[u] + self.depths[v] - 2*self.depths[l]) as f64 * ebc[u + self.n * v];

                }
            }
        }
        
        // let ans = s / self.n as f64 / (self.n - 1) as f64;
        // if ans >10000.0 {
        //     println!("{}", self.to_graph().is_connected());

        //     panic!()
        // }

        s / self.n as f64 / (self.n - 1) as f64
    } 

    #[cfg(feature = "stretch_heuristic")]
    pub fn heuristic<T: GraphCore>(&mut self, g: &T, _edges: &Vec<[usize; 2]>,
            tarjan_solver: &mut TarjanSolver, _ebc: &Vec<f64>, _dm: &Vec<u32>) -> Num {
                
        let (cidx, children) = self.get_children_compressed_vecvec();

        let (lca_idx, lca) = tarjan_solver.launch(self, g, &cidx, &children);
        let mut s = 0.0;
        for u in 0..self.n {
            for (i, &v) in g.get_neighbors(u).iter().enumerate() {
                let l = lca[lca_idx[u] + i];
                if l < usize::MAX {
                    // assert!(debug_hmap.insert([u.min(v), u.max(v)]), "{} {}", lca.len(),
                    //     lca.iter().fold(0, |i, x| {if *x == usize::MAX {i + 1} else {i}}));

                    s += (self.depths[u] + self.depths[v] - 2*self.depths[l]) as f64;
                }
            }
        }
        s / (lca.len() as f64 / 2.0)
    }

    #[cfg(not(feature = "use_heuristic"))]
    pub fn heuristic<T: GraphCore>(&self, g: &T, _edges: &Vec<[usize; 2]>,
            _tarjan_solver: &mut TarjanSolver, _ebc: &Vec<f64>, dm: &Vec<u32>) -> Num {

        self.distorsion::<T>(g, dm)
    }

    #[cfg(feature = "mean_path_heuristic")]
    pub fn heuristic<T: GraphCore>(&mut self, _g: &T, _edges: &Vec<[usize; 2]>,
            _tarjan_solver: &mut TarjanSolver, _ebc: &Vec<f64>, _dm: &Vec<u32>) -> Num {
                use crate::counters;

        counters::incr(0);
        self.new_disto_approx2()
    }


    fn s22(&self, u: usize, idx: &Vec<usize>, size_sum: &Vec<u64>, sd_sum: &Vec<u64>,
            cidx: &Vec<usize>, children: &Vec<usize>) -> u64 {
        let cn = self.arity[u];

        let ans = if cn > 0 {
            self.sd(u, idx, size_sum, sd_sum) + self.s22_aux(u, 0, cn, idx, size_sum, sd_sum, cidx, children)
        } else {
            0
        };

        //println!("u={}, ans={}", u, ans);
        ans
    }
    fn s22_aux(&self, u: usize, i: usize, j: usize, idx: &Vec<usize>, size_sum: &Vec<u64>, sd_sum: &Vec<u64>,
            cidx: &Vec<usize>, children: &Vec<usize>) -> u64 {
        if j - i == 1 {
            self.s22(children[cidx[u]+i], idx, size_sum, sd_sum, cidx, children)
        } else {
            let k = (j + i) / 2;
            
            self.s22_aux(u, i, k, idx, size_sum, sd_sum, cidx, children) +
            self.s22_aux(u, k, j, idx, size_sum, sd_sum, cidx, children) +

            (size_sum[idx[u]+k] - size_sum[idx[u]+i]) * (sd_sum[idx[u]+j] - sd_sum[idx[u]+k]) +
            (size_sum[idx[u]+j] - size_sum[idx[u]+k]) * (sd_sum[idx[u]+k] - sd_sum[idx[u]+i]) +
            2 * (size_sum[idx[u]+k] - size_sum[idx[u]+i]) * (size_sum[idx[u]+j] - size_sum[idx[u]+k])

        }
    }

    fn size(&self, u: usize, idx: &Vec<usize>, size_sum: &Vec<u64>) -> u64 {
        let cn = self.arity[u];
        size_sum[idx[u] + cn] + 1
    }

    fn sd(&self, u: usize, idx: &Vec<usize>, size_sum: &Vec<u64>, sd_sum: &Vec<u64>) -> u64 {
        let cn = self.arity[u];
        sd_sum[idx[u] + cn] + self.size(u, idx, size_sum) - 1
    }

    fn precalcul(&self, u: usize, idx: &Vec<usize>, size_sum: &mut Vec<u64>, sd_sum: &mut Vec<u64>, 
            cidx: &Vec<usize>, children: &Vec<usize>) {

        for (i, &v) in children[cidx[u]..(cidx[u]+self.arity[u])].iter().enumerate() {
            self.precalcul(v, idx, size_sum, sd_sum, cidx, children);

            size_sum[idx[u] + i+1] = size_sum[idx[u] + i] + self.size(v, idx, size_sum);
        }

        for (i, &v) in children[cidx[u]..(cidx[u]+self.arity[u])].iter().enumerate() {
            sd_sum[idx[u]+i+1] = sd_sum[idx[u]+i] + self.sd(v,idx, size_sum, sd_sum);
        }
    }

    fn precalcul_init(&self) -> (Vec<usize>, Vec<u64>, Vec<u64>) {

        let mut idx = vec![0; self.n];

        for i in 1..self.n {
            let cn = self.arity[i-1];

            idx[i] = idx[i-1] + cn + 1;
        }
        let s = idx[self.n-1] + self.arity[self.n-1] + 1;


        (idx, vec![0; s], vec![0; s])

    }

    pub fn new_disto_approx(&mut self) -> u64 {
        let (cidx, children) = self.get_children_compressed_vecvec();
        let (idx, mut size_sum, mut sd_sum) = self.precalcul_init();
        self.precalcul(self.root, &idx, &mut size_sum, &mut sd_sum, &cidx, &children);

        // println!("sd {:?}", sd_sum);
        // println!("size {:?}", size_sum);
        // for i in 0..self.n {
        //     println!("u={}, size={}, sd={}", i, self.size(i, &size_sum), self.sd(i, &size_sum, &sd_sum));
        // }

        self.s22(self.root, &idx, &size_sum, &sd_sum, &cidx, &children)
        
    }


    pub fn new_disto_approx2(&mut self) -> u64 {
        let mut size = vec![0; self.n];
        self.precalcul_sizes(self.root, &mut size);
        let mut s3 = 0;
        for u in 0..self.n {
            if self.root == u {
                s3 += self.depths[u] as u64 * (1 + self.n as u64)
            } else {
                let su = size[u];
                let spu = size[self.parent[u]];
                s3 += self.depths[u] as u64 * (1 + self.n as u64 - su * (spu - su + 1))
                        + su * (spu - su - 1);
            }
        }
        s3
    }

    // pub fn afficher_des_trucs(&mut self) {
    //     let (idx, mut size_sum, mut sd_sum) = self.precalcul_init();
    //     self.precalcul(self.root, &idx, &mut size_sum, &mut sd_sum);
    //     self.update_parents();
    //     let mut s = 0;
    //     for u in 0..self.n {
    //         if u == self.root {
    //             s += self.sd(u, &idx, &size_sum, &sd_sum);
    //         } else {
    //             s += self.sd(u, &idx, &size_sum, &sd_sum) * (self.size(self.parents[u], &idx, &size_sum) - self.size(u, &idx, &size_sum));
    //         }
    //     }

    //     let mut s2 = 0;
    //     for v in 0..self.n {
    //         let mut u = v;
    //         while u != self.root {
    //             s2 += (self.depths[v] - self.depths[u]) as u64 * 
    //             (self.size(self.parents[u], &idx, &size_sum) - self.size(u, &idx, &size_sum));
    //             u = self.parents[u];
    //         }

    //         s2 += self.depths[v] as u64;
    //     }


    //     let mut s3 = 0;
    //     for u in 0..self.n {
    //         if self.root == u {
    //             s3 += self.depths[u] as u64 * (1 + self.n as u64)
    //         } else {
    //             let su = self.size(u, &idx, &size_sum);
    //             let spu = self.size(self.parents[u], &idx, &size_sum);
    //             s3 += self.depths[u] as u64 * (1 + self.n as u64 - su * (spu - su + 1))
    //                     + su * (spu - su - 1);

                

    //         }
    //     }

    //     println!("s={}", s);
    //     println!("s2={}", s2);
    //     println!("s3={}", s3);
    //     println!("nd={}", self.new_disto_approx());

    //     println!("sd_part_only={}", self.new_disto_approx_sd_part_only());



    // }

    pub fn slow_disto_approx(&self, g: &MatGraph, edges: &Vec<[usize; 2]>, ebc: &Vec<f64>) -> f64 {
        let mut t: MatGraph = self.to_graph(g);

        t.distorsion_approx(&mut t.get_dist_matrix(), edges, ebc)
    }
 
    pub fn distorsion<T: GraphCore>(&self, g: &T, dm: &Vec<u32>) -> f64 {
        let t: T = self.to_graph(g);

        t.distorsion(&mut vec![u32::MAX; self.n*self.n], &dm)
    }

    pub fn s22_slow(&self, g: &MatGraph) -> f64 {
        let t: MatGraph = self.to_graph(g);

        t.s22_slow(&mut t.get_dist_matrix())
    }

}