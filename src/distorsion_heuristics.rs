use crate::{graph::Graph, utils::TarjanSolver};
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
    pub fn disto_approx(&self, g: &Graph, edges: &Vec<[usize; 2]>,
            tarjan_solver: &mut TarjanSolver, ebc: &Vec<f64>, _dm: &Vec<u32>) -> Num {

        let lca = tarjan_solver.launch(self, g);

        let mut s = 0.0;

        for &[u, v] in edges {
            let i = u + self.n * v;
            s += (self.depths[u] + self.depths[v] - 2*self.depths[lca[i]]) as f64 * ebc[i];
        }
        
        // let ans = s / self.n as f64 / (self.n - 1) as f64;
        // if ans >10000.0 {
        //     println!("{}", self.to_graph().is_connected());

        //     panic!()
        // }

        s / self.n as f64 / (self.n - 1) as f64
    }

    #[cfg(feature = "stretch_heuristic")]
    pub fn disto_approx(&self, g: &Graph, edges: &Vec<[usize; 2]>,
            tarjan_solver: &mut TarjanSolver, ebc: &Vec<f64>, _dm: &Vec<u32>) -> Num {

        let lca = tarjan_solver.launch(self, g);

        let mut s = 0.0;

        for &[u, v] in edges {
            let i = u + self.n * v;
            s += (self.depths[u] + self.depths[v] - 2*self.depths[lca[i]]) as f64;
        }


        s / self.n as f64 / (self.n - 1) as f64
    }

    #[cfg(not(feature = "use_heuristic"))]
    pub fn disto_approx(&self, _g: &Graph, _edges: &Vec<[usize; 2]>,
            _tarjan_solver: &mut TarjanSolver, _ebc: &Vec<f64>, dm: &Vec<u32>) -> Num {

        let mut t = self.to_graph();

        t.distorsion(&mut t.get_dist_matrix(), &dm)
    }

    #[cfg(feature = "mean_path_heuristic")]
    pub fn disto_approx(&self, _g: &Graph, _edges: &Vec<[usize; 2]>,
            _tarjan_solver: &mut TarjanSolver, _ebc: &Vec<f64>, dm: &Vec<u32>) -> Num {
                use crate::counters;

        counters::incr(0);
        self.new_disto_approx()
    }


    fn s22(&self, u: usize, size_sum: &Vec<Vec<u64>>, sd_sum: &Vec<Vec<u64>>) -> u64 {
        let cn = self.children[u].len();

        let ans = if cn > 0 {
            self.sd(u, size_sum, sd_sum) + self.s22_aux(u, 0, cn, size_sum, sd_sum)
        } else {
            0
        };

        //println!("u={}, ans={}", u, ans);
        ans
    }

    fn s22_aux(&self, u: usize, i: usize, j: usize, size_sum: &Vec<Vec<u64>>, sd_sum: &Vec<Vec<u64>>) -> u64 {
        if j - i == 1 {
            self.s22(self.children[u][i], size_sum, sd_sum)
        } else {
            let k = (j + i) / 2;

            self.s22_aux(u, i, k, size_sum, sd_sum) +
            self.s22_aux(u, k, j, size_sum, sd_sum) +

            (size_sum[u][k] - size_sum[u][i]) * (sd_sum[u][j] - sd_sum[u][k]) +
            (size_sum[u][j] - size_sum[u][k]) * (sd_sum[u][k] - sd_sum[u][i]) +
            2 * (size_sum[u][k] - size_sum[u][i]) * (size_sum[u][j] - size_sum[u][k])

        }
    }

    fn size(&self, u: usize, size_sum: &Vec<Vec<u64>>) -> u64 {
        let cn = self.children[u].len();
        size_sum[u][cn] + 1
    }

    fn sd(&self, u: usize, size_sum: &Vec<Vec<u64>>, sd_sum: &Vec<Vec<u64>>) -> u64 {
        let cn = self.children[u].len();
        sd_sum[u][cn] + self.size(u, size_sum) - 1
    }

    fn precalcul(&self, u: usize, size_sum: &mut Vec<Vec<u64>>, sd_sum: &mut Vec<Vec<u64>>) {
        for (i, &v) in self.children[u].iter().enumerate() {
            self.precalcul(v, size_sum, sd_sum);

            size_sum[u][i+1] = size_sum[u][i] + self.size(v, size_sum);
        }

        for (i, &v) in self.children[u].iter().enumerate() {
            sd_sum[u][i+1] = sd_sum[u][i] + self.sd(v, size_sum, sd_sum);
        }
    }

    fn precalcul_init(&self) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        let mut sum_tab = Vec::with_capacity(self.n);
        for ctab in self.children.iter() {
            let cn = ctab.len();
            sum_tab.push(vec![0; cn + 1]);
        }
        (sum_tab.clone(), sum_tab)
    }

    pub fn new_disto_approx(&self) -> u64 {
        let (mut size_sum, mut sd_sum) = self.precalcul_init();
        self.precalcul(self.root, &mut size_sum, &mut sd_sum);

        // println!("sd {:?}", sd_sum);
        // println!("size {:?}", size_sum);
        // for i in 0..self.n {
        //     println!("u={}, size={}, sd={}", i, self.size(i, &size_sum), self.sd(i, &size_sum, &sd_sum));
        // }

        self.s22(self.root, &size_sum, &sd_sum)
        
    }

    pub fn slow_disto_approx(&self, edges: &Vec<[usize; 2]>, ebc: &Vec<f64>) -> f64 {
        let mut t = self.to_graph();

        t.distorsion_approx(&mut t.get_dist_matrix(), edges, ebc)
    }

    pub fn distorsion(&self, dm: &Vec<u32>) -> f64 {
        let mut t = self.to_graph();

        t.distorsion(&mut t.get_dist_matrix(), &dm)
    }

    pub fn s22_slow(&self) -> f64 {
        let t = self.to_graph();

        t.s22_slow(&mut t.get_dist_matrix())
    }

}