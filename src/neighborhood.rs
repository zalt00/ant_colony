use std::time::Instant;

use rand::{seq::SliceRandom, RngCore, SeedableRng};

use crate::{counters, distorsion_heuristics::{constants, Num}, graph::{Graph, RootedTree, N}, my_rand::Prng, trace::TraceData, utils::{TarjanSolver, Uf}};


impl RootedTree {
    pub fn edge_removable_for_swap(&mut self, ei: usize, edges: &Vec<[usize; 2]>) -> [Vec<usize>; 2]
    {

        // essentiellement, renvoie un chemin dans l'arbre entre les deux extremites de ei
        let [u, v] = edges[ei];

        if self.parents[u] != v && self.parents[v] != u {

            let mut resu = vec![];
            let mut resv = vec![];
            let mut wu = u;
            let mut wv = v;

            while self.depths[wu] > self.depths[wv] {
                resu.push(wu);
                wu = self.parents[wu];
            }

            while self.depths[wv] > self.depths[wu] {
                resv.push(wv);
                wv = self.parents[wv];
            }

            while wu != wv {
                // println!("w {} {} {}", wu, wv, self.root);
                // println!("{} {}", self.depths[wu], self.depths[wv]);
                resu.push(wu);
                resv.push(wv);

                wu = self.parents[wu];
                if wv == usize::MAX {
                    println!("{:?} {:?} {}", resu, resv, self.root);
                }
                wv = self.parents[wv];
            }

            resu.push(wu);
            resv.push(wv);

            [resu, resv]
        } else {
            [vec![], vec![]]
        }
    }

    pub fn do_the_edge_swap(&mut self, dt_rm: &Vec<usize>, dt_oth: &Vec<usize>, rmi: usize) {
        {
            let mut wi = rmi;
            // println!("rmi={}", rmi);

            while wi > 0 {

                self.parents[dt_rm[wi]] = dt_rm[wi - 1]; // change le parent
                self.children[dt_rm[wi-1]].push(dt_rm[wi]);

                let mut rmj = usize::MAX;
                for (j, &node) in self.children[dt_rm[wi + 1]].iter().enumerate() {
                    if node == dt_rm[wi] {
                        rmj = j
                    }
                }
                self.children[dt_rm[wi + 1]].swap_remove(rmj); // supprime l'enfant du parent

                wi -= 1;
            }
        }
        self.parents[dt_rm[0]] = dt_oth[0];
        self.children[dt_oth[0]].push(dt_rm[0]);
        let mut rmj = usize::MAX;
        for (j, &node) in self.children[dt_rm[1]].iter().enumerate() {
            if node == dt_rm[0] {
                rmj = j
            }
        }
        self.children[dt_rm[1]].swap_remove(rmj); // supprime l'enfant du parent


        
        // aaaaarg j'avais oublie a ce truc -> dans l'idee pas forcement utile de tout calculer
        self.recompute_depths_rec(self.root, 0);

    }

    pub fn edge_swap_random(&mut self, prng: &mut Prng, edges: &Vec<[usize; 2]>) -> bool {
        // /!\ appeler update_parent avant

        // pour tester, essayer d'enlever plus tard
        //self.update_parents();

        let ei = (prng.next_u64() % edges.len() as u64) as usize;
        // println!("{:?}", edges[ei]);
        let dt = self.edge_removable_for_swap(ei, edges);
        // println!("{:?}", dt);

        let [resu, resv] = &dt;
        if resu.len() == 0 {return false;}
        let k = resu.len() + resv.len() - 2;
        let mut rk = (prng.next_u64() % k as u64) as usize;

        let mut dtrmi = 0;
        let mut dtothi = 1;
        // println!("{:?} k={}  {}", dt, k, rk);

        if rk >= resu.len() - 1 {
            rk -= resu.len() - 1;
            dtrmi = 1;
            dtothi = 0;
        }


        self.do_the_edge_swap(&dt[dtrmi], &dt[dtothi], rk);

        true
    }

    pub fn subtree_swap_with_edge(&mut self, ei: usize, prng: &mut Prng,
        edges: &Vec<[usize; 2]>, g: &Graph, tree_buf: &mut Graph) -> bool
    {

        self.update_parents();
        let dt = self.edge_removable_for_swap(ei, edges);
        if dt[0].len() == 0 {
            return false;
        }
        let mut vertices = vec![];

        for &v in dt[0].iter() {
            vertices.push(v);
        }
        for &v in dt[1][0..(dt[1].len() - 1)].iter() {
            vertices.push(v);
        }

        self.subtree_swap_with_vertices(prng, &vertices, g, tree_buf);

        true

    }

    pub fn subtree_swap_with_random_edge(&mut self, prng: &mut Prng, 
        edges: &Vec<[usize; 2]>, g: &Graph, tree_buf: &mut Graph) -> bool {
        
        let ei = (prng.next_u64() % edges.len() as u64) as usize;
        self.subtree_swap_with_edge(ei, prng, edges, g, tree_buf)
    }

    // critical path subtree swap
    pub fn shuffle_tree_and_random_leaf(&mut self, prng: &mut Prng) -> usize {
        for children in self.children.iter_mut() {
            children.shuffle(prng);
        }

        let mut v = self.root;

        loop {
            if self.children[v].is_empty() {
                return v
            }

            v = self.children[v][0];
        }
    }

    pub fn get_critical_path(&mut self, prng: &mut Prng, tree_buf: &mut Graph) -> Vec<usize> {
        let mut ans = vec![];

        let leaf = self.shuffle_tree_and_random_leaf(prng);
        //println!("{}", leaf);
        self.fill_graph(tree_buf);

        let leaf2 = tree_buf.bfs_further_vertex(leaf, false, &mut vec![]);
        //println!("{}", leaf2);
        let _ = tree_buf.bfs_further_vertex(leaf2, true, &mut ans);

        ans
    }

    pub fn subtree_swap_with_random_critical_path(&mut self, prng: &mut Prng, g: &Graph, tree_buf: &mut Graph) {
        let cp = self.get_critical_path(prng, tree_buf);
        self.subtree_swap_with_vertices(prng, &cp, g, tree_buf)
    }
    
    fn subtree_swap_with_vertices(&self, prng: &mut Prng, vertices: &Vec<usize>,
        g: &Graph, tree_buf: &mut Graph)
    {
        //println!("{:?}", vertices);

        let mut covered_vertices = vec![false; self.n];
        //let vertices = self.get_critical_path(prng, tree_buf);
        tree_buf.clear();
        for &v in vertices {
            covered_vertices[v] = true;
        }

        //println!("{:?}", vertices);

        let mut possible_edges = vec![];

        for &v in vertices.iter() {
            for &w in g.get_neighbors(v) {
                if v < w && covered_vertices[w] {
                    possible_edges.push([v, w])
                }
            }
        }

        possible_edges.shuffle(prng);

        let n2 = vertices.len();
        let mut uf = Uf::init(self.n);

        let mut m = 0;
        let mut i = 0;
        while m < n2 - 1 {
            let [u, v] = possible_edges[i];

            if uf.find(u) != uf.find(v) {
                uf.union(u, v);
                tree_buf.add_edge_unckecked(u, v);
                m += 1;
            }

            i += 1;
        }

        for (u, children) in self.children.iter().enumerate() {
            for &v in children.iter() {
                if !covered_vertices[u] || !covered_vertices[v] {
                    tree_buf.add_edge_unckecked(u, v);
                }
            }
        }

        

    }


}


impl Graph {
    pub fn bfs_further_vertex(&self, u: usize, construct_path: bool, path: &mut Vec<usize>) -> usize {
        static mut QUEUE: [(usize, u32); N] = [(0, 0); N];
        
        unsafe{QUEUE[0] = (u, 0)};

        let mut previous = vec![-1; self.n];
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



#[derive(Debug, Clone, Copy)]
pub enum NeighborhoodStrategies {
    EdgeSwap,
    EdgeSubtreeRelocation,
    CriticalPathSubtreeRelocation
}

pub struct VNS {
    n: usize,
    g: Graph,
    tree_buf: Graph,
    tarjan_solver: TarjanSolver,
    edges: Vec<[usize; 2]>,
    prng: Prng,
    edge_betweeness_centrality: Vec<f64>,

    k: usize, // current neighborhood
    l: usize, // current neighborhood (VND)
    neighborhood_strategies: &'static [NeighborhoodStrategies],

    neighborhood_sample_sizes: &'static [usize],

    dist_matrix: Vec<u32>
}   

impl VNS {

    pub fn new(g: Graph, seed_u64: u64, edge_betweeness_centrality: Vec<f64>, dist_matrix: Vec<u32>, mode: usize) -> VNS {
        use NeighborhoodStrategies::*;
        static NEIGHBORHOOD_STRATEGIES: [[NeighborhoodStrategies; 3]; 3] = [
            [EdgeSubtreeRelocation, CriticalPathSubtreeRelocation, EdgeSubtreeRelocation],
            [CriticalPathSubtreeRelocation, EdgeSubtreeRelocation, EdgeSwap],
            [EdgeSubtreeRelocation, CriticalPathSubtreeRelocation, EdgeSwap]   
        ];
        static NEIGHBORHOOD_SAMPLE_SIZES: [[usize; 3]; 3] = [
            [40, 25, 25],
            [25, 25, 40],
            [40, 25, 25]
        ];

        let prng = Prng::seed_from_u64(seed_u64);
        let edges = g.get_edges();
        let n = g.n;


        VNS { n, g, tree_buf: Graph::new_empty(n),
            tarjan_solver: TarjanSolver::new(n), edges, prng, edge_betweeness_centrality,
            k: 0, l: 0, neighborhood_strategies: &NEIGHBORHOOD_STRATEGIES[mode],
            neighborhood_sample_sizes: &NEIGHBORHOOD_SAMPLE_SIZES[mode], dist_matrix }


    }


    pub fn init_strategy(&mut self, x: &mut RootedTree, i: usize) {
        use NeighborhoodStrategies::*;
        match self.neighborhood_strategies[i] {
            EdgeSwap => {
                x.update_parents();  // todo cette chose n'a pas a exister
            },
            _ => ()
        }
    }

    pub fn get_neighbor(&mut self, x: &mut RootedTree, i: usize) -> RootedTree {
        // /!\ appeler init_strategy avant
        counters::incr(1);

        use NeighborhoodStrategies::*;
        match self.neighborhood_strategies[i] {
            EdgeSwap => {
                let mut y = x.clone();
                //self.tarjan_solver.launch(&y, &self.g);
                while !y.edge_swap_random(&mut self.prng, &self.edges) {};
                y
            },
            EdgeSubtreeRelocation => {
                while !x.subtree_swap_with_random_edge(&mut self.prng, &self.edges, &self.g, &mut self.tree_buf) {};
                let root = (self.prng.next_u64() % self.n as u64) as usize;
                RootedTree::from_graph(&self.tree_buf, root)
            },
            CriticalPathSubtreeRelocation => {
                x.subtree_swap_with_random_critical_path(&mut self.prng, &self.g, &mut self.tree_buf);
                let root = (self.prng.next_u64() % self.n as u64) as usize;
                RootedTree::from_graph(&self.tree_buf, root)
            }
        }
    }

    pub fn improve(&mut self, mut x: RootedTree, mut xdist: Num, i: usize) -> (RootedTree, Num) {
        // pour le moment: best improvement repetee jusqu'a ne plus avoir d'improvement

        self.init_strategy(&mut x, i);


        let mut keep_going;

        loop {
            keep_going = false;

            let mut iter_best_tree = RootedTree::new(self.n, 0);
            let mut iter_best_disto = constants::INF;
            for _sample_id in 0..self.neighborhood_sample_sizes[i] {
                let y = self.get_neighbor(&mut x, i);
                let disty = y.disto_approx(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);
            
                if disty < xdist && disty < iter_best_disto {
                    iter_best_disto = disty;
                    iter_best_tree = y;
                    keep_going = true;  // au moins 1 improvement => on continue
                }
            }
            
            if !keep_going {break}

            x = iter_best_tree;
            xdist = iter_best_disto;
        }

        (x, xdist)

    }

    pub fn vnd(&mut self, mut x: RootedTree, mut xdist: Num) -> (RootedTree, Num) {
        self.l = 0;
        while self.l < self.neighborhood_strategies.len() {
            let xdist_previous = xdist;
            (x, xdist) = self.improve(x, xdist, self.l);

            if xdist < xdist_previous {
                self.l = 0;
            } else {
                self.l += 1;
            }
        }

        (x, xdist)
    }

    pub fn gvns(&mut self, mut x: RootedTree, mut xdist: Num, niter: usize, time_limit: f64) -> (RootedTree, Num, f64, Vec<TraceData>) {

        let mut x_real_dist = f64::INFINITY;
        let has_time_limit = time_limit >= 0.0;

        let mut trace: Vec<TraceData> = vec![];

        let now = if has_time_limit {
            Some(Instant::now())
        } else {
            None
        };


        for _iter_id in 0..niter {
            //println!("iter number {}", _iter_id + 1);

            if has_time_limit {
                let elapsed = now.unwrap().elapsed();
                trace.push(TraceData::new(x_real_dist, _iter_id, elapsed.as_secs_f64()));

                if elapsed.as_secs_f64() >= time_limit {
                    println!("elapsed: {:?}", elapsed);
                    println!("iter number {}", _iter_id + 1);

                    break
                }

            }


            self.k = 0;

            while self.k < self.neighborhood_strategies.len() {
                let xdist_previous = xdist;

                // shake
                self.init_strategy(&mut x, self.k);
                let y = self.get_neighbor(&mut x, self.k);
                let ydist = y.disto_approx(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);

                // descente
                let (x2, xdist2) = self.vnd(y, ydist);
                let x2_real_dist = x2.distorsion(&self.dist_matrix);

                // update
                if x2_real_dist < x_real_dist && xdist2 < xdist_previous {
                    (x, xdist) = (x2, xdist2);
                    x_real_dist = x2_real_dist;
                    self.k = 0;
                } else {
                    self.k += 1
                }
            }
            if cfg!(feature="verbose") {println!("dist approx: {}, disto {}", xdist, x_real_dist)};

        }


        (x, xdist, x_real_dist, trace)
    }

    pub fn gvns_random_start_nonapprox(&mut self, niter: usize) -> (f64, Vec<TraceData>) {
        let x = self.g.random_subtree(&mut self.prng);
        let xdist = x.disto_approx(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);

        let (_y, _ydist, y_real_dist, trace) = self.gvns(x, xdist, niter, -1.0);

        (y_real_dist, trace)
    }

    pub fn gvns_random_start_nonapprox_timeout(&mut self, time_limit: f64) -> (f64, Vec<TraceData>) {
        let x = self.g.random_subtree(&mut self.prng);
        let xdist = x.disto_approx(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);

        let (_y, _ydist, y_real_dist, trace) = self.gvns(x, xdist, 10000, time_limit);

        (y_real_dist, trace)
    }

}














































