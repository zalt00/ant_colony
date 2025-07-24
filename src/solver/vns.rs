use std::time::Instant;

use crate::distorsion_heuristics::constants::INF;
use crate::neighborhood::{NSVal, NeighborhoodStrategies};
use crate::utils::TarjanSolver;
use crate::trace::TraceData;
use crate::my_rand::Prng;
use crate::graph::graph_generator::GraphRng;
use crate::counters;
use crate::graph::graph_core::GraphCore;
use crate::graph::RootedTree;
use crate::distorsion_heuristics::{constants, Num};
use rand::SeedableRng;
use rand::RngCore;


pub struct VNS<T: GraphCore+GraphRng> {
    n: usize,
    g: T,
    tree_buf: T,
    tarjan_solver: TarjanSolver,
    edges: Vec<[usize; 2]>,
    prng: Prng,
    edge_betweeness_centrality: Vec<f64>,

    k: usize, // current neighborhood
    l: usize, // current neighborhood (VND)
    neighborhood_strategies: &'static [NeighborhoodStrategies],

    neighborhood_sample_sizes: &'static [usize],

    dist_matrix: Vec<u32>,

    pub recompute_distorsion: bool,
    pub verbose: bool,

    base_disto: Num
}   



impl<T: GraphCore+GraphRng> VNS<T> {

    pub fn new(g: T, seed_u64: u64, edge_betweeness_centrality: Vec<f64>, dist_matrix: Vec<u32>, mode: usize) -> VNS<T> {
        use NeighborhoodStrategies::*;
        static NEIGHBORHOOD_STRATEGIES: [[NeighborhoodStrategies; 4]; 3] = [
            [SubtreeSubtreeVNS(NSVal::N(1, 50)), SpiderSubtreeSwap(NSVal::N(1, 100)), SubtreeSubtreeSwap(NSVal::N(1, 5)), SpiderSubtreeVNS(NSVal::N(1, 10))],
            [CriticalPathSubtreeRelocation, EdgeSubtreeRelocation, EdgeSwap, CriticalPathSubtreeRelocation],
            [SubtreeSubtreeVNS(NSVal::N(1, 5)), SpiderSubtreeVNS(NSVal::N(1, 10)), SubtreeSubtreeSwap(NSVal::N(1, 3)), SpiderSubtreeSwap(NSVal::N(1, 100))]   
        ];
        static NEIGHBORHOOD_SAMPLE_SIZES: [[usize; 4]; 3] = [
            [25, 60, 2, 25],
            [25, 25, 40, 25],
            [50, 70, 2, 40]
        ];


        // 1.9279: SpiderSubtreeSwap(200), SpiderSubtreeVNS(2000), SpiderSubtreeVNS(100) => 3, 30, 200

        let prng = Prng::seed_from_u64(seed_u64);
        let edges = g.get_edges();
        let n = g.vertex_count();
        let tarjan_solver = TarjanSolver::new(n, &g);
        let tree_buf = g.clone_empty();
        VNS { n, g, tree_buf,
            tarjan_solver, edges, prng, edge_betweeness_centrality,
            k: 0, l: 0, neighborhood_strategies: &NEIGHBORHOOD_STRATEGIES[mode],
            neighborhood_sample_sizes: &NEIGHBORHOOD_SAMPLE_SIZES[mode], dist_matrix,
        recompute_distorsion: false, verbose: true, base_disto: INF }


    }


    pub fn init_strategy(&mut self, _x: &mut RootedTree, i: usize) {
        use NeighborhoodStrategies::*;
        match self.neighborhood_strategies[i] {
            EdgeSwap => {
                // ah x.update_parents();  // todo cette chose n'a pas a exister
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
                
                for _ in 0..10 {
                    if y.edge_swap_random(&mut self.prng, &self.edges) {
                        break
                    }
                }
                y
            },
            EdgeSubtreeRelocation => {
                for _ in 0..10 {
                    if x.subtree_swap_with_random_edge(&mut self.prng, &self.edges, &self.g, &mut self.tree_buf)
                        {break};
                };
                let root = (self.prng.next_u64() % self.n as u64) as usize;
                RootedTree::from_graph(&self.tree_buf, root)
            },
            CriticalPathSubtreeRelocation => {
                x.subtree_swap_with_random_critical_path(&mut self.prng, &self.g, &mut self.tree_buf);
                let root = (self.prng.next_u64() % self.n as u64) as usize;
                RootedTree::from_graph(&self.tree_buf, root)
            },
            CriticalPathSubtreeVNS => {
                x.subtree_vns_with_random_spider(&mut self.prng, self.n.isqrt(), &self.g, &mut self.tree_buf);
                let root = (self.prng.next_u64() % self.n as u64) as usize;
                RootedTree::from_graph(&self.tree_buf, root)
            },
            SpiderSubtreeSwap(n2) => {
                x.subtree_swap_with_random_spider(&mut self.prng, n2.to_n2(self.n), &self.g, &mut self.tree_buf);
                let root = (self.prng.next_u64() % self.n as u64) as usize;
                RootedTree::from_graph(&self.tree_buf, root)
            },
            SpiderSubtreeVNS(n2) => {
                x.subtree_vns_with_random_spider(&mut self.prng, n2.to_n2(self.n), &self.g, &mut self.tree_buf);
                let root = (self.prng.next_u64() % self.n as u64) as usize;
                RootedTree::from_graph(&self.tree_buf, root)
            },
            SubtreeSubtreeSwap(n2) => {
                x.subtree_swap_with_random_subtree(&mut self.prng, n2.to_n2(self.n), &self.g, &mut self.tree_buf);
                let root = (self.prng.next_u64() % self.n as u64) as usize;
                RootedTree::from_graph(&self.tree_buf, root)
            },
            SubtreeSubtreeVNS(n2) => {
                x.subtree_vns_with_random_subtree(&mut self.prng, n2.to_n2(self.n), &self.g, &mut self.tree_buf);
                let root = (self.prng.next_u64() % self.n as u64) as usize;
                RootedTree::from_graph(&self.tree_buf, root)
            }
        }
    }

    pub fn improve(&mut self, mut x: RootedTree, mut xdist: Num, i: usize, start: Instant, time_limit: f64, check_time: bool) -> (RootedTree, Num) {
        // pour le moment: best improvement repetee jusqu'a ne plus avoir d'improvement

        self.init_strategy(&mut x, i);


        let mut keep_going;

        loop {
            keep_going = false;

            let mut iter_best_tree = RootedTree::new(self.n, 0);
            let mut iter_best_disto = constants::INF;
            for _sample_id in 0..self.neighborhood_sample_sizes[i] {
                //println!("{}/{}", _sample_id + 1, self.neighborhood_sample_sizes[i]);
                let mut y = self.get_neighbor(&mut x, i);
                let disty = y.heuristic(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);
            
                if disty < xdist && disty < iter_best_disto {
                    iter_best_disto = disty;
                    iter_best_tree = y;
                    //if self.verbose {println!("euh {}", iter_best_disto);}
                    keep_going = true;  // au moins 1 improvement => on continue
                    //println!("{}", iter_best_disto);
                }
            }
            if !keep_going {break}

            
            assert!(xdist > iter_best_disto);
            x = iter_best_tree;
            xdist = iter_best_disto;



            if check_time {
                if self.verbose {println!("{:.2}%, {}", iter_best_disto as f64 / self.base_disto as f64 * 100.0, i);}

                let elapsed = start.elapsed().as_secs_f64();
                if elapsed > time_limit {
                    break;
                }
            }
        }

        (x, xdist)

    }

    pub fn vnd(&mut self, mut x: RootedTree, mut xdist: Num, start: Instant, time_limit: f64, check_time: bool) -> (RootedTree, Num) {
        self.l = 0;
        while self.l < self.neighborhood_strategies.len() {
            let xdist_previous = xdist;
            (x, xdist) = self.improve(x, xdist, self.l, start, time_limit, check_time);

            if xdist < xdist_previous {
                self.l = 0;
            } else {
                self.l += 1;
            }

            if check_time && start.elapsed().as_secs_f64() > time_limit {
                break;
            }
        }

        (x, xdist)
    }

    pub fn gvns2(&mut self, mut x: RootedTree, niter: usize, time_limit: f64) -> RootedTree {
        let h = x.heuristic(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);
        
        if self.verbose {println!("base dist: {}", h);}
        self.gvns(x, h, niter, time_limit, self.n > 100000).0
    }

    pub fn gvns(&mut self, mut x: RootedTree, mut xdist: Num, niter: usize, time_limit: f64, check_time_in_vnd: bool) -> (RootedTree, Num, f64, Vec<TraceData>) {
        self.base_disto = xdist;
        let mut x_real_dist = f64::INFINITY;
        let has_time_limit = time_limit >= 0.0;

        let mut trace: Vec<TraceData> = vec![];

        let now = if has_time_limit {
            Some(Instant::now())
        } else {
            None
        };

        let recompute_dist = self.recompute_distorsion;


        let (x2, xdist2) = self.vnd(x.clone(), xdist, now.unwrap(), time_limit, check_time_in_vnd);


        // update
        if //x2_real_dist < x_real_dist &&
            xdist2 < xdist {
            (x, xdist) = (x2, xdist2);
            //x_real_dist = x2_real_dist;
        } 

        if self.verbose {println!("after 1 vnd: {}", xdist);}


        for _iter_id in 0..niter {
            //println!("iter number {}", _iter_id + 1);

            if has_time_limit {
                let elapsed = now.unwrap().elapsed();
                trace.push(TraceData::new(x_real_dist, _iter_id, elapsed.as_secs_f64()));

                if elapsed.as_secs_f64() >= time_limit {
                    if self.verbose {
                        println!("elapsed: {:?}", elapsed);
                        println!("iter number {}", _iter_id + 1);
                    }

                    break
                }

            }


            




            self.k = 0;

            while self.k < self.neighborhood_strategies.len() {
                let xdist_previous = xdist;

                // shake
                self.init_strategy(&mut x, self.k);
                if cfg!(feature="verbose") && self.verbose {println!("shake strat {}", self.k)};
                let mut y = self.get_neighbor(&mut x, self.k);
                let ydist = y.heuristic(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);

                // descente
                let (x2, xdist2) = self.vnd(y, ydist, now.unwrap(), time_limit, check_time_in_vnd);
                if recompute_dist {
                    let x2_real_dist = x2.distorsion::<T>(&self.g, &self.dist_matrix);

                    // update
                    if x2_real_dist < x_real_dist &&
                    xdist2 < xdist_previous {
                        (x, xdist) = (x2, xdist2);
                        x_real_dist = x2_real_dist;
                        self.k = 0;
                    } else {
                        self.k += 1
                    }
                } else {

                    // update
                    if //x2_real_dist < x_real_dist &&
                    xdist2 < xdist_previous {
                        (x, xdist) = (x2, xdist2);
                        //x_real_dist = x2_real_dist;
                        self.k = 0;
                    } else {
                        self.k += 1
                    }
                }


            }
            if cfg!(feature="verbose") && self.verbose {println!("dist approx: {:.2}%", xdist as f64 / self.base_disto as f64 * 100.0)};

        }

        //if !recompute_dist {x_real_dist = x.distorsion::<T>(&self.g, &self.dist_matrix)};
        (x, xdist, x_real_dist, trace)
    }

    pub fn gvns_random_start_nonapprox(&mut self, niter: usize) -> (f64, Vec<TraceData>) {
        let mut x = self.g.random_subtree(&mut self.prng);
        let xdist = x.heuristic(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);

        let (_y, _ydist, _, trace) = self.gvns(x, xdist, niter, -1.0, false);
        let y_real_dist = _y.distorsion::<T>(&self.g, &self.dist_matrix);
        (y_real_dist, trace)
    }




    pub fn gvns_random_start_nonapprox_timeout(&mut self, time_limit: f64) -> (f64, Vec<TraceData>) {
        let mut x = self.g.random_subtree(&mut self.prng);
        let xdist = x.heuristic(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);

        let (_y, _ydist, y_real_dist, trace) = self.gvns(x, xdist, 10000, time_limit, self.n > 100000);

        (y_real_dist, trace)
    }

    
    pub fn gvns_random_start_timeout_no_distorsion(&mut self, time_limit: f64) -> (RootedTree, Vec<TraceData>) {
        let mut x = self.g.random_subtree(&mut self.prng);
        let xdist = x.heuristic(&self.g, &self.edges, &mut self.tarjan_solver, &self.edge_betweeness_centrality, &self.dist_matrix);
        self.recompute_distorsion = false;
        let (_y, _ydist, _, trace) = self.gvns(x, xdist, 10000, time_limit, self.n > 100000);

        (_y, trace)
    }


}


