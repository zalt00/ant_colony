use std::time::Instant;

use rand::SeedableRng;

use crate::utils::TarjanSolver;
use crate::solver::{greedy::{greedy_bfs, multiple_greedy_bfs, random_greedy_bfs}, vns::VNS};
use crate::my_rand::Prng;
use crate::graph::{compressed_graph::CompressedGraph, graph_core::GraphCore, graph_generator::GraphRng, RootedTree};


pub mod aco2;
pub mod annealing;
pub mod greedy;
pub mod vns;

pub trait Solver {
    type T: GraphCore+GraphRng;
    fn auto_parameters_solve(g: Self::T, ebc: Vec<f64>, dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree;

    fn auto_solve_no_ebcdm(g: Self::T, seed: u64, time_limit: f64) -> RootedTree {
        Self::auto_parameters_solve(g, vec![], vec![], seed, time_limit)
    }
}

impl<T: GraphCore+GraphRng> Solver for VNS<T> {
    type T = T;
    fn auto_parameters_solve(g: Self::T, ebc: Vec<f64>, dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree {
        assert!(g.is_connected());
        let mut vns: VNS<T> = VNS::new(g, seed, ebc, dm, 2);
        vns.gvns_random_start_timeout_no_distorsion(time_limit).0
    }
}

pub struct VNSWithStart<T: GraphCore+GraphRng, S: Solver, const MODE_LG: usize, const MODE_SG: usize> {
    _vns: VNS<T>,
    _solver: S
}

impl<T: GraphCore+GraphRng, S: Solver<T=T>, const MODE_LG: usize, const MODE_SG: usize> Solver for VNSWithStart<T, S, MODE_LG, MODE_SG> {
    type T = T;

    fn auto_parameters_solve(g: Self::T, ebc: Vec<f64>, dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree {

        let (cc, _ccv) = g.bfs_connected_components();
        assert_eq!(cc, 1);

        let mode = if g.vertex_count() < 1500 {
            MODE_SG
        } else {
            MODE_LG
        };
        if cfg!(feature="verbose") {
            println!("mode: {}", mode);
        }
        let mut vns: VNS<T> = VNS::new(g.clone(), seed, ebc.clone(), dm.clone(), mode);
        vns.recompute_distorsion = false;
        let base_tree = S::auto_parameters_solve(g.clone(), ebc.clone(), dm.clone(), seed + 15, time_limit);

        vns.gvns2(base_tree, 10000, time_limit)
    }
}

pub struct VNSWithStartMode1<T: GraphCore+GraphRng, S: Solver> {
    _vns: VNS<T>,
    _solver: S
}

impl<T: GraphCore+GraphRng, S: Solver<T=T>> Solver for VNSWithStartMode1<T, S> {
    type T = T;

    fn auto_parameters_solve(g: Self::T, ebc: Vec<f64>, dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree {
        let mut vns: VNS<T> = VNS::new(g.clone(), seed, ebc.clone(), dm.clone(), 1);
        vns.recompute_distorsion = false;
        vns.verbose = false;
        let base_tree = S::auto_parameters_solve(g.clone(), ebc.clone(), dm.clone(), seed + 15, time_limit);

        vns.gvns2(base_tree, 10000, time_limit)
    }
}

pub struct BestRandom;

impl Solver for BestRandom {
    type T = CompressedGraph;
    fn auto_parameters_solve(g: Self::T, _ebc: Vec<f64>, _dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree {
        assert!(&g.is_connected());

        let mut prng = Prng::seed_from_u64(seed);
        let now = Instant::now();
        
        let mut tree = g.random_subtree(&mut prng);
        let dm = g.get_dist_matrix();
        let mut disto = tree.distorsion(&g, &dm);

        while now.elapsed().as_secs_f64() < time_limit {
            let tree2 = g.random_subtree(&mut prng);
            let disto2 = tree2.distorsion(&g, &dm);
            if disto2 < disto {
                disto = disto2;
                tree = tree2;
            }
        }

        tree

    }
}

pub struct BestRandomVND;

impl Solver for BestRandomVND {
    type T = CompressedGraph;
    fn auto_parameters_solve(g: Self::T, ebc: Vec<f64>, _dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree {
        assert!(&g.is_connected());

        let mut prng = Prng::seed_from_u64(seed);
        let now = Instant::now();
        
        let mut tree = g.random_subtree(&mut prng);
        let dm = g.get_dist_matrix();
        let mut disto = tree.distorsion(&g, &dm);
        let mut vns = VNS::new(g.clone(), seed, ebc.clone(), dm.clone(), 2);
        let edges = g.get_edges();
        let mut ts = TarjanSolver::new(g.n, &g);
        
        while now.elapsed().as_secs_f64() < time_limit {
            let mut tree2 = g.random_subtree(&mut prng);
            let heuristic = tree2.heuristic(&g, &edges, &mut ts, &ebc, &dm);
            let (tree2, _) = vns.vnd(tree2, heuristic, Instant::now(), -1.0, false);
            let disto2 = tree2.distorsion(&g, &dm);
            if disto2 < disto {
                disto = disto2;
                tree = tree2;
            }
        }

        tree

    }
}


pub struct MultiBFSTree<const K: usize = 50, T: GraphCore+GraphRng = CompressedGraph> {
    __: T
}

impl<const K: usize, T: GraphCore+GraphRng> Solver for MultiBFSTree<K, T> {
    type T = T;

    fn auto_parameters_solve(g: Self::T, _ebc: Vec<f64>, _dm: Vec<u32>, _seed: u64, _time_limit: f64) -> RootedTree {
        multiple_greedy_bfs(&g, K).1
    }
}

pub struct BFSTree<T: GraphCore+GraphRng> {
    __: T
}

impl<T: GraphCore+GraphRng> Solver for BFSTree<T> {
    type T = T;

    fn auto_parameters_solve(g: Self::T, _ebc: Vec<f64>, _dm: Vec<u32>, _seed: u64, _time_limit: f64) -> RootedTree {
        greedy_bfs(&g)
    }
}

pub struct RandomStartBFSTree<T: GraphCore+GraphRng> {
    __: T
}

impl<T: GraphCore+GraphRng> Solver for RandomStartBFSTree<T> {
    type T = T;

    fn auto_parameters_solve(g: Self::T, _ebc: Vec<f64>, _dm: Vec<u32>, _seed: u64, _time_limit: f64) -> RootedTree {
        random_greedy_bfs(&g, _seed)
    }
}


pub struct TestRandom;

impl Solver for TestRandom {
    type T = CompressedGraph;

    fn auto_parameters_solve(g: Self::T, _ebc: Vec<f64>, _dm: Vec<u32>, seed: u64, _time_limit: f64) -> RootedTree {
        g.random_subtree(&mut Prng::seed_from_u64(seed))
    }
}


