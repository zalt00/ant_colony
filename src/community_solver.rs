use std::fs::File;
use std::{collections::HashMap, time::Instant};

use pyo3::ffi::c_str;
use pyo3::types::PyDict;
use rand::{RngCore, SeedableRng};

use crate::greedy::{greedy_bfs, multiple_greedy_bfs};
use crate::vns::VNS;
use crate::utils::{HashMapExt, PairExt, TarjanSolver};
use crate::my_rand::Prng;
use crate::graph::{compressed_graph::CompressedGraph, graph_core::GraphCore};
use crate::graph::RootedTree;
use crate::graph::graph_generator::{Data, GraphRng};

pub trait Solver {
    type T: GraphCore+GraphRng;
    fn auto_parameters_solve(g: Self::T, ebc: Vec<f64>, dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree;
}

impl<T: GraphCore+GraphRng> Solver for VNS<T> {
    type T = T;
    fn auto_parameters_solve(g: Self::T, ebc: Vec<f64>, dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree {
        assert!(g.is_connected());
        let mut vns: VNS<T> = VNS::new(g, seed, ebc, dm, 2);
        vns.gvns_random_start_timeout_no_distorsion(time_limit).0
    }
}

pub struct VNSWithStart<T: GraphCore+GraphRng, S: Solver> {
    _vns: VNS<T>,
    _solver: S
}

impl<T: GraphCore+GraphRng, S: Solver<T=T>> Solver for VNSWithStart<T, S> {
    type T = T;

    fn auto_parameters_solve(g: Self::T, ebc: Vec<f64>, dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree {
        let mut vns: VNS<T> = VNS::new(g.clone(), seed, ebc.clone(), dm.clone(), 2);
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
            let (tree2, _) = vns.vnd(tree2, heuristic);
            let disto2 = tree2.distorsion(&g, &dm);
            if disto2 < disto {
                disto = disto2;
                tree = tree2;
            }
        }

        tree

    }
}


pub struct BFSTree<T: GraphCore+GraphRng> {
    __: T
}

impl<T: GraphCore+GraphRng> Solver for BFSTree<T> {
    type T = T;

    fn auto_parameters_solve(g: Self::T, ebc: Vec<f64>, dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree {
        multiple_greedy_bfs(&g, 10).1
    }
}



pub struct TestRandom;

impl Solver for TestRandom {
    type T = CompressedGraph;

    fn auto_parameters_solve(g: Self::T, _ebc: Vec<f64>, _dm: Vec<u32>, seed: u64, _time_limit: f64) -> RootedTree {
        g.random_subtree(&mut Prng::seed_from_u64(seed))
    }
}

// partionner

pub trait Partitioner {
    fn partitioner_label() -> &'static str;
    fn partition<T: GraphCore+GraphRng>(&mut self, g: &T) -> Vec<u64>;
    fn save_partition(&self, partition: &Vec<u64>, path_prefix: &str) {
        bincode::encode_into_std_write(
                partition, 
                &mut File::create(format!("{}-blocks-{}.data", path_prefix, Self::partitioner_label())).expect("welp"),
                bincode::config::standard()
        ).expect("welp2");
    }
    fn load_partition_or_compute_it<T: GraphCore+GraphRng>(&mut self, g: &T, path_prefix: &str, force_recompute: bool) -> Vec<u64> {
        if force_recompute {
            println!("info - recomputing partition");
            self.partition(g)
        } else {
            if let Ok(mut file) = File::open(format!("{}-blocks-{}.data", path_prefix, Self::partitioner_label())) {
                bincode::decode_from_std_read(
                    &mut file,
                    bincode::config::standard()
                ).expect("wee")
            } else {
                println!("info - computing partition because the file was not found");
                self.partition(g)
            }
        }
    }
}

pub struct LouvainPartitioner;
impl Partitioner for LouvainPartitioner {
    fn partitioner_label() -> &'static str {
        "louvain"
    }

    fn partition<T: GraphCore+GraphRng>(&mut self, g: &T) -> Vec<u64> {
        println!("launching {}", Self::partitioner_label());
        use pyo3::prelude::*;

        pyo3::prepare_freethreaded_python();
        let code = c_str!(include_str!("../graph_tool_test.py"));

        let k = g.vertex_count().isqrt();

        //let g = CompressedGraph::clique_cycle(50, 50);

        //let y = VNS::<CompressedGraph>::auto_parameters_solve(gdt, 121, 40.0);
        let mut blocks: Vec<u64> = vec![];
        Python::with_gil(|py| {
            let fun: Py<PyAny> = PyModule::from_code(
                py,
                code,
                c".\\..\\graph_tool_test.py",
                c"graph_tool_test",
            ).or_else(|err| {println!("{}", err.traceback(py).unwrap()); Err(err)})
            .unwrap()
            .getattr("find_communities").expect("rip2")
            .into();

            let kwargs = PyDict::new(py);
            kwargs.set_item("edges", g.get_edges()).expect("bah");
            kwargs.set_item("kmin", k / 2).expect("bah");
            kwargs.set_item("kmax", k * 2).expect("bah");

            let blocks_py = fun.call(py, (), Some(&kwargs)).expect("beuh");
            blocks = blocks_py.extract(py).expect("beuh");
            //println!("{:?}", blocks);
        });

        blocks

    }
}

pub struct MultiBfsPartitioner;

impl Partitioner for MultiBfsPartitioner {
    fn partitioner_label() -> &'static str {
        "multibfs"
    }

    fn partition<T: GraphCore+GraphRng>(&mut self, g: &T) -> Vec<u64> {
        assert!(g.is_connected());
        g.multisource_bfs_partition(g.vertex_count().isqrt(), &mut Prng::seed_from_u64(121))
    }
}



pub struct CommunitySolver<T: GraphCore+GraphRng+Default> {
    g: T,
    blocks: Vec<u64>,
    unique_block_count: usize,
    sub_graphs: Vec<T>,
    node_renumbering: Vec<HashMap<usize, usize>>,
    block_graph: T,
    block_graph_edges_hmap: HashMap<[usize; 2], Vec<[usize; 2]>>
}

pub fn renumber_edges(edges: &mut Vec<[usize; 2]>) -> HashMap<usize, usize> {
    // old vertex id -> new vertex id
    let mut hmap = HashMap::with_capacity(edges.len() * 2);
    let mut i = 0;
    for [u, v] in edges {
        hmap.entry(*u).or_insert_with(|| {let j = i; i += 1; j});
        hmap.entry(*v).or_insert_with(|| {let j = i; i += 1; j});

        *u = hmap[u];
        *v = hmap[v];
    }
    hmap
}

impl<'a, T: GraphCore+GraphRng+Default> CommunitySolver<T> {
    pub fn new(g: T, blocks: Vec<u64>, unique_block_count: usize) -> Self {
        Self { g, blocks, unique_block_count, 
            sub_graphs: Vec::with_capacity(unique_block_count),
            node_renumbering: Vec::with_capacity(unique_block_count),
            block_graph: Default::default(),
            block_graph_edges_hmap: Default::default() 
        }
    }

    pub fn init_block_graph(&mut self) {

        // premiere passe
        println!("first pass, unique blocks count={}", self.unique_block_count);
        let mut count = vec![0; self.unique_block_count];
        for b in self.blocks.iter() {
            count[*b as usize] += 1;
        }
        //println!("{:?}", count);
        let mut edges_vecvec = vec![vec![]; self.unique_block_count];

        for &[u, v] in self.g.get_edges().iter() {
            let [bu, bv] = [self.blocks[u] as usize, self.blocks[v] as usize].sorted();
            if bu == bv {
                edges_vecvec[self.blocks[u] as usize].push([u, v]);
            }
        }

        for edges in &mut edges_vecvec {
            let hmap = renumber_edges(edges);
            self.node_renumbering.push(hmap.inverse())
        }

        for (i, edges) in edges_vecvec.iter().enumerate() {
            let sg = T::from_edges(count[i], edges);
            //err += sg.vertex_count().abs_diff(562);
            //println!("sgn={}", sg.vertex_count());
            //sg.is_connected();
            let (cc, vis) = sg.bfs_connected_components();
            //println!("cc {}", cc);

            for (u_sg, &c) in vis.iter().enumerate() {
                //println!("i={} ec={} sgn={} {} {}", i, edges.len(),sg.vertex_count(), u_sg, c);
                if sg.vertex_count() > 1 {
                    let u = self.node_renumbering[i][&u_sg];
                    if c != 0 {
                        self.blocks[u] = self.unique_block_count as u64 + (c as u64 - 1);
                    }
                }

            }
            self.unique_block_count += cc as usize - 1;

        }



        // seconde passe

        println!("second pass, unique connected blocks count={}", self.unique_block_count);
        let mut count = vec![0; self.unique_block_count];
        for b in self.blocks.iter() {
            count[*b as usize] += 1;
        }
        //println!("{:?}", count);


        let mut edges_vecvec = vec![vec![]; self.unique_block_count];

        for &[u, v] in self.g.get_edges().iter() {
            let [bu, bv] = [self.blocks[u] as usize, self.blocks[v] as usize].sorted();
            if bu == bv {
                edges_vecvec[bu as usize].push([u, v]);
            } else {
                self.block_graph_edges_hmap
                    .entry([bu, bv])
                    .and_modify(|x| {x.push([u, v]);})
                    .or_insert(vec![[u, v]]);
            }
        }

        self.node_renumbering.clear();
        for edges in &mut edges_vecvec {
            let hmap = renumber_edges(edges);
            self.node_renumbering.push(hmap.inverse())
        }

        for (i, edges) in edges_vecvec.iter().enumerate() {
            let sg = T::from_edges(count[i], edges);
            assert!(sg.is_connected());
            self.sub_graphs.push(sg);
            //let (cc, vis) = sg.bfs_connected_components();
            //println!("cc {}", cc);

        }
        println!("edges to check {:?}", self.block_graph_edges_hmap.iter().map(|(k, v)| {v.len()}).sum::<usize>());
        self.block_graph = T::from_edges(self.unique_block_count, &self.block_graph_edges_hmap.iter().map(|tpl| {*tpl.0}).collect());
    }

    pub fn launch<Sbig: Solver<T=T>, Ssmall: Solver<T=T>>(&mut self, trace_save_path: Option<&str>) -> RootedTree {
        let mut ans_tree = self.g.clone_empty();
        let threshold = self.g.vertex_count() / self.unique_block_count / 5;
        let mut prng = Prng::seed_from_u64(1111);
        let mut community_trees = Vec::new();
        
        for (i, sg) in self.sub_graphs.drain(..).enumerate() {
            println!("solving for subgraph {}/{}, n={}", i + 1, self.unique_block_count, sg.vertex_count());
            let mut tree = if sg.vertex_count() > threshold {
                println!("using solver 1");
                Sbig::auto_parameters_solve(sg, vec![], vec![], 18268 + i as u64 * 21, 200.0)
            } else {
                println!("using solver 2 (threshold={})", threshold);
                Ssmall::auto_parameters_solve(sg, vec![], vec![], 18268 + i as u64 * 21, 1.0)
            };

            println!("disto obtained: {}", tree.new_disto_approx4());
            println!("");

            
            for [u, v] in tree.edges() {
                ans_tree.add_edge_unckecked(self.node_renumbering[i][&u], self.node_renumbering[i][&v]);
            }

            community_trees.push(tree);


        }
        println!("solving for block graph");
        let block_tree = Sbig::auto_parameters_solve(self.block_graph.clone(), vec![], vec![], 1212, 20.0);
        
        println!("tree reconstruction");
        for e in block_tree.edges() {
            let possible_edge_lst = &self.block_graph_edges_hmap[&e.sorted()];
            let l = possible_edge_lst.len();
            let i = (prng.next_u64() % l as u64) as usize;
            let [u, v] = possible_edge_lst[i];
            ans_tree.add_edge_unckecked(u, v);
        }
        
        let mut ans_rooted_tree = RootedTree::from_graph(&ans_tree, 0);
        if let Some(path) = trace_save_path {
            let trace = (ans_rooted_tree.new_disto_approx4(), &ans_rooted_tree, community_trees, block_tree);
            
            serde_json::to_writer_pretty(&mut File::create(path).expect("welp"), &trace).expect("welp");

        }

        ans_rooted_tree
    
    
    }



}







