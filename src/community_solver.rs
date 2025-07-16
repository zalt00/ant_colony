use std::{collections::HashMap, hash::Hash, time::Instant};

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::{graph::{compressed_graph::CompressedGraph, graph_core::GraphCore, graph_generator::{GraphData, GraphRng}, RootedTree}, my_rand::Prng, utils::{HashMapExt, PairExt}, vns::VNS};
use crate::utils::PairIterExt;

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

pub struct BestRandom;

impl Solver for BestRandom {
    type T = CompressedGraph;
    fn auto_parameters_solve(g: Self::T, ebc: Vec<f64>, dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree {
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

pub struct TestRandom;

impl Solver for TestRandom {
    type T = CompressedGraph;

    fn auto_parameters_solve(g: Self::T, ebc: Vec<f64>, dm: Vec<u32>, seed: u64, time_limit: f64) -> RootedTree {
        g.random_subtree(&mut Prng::seed_from_u64(seed))
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

            //sg.is_connected();
            let (cc, vis) = sg.bfs_connected_components();
            //println!("cc {}", cc);

            for (u_sg, &c) in vis.iter().enumerate() {
                let u = self.node_renumbering[i][&u_sg];
                if c != 0 {
                    self.blocks[u] = self.unique_block_count as u64 + (c as u64 - 1);
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

        self.block_graph = T::from_edges(self.unique_block_count, &self.block_graph_edges_hmap.iter().map(|tpl| {*tpl.0}).collect());
    }

    pub fn launch<Sbig: Solver<T=T>, Ssmall: Solver<T=T>>(&mut self) -> RootedTree {
        let mut ans_tree = self.g.clone_empty();

        
        for (i, sg) in self.sub_graphs.drain(..).enumerate() {
            println!("solving for subgraph {}/{}, n={}", i + 1, self.unique_block_count, sg.vertex_count());
            let mut tree = if sg.vertex_count() > 100 {
                println!("using solver 1");
                Sbig::auto_parameters_solve(sg, vec![], vec![], 18268 + i as u64 * 21, 200.0)
            } else {
                println!("using solver 2");
                Ssmall::auto_parameters_solve(sg, vec![], vec![], 18268 + i as u64 * 21, 1.0)
            };

            println!("disto obtained: {}", tree.new_disto_approx4());
            println!("");

            
            for [u, v] in tree.edges() {
                ans_tree.add_edge_unckecked(self.node_renumbering[i][&u], self.node_renumbering[i][&v]);
            }

        }
        println!("solving for block graph");
        let block_tree = Sbig::auto_parameters_solve(self.block_graph.clone(), vec![], vec![], 1212, 20.0);
        
        println!("tree reconstruction");
        for e in block_tree.edges() {
            let possible_edge_lst = &self.block_graph_edges_hmap[&e.sorted()];
            let [u, v] = possible_edge_lst[0];
            ans_tree.add_edge_unckecked(u, v);
        }
        
        RootedTree::from_graph(&ans_tree, 0)
    
    
    }



}







