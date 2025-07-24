use std::time::Instant;
use std::{collections::HashMap, fs::File, io::Write};





use crate::solver::aco2::ACO2;
use crate::solver::annealing::SA;
#[cfg(not(feature="louvain"))]
use crate::solver::community_solver::MultiBfsPartitioner;
use crate::solver::community_solver::{CommunitySolver, LouvainPartitioner, Partitioner};
use crate::solver::greedy::{greedy_bfs, greedy_ebc_delete_no_recompute};
use crate::solver::vns::VNS;
use crate::solver::{BFSTree, MultiBFSTree, Solver, VNSWithStart};
use pyo3::ffi::c_str;
use pyo3::types::PyDict;
use rand::{RngCore, SeedableRng};

use crate::graph::compressed_graph::CompressedGraph;
use crate::graph::graph_core::GraphCore;
use crate::trace::{TraceData, TraceResult};
use crate::utils::test_segment_tree;
use crate::my_rand::{random_permutation, Prng};
use crate::graph::graph_generator::{Data, GraphData, GraphRng};
use crate::graph::RootedTree;
use crate::graph::MatGraph;
use crate::config::Profile;
use crate::config::Config;
use crate::config::AntColonyProfile;

pub mod solver;
pub mod graph;
pub mod my_rand;
pub mod utils;
pub mod config;
pub mod neighborhood;
pub mod trace;
pub mod distorsion_heuristics;
pub mod counters;

pub fn test_on_graph(gdt: &GraphData, c: f64, evap: f64, seed: u64, _w: f64) {
    println!("n={}, m={}", gdt.n, gdt.m);
    let (g, ebc, dm) = gdt.graph_ebc_dist_matrix::<MatGraph>();
    assert!(g.is_connected());

    // let c = 100.0;
    // let evap = 0.01;
    let k = 10;
    let _ic = 600;


    let max_tau = 80.0;
    let min_tau = 0.2;
    let tau_init = 76.;


    let (dist1, _) = (0.0, ());//aco.launch(ic);

    // println!("{:?}", now.elapsed());

    // let now = Instant::now();

    let mut dist2_sum = 0.0;
    let mut counter = 0;

    let mut vecs = vec![];

    for i in 0..1 {
        println!("init aco2");

        // let (disto, _) = greedy_algo(&g, &dm);
        // println!("{}", disto);

        let mut _aco2 = ACO2::new(g.clone(), k, c, evap, min_tau, max_tau, tau_init, seed + 212*i, None, ebc.clone(),
        dm.clone());
            println!("launch aco2");

        let dist2 = 0.0;//aco2.launch(ic, w);
        dist2_sum += dist2;
        counter += 1;

        vecs.push(_aco2.trace)
    }


    // println!("{:?}", now.elapsed());

    let s = serde_json::to_string(&vecs).expect("welp");
    // println!("{:?}", g.get_edge_betweeness_centrality());
    println!("saving...");
    let mut output = File::create("trace.json").expect("welp2");
    output.write_fmt(core::format_args!("{}",s)).expect("weee");

    println!("aco1={}, aco2={}", dist1, dist2_sum / counter as f64);

}

pub fn save_result(gdt: &GraphData, label: &str, d: f64, trace: Vec<TraceData>) {
    let path = format!("{}_result-{}.json", gdt.label, label);
    println!("Done. Saving results in \"{}\"\n", &path);
    let mut file = File::create(&path).expect("bah");
    serde_json::to_writer_pretty(&mut file, &TraceResult::new(d, trace)).expect("error");

}

pub fn test_with_multiple_algos(i: u64, gdt: &GraphData) {
    let (g, ebc, dm) = gdt.graph_ebc_dist_matrix::<MatGraph>();

    let time_limit = if gdt.n > 500 {
        10. * 60.
    } else {
        60.0
    };

    println!("Sample <{}>", gdt.label);

    for mode in 0..0 {
        let label = format!("vns_mode{}", mode);

        println!("launching: <{}>", &label);
        let mut vns = VNS::new(g.clone(), 1234 + 34*i + mode as u64, ebc.clone(), dm.clone(), mode);
        let (d, t) = vns.gvns_random_start_nonapprox_timeout(time_limit);
        save_result(gdt, &label, d, t);
    }

    let label = "aco";

    println!("launching: <{}>", &label);

    let max_tau = 80.0;
    let min_tau = 0.2;
    let tau_init = 76.;

    let mut aco2 = ACO2::new(g.clone(), 10, 6000.0, 0.4, min_tau, max_tau, tau_init, 121 + 12*i, None, ebc.clone(), dm.clone());
    let d = aco2.launch(1000000, 0.5, time_limit, 2.0);
    save_result(gdt, &label, d.0, d.1);



    let label = "aco_hybrid";

    println!("launching: <{}>", &label);

    let max_tau = 80.0;
    let min_tau = 0.2;
    let tau_init = 76.;

    let mut aco2 = ACO2::new(g.clone(), 10, 6000.0, 0.4, min_tau, max_tau, tau_init, 121 + i, None, ebc.clone(), dm.clone());
    aco2.vnd_hybrid = true;
    let d = aco2.launch(1000000, 0.5, time_limit, 2.0);
    save_result(gdt, &label, d.0, d.1);


    let label = "beuh";

    println!("launching: <{}>", &label);

    let mut sa = SA::new(g.clone(), 1203 + 4*i, ebc.clone(), dm.clone());
    let d = sa.beuh(time_limit);
    save_result(gdt, &label, d.0, d.1);


    let label = "greedy";

    println!("launching: <{}>", &label);

    let ebc2 = if cfg!(not(feature = "need_ebc")) {
        println!("compute ebc for greedy..");
        &g.get_edge_betweeness_centrality()
    } else {
        &ebc
    };

    let d = greedy_ebc_delete_no_recompute(&g, ebc2, &dm);
    save_result(gdt, &label, d.0, vec![]);


}

#[cfg(not(feature="louvain"))]
type MyPartitioner = MultiBfsPartitioner;
#[cfg(feature="louvain")]
type MyPartitioner = LouvainPartitioner;

fn main() {
    test_segment_tree();
    let args: Vec<String> = std::env::args().collect();

    let mode = if let Some(_mode) = args.get(1) {
        _mode
    } else {
        "ac-c8000-evap0.4-w0.5-seed121"
    };

    if mode == "setup" {
        println!("setup");
        Data::generate_samples(1, 1000, 20000, 87876878).save("data/samples1000-20000-2.data");
        Data::generate_samples(1, 1000, 20000, 979).save("data/samples1000-20000-3.data");

        let dt = Data::load_benchmark_directory("./data/Graph Benchmark");
        dt.save("./binary_data/graph-benchmark-samples.data");

        let dt = Data::load_benchmark_directory("./data/social_network");
        dt.save("./binary_data/social-network-samples.data");

        // let dt = Data::load_benchmark_directory("./data/soc-LiveJournal1");
        // dt.save("./binary_data/soc-LiveJournal1.data");

        // let dt = Data::load_benchmark_directory("./data/soc-pokec-relationships");
        // dt.save("./binary_data/soc-pokec-relationships.data");

        // let dt = Data::load_benchmark_directory("./data/web-Google");
        // dt.save("./binary_data/web-Google.data");

        let dt = Data::load_benchmark_directory("./data/other-large-graphs");
        dt.save("./binary_data/other-large-graphs.data");

        let mut profiles: HashMap<String, Profile> = HashMap::new();
        profiles.insert("disto_approx".to_string(), Profile::DistoApprox);
        for seed in [121, 143] {
            profiles.insert(format!("ac-c8000-evap0.4-w0.5-seed{}", seed), Profile::AntColony(
                AntColonyProfile {c: 8000.0, evap: 0.4, seed, w: 0.5, k: 10, ic: 600}
            ));
        }

        profiles.insert("clique-cycle".to_string(), Profile::CliqueCycle);
        
        profiles.insert("benchmark".to_string(), Profile::VNSFullTest);
        profiles.insert("benchmark2".to_string(), Profile::Benchmark);

        profiles.insert("ntest1".to_string(), Profile::NeighborhoodTest);
        profiles.insert("new_dist_approx".to_string(), Profile::NewDistoApprox);
        profiles.insert("reg_graph".to_string(), Profile::RegularGraph);

        profiles.insert(format!("vns-vs-aco"), Profile::VNSvsACO(
            AntColonyProfile {c: 8000.0, evap: 0.4, seed: 123, w: 0.5, k: 10, ic: 600}
        ));

        profiles.insert("clustering_dblp".to_string(), Profile::ClusteringTest(0));
        profiles.insert("clustering_enron".to_string(), Profile::ClusteringTest(1));
        profiles.insert("clustering_facebook".to_string(), Profile::ClusteringTest(2));


        let cfg = Config {profiles};

        serde_json::to_writer_pretty(File::create("config.json").expect("beuh"), &cfg).expect("bouuh");
        
    } else {

        let cfg: Config = serde_json::from_reader(File::open("config.json").expect("wee")).expect("waa");
        
        if let Some(profile) = cfg.profiles.get(mode) {
            println!("% launching profile <{}>:", mode);

            match profile {
                Profile::ClusteringTest(gi) => {


                    let mut map = HashMap::new();

                    map.entry("poneyland").or_insert_with(|| "ahah");
                    map.entry("poneyland").or_insert_with(|| panic!());

                    assert_eq!(map["poneyland"], "ahah");


                    let now = Instant::now();
                    let _data = Data::load("data/social-network-samples.data");
                    let gdt = &_data.samples[*gi];
                    println!("{}", gdt.label);
                    println!("n={}, m={}", gdt.n, gdt.m);
                    let (g, ebc, dm) = gdt.graph_ebc_dist_matrix::<CompressedGraph>();
                    
                    let mut d = greedy_bfs(&g);
                    println!("greedy bfs result: {}", d.new_disto_approx4());
                    
                    
                    
                    println!("loading blocks...");
                    let mut partitioner = MyPartitioner {};
                    let mut blocks = partitioner.load_partition_or_compute_it(&g, &gdt.label, false);
                    partitioner.save_partition(&blocks, &gdt.label);
                    
                    let mut hmap = HashMap::new();
                    
                    let mut i = 0_u64;
                    println!("renumbering block id");
                    for b in blocks.iter() {
                        //print!("{} ", b);
                        hmap.entry(*b).or_insert_with(|| {let j=i; i += 1; j});
                    }
                    for b in blocks.iter_mut() {
                        let v = hmap[b];
                        *b = v;
                    }
                    //let i = 183;
                    println!("{}", i);
                    //println!("{:?}", &blocks[33000..33100]);

                    let mut solver: CommunitySolver<CompressedGraph> = CommunitySolver::new(g, blocks, i as usize);
                    solver.init_block_graph();



                    let mut tree = solver.launch::<BFSTree<CompressedGraph>, BFSTree<CompressedGraph>>(Some(&format!("{}-launch-result-{}.json", gdt.label, MyPartitioner::partitioner_label())));
                    println!("heuristic: {}", tree.new_disto_approx4());

                    println!("total execution time: {:?}", now.elapsed());

                },
                Profile::RegularGraph => {

                    let mut prng = Prng::seed_from_u64(12);
                    println!("loading samples...");
                    let data = Data::load("binary_data/graph-benchmark-samples.data");
                    let gdt = &data.samples[1];
                    println!("{}", gdt.label);
                    let (g, _, _) = gdt.graph_ebc_dist_matrix::<CompressedGraph>();
                    //let g = CompressedGraph::random_graph(100,800, &mut prng);
                    // let now = Instant::now();
                    // let t = greedy_bfs(&g);
                    // println!("{:?}, {}", now.elapsed(), t.0);
                    // let mut t = g.random_subtree(&mut prng);
                    // // let mut tree_buf = g.clone_empty();
                    // // let cpath = t.random_spider(7, &mut prng);
                    // // println!("{:?}", &cpath);  

                    // // t.to_graph(&g).to_dot("tree.dot");

                    // // t.subtree_vns_with_vertices(&mut prng, &cpath, &g, &mut tree_buf);

                    // // tree_buf.to_dot("tree2.dot");
                    // // std::process::Command::new("./gen_tree_png.bs").spawn().expect("bah");

                    // // starting_node_test_greedy_bfs(&g);
                    // let d = greedy_bfs(&g);
                    // println!("{}", d.0);

                    // let d = multiple_greedy_bfs(&g, 5);
                    // println!("{}", d.0);
                    let mut tbfs = MultiBFSTree::<CompressedGraph>::auto_parameters_solve(g.clone(), vec![], vec![], 112, 1.0);
                    let mut t1 = VNSWithStart::<CompressedGraph, MultiBFSTree<CompressedGraph>, 2, 0>::auto_parameters_solve(g.clone(), vec![], vec![], 134, 60.0);
                    // // //let t2 = VNS::<CompressedGraph>::auto_parameters_solve(g.clone(), vec![], vec![], 1234, 60.0);


                    println!("{:.2}%", t1.new_disto_approx4() as f64 / tbfs.new_disto_approx4() as f64 * 100.0);
                    
                    // let mut t1 = VNSWithStart::<CompressedGraph, MultiBFSTree<CompressedGraph>, 2, 1>::auto_parameters_solve(g.clone(), vec![], vec![], 134, 60.0);
                    // // // //let t2 = VNS::<CompressedGraph>::auto_parameters_solve(g.clone(), vec![], vec![], 1234, 60.0);


                    // println!("{:.2}%", t1.new_disto_approx4() as f64 / tbfs.new_disto_approx4() as f64 * 100.0)
                    
                    
                    //println!("vns: {}", t2.distorsion(&g, &dm))
                    // let mut sa = SA::new(g.clone(), 111, vec![], g.get_dist_matrix());
                    // sa.beuh(10.0);


                    // let data = Data::load("data/graph-benchmark-samples.data");

                    // for gdt in data.samples.iter() {
                    //     let (g, _,  dm) = gdt.graph_ebc_dist_matrix::<CompressedGraph>();
                    //     println!("{}", gdt.label);
                    //     let (d, t) = greedy_bfs(&g);
                    //     println!("{}", t.distorsion(&g, &dm))
                    // }


                },


                Profile::NewDistoApprox => {

                    use pyo3::prelude::*;
                    
                    pyo3::prepare_freethreaded_python();
                    let code = c_str!(include_str!("../graph_tool_test.py"));

                    let mut prng = Prng::seed_from_u64(1671);
                    let _data = Data::load("data/social-network-samples.data");
                    let gdt = &_data.samples[1];
                    println!("{}", gdt.label);
                    println!("n={}, m={}", gdt.n, gdt.m);
                    let (g, ebc, dm) = gdt.graph_ebc_dist_matrix::<CompressedGraph>();
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
                        kwargs.set_item("kmin", 300).expect("bah");
                        kwargs.set_item("kmax", 800).expect("bah");

                        let blocks_py = fun.call(py, (), Some(&kwargs)).expect("beuh");
                        blocks = blocks_py.extract(py).expect("beuh");
                        //println!("{:?}", blocks);
                    });

                    let mut f = File::create(format!("{}-blocks.json", gdt.label)).expect("a");
                    write!(f, "{}", serde_json::to_string(&blocks).unwrap()).unwrap();

                },

                Profile::Benchmark => {
                },
                Profile::AntColony(_dt) => {
                },
                Profile::DistoApprox => {
                },
                Profile::NeighborhoodTest => {
                },
                Profile::VNSvsACO(_dt) => {
                },
                Profile::VNSFullTest => {
                },

                Profile::CliqueCycle => {
                    println!("clique cycle");
                    let mut prng = Prng::seed_from_u64(123);
                    let k = 20;
                    let l = 60;

                    let permutation = random_permutation(k*l, &mut prng);
                    let g = MatGraph::clique_cycle(k, l).renumber(&permutation);
                    let tree = MatGraph::clique_cycle_mindisto_tree(k, l).renumber(&permutation);
                    let rooted_tree = RootedTree::from_graph(&tree, (prng.next_u64() % (k*l) as u64) as usize);

                    let dm = g.get_dist_matrix();
                    let ebc = g.get_edge_betweeness_centrality();

                    println!("disto: {}", rooted_tree.distorsion::<MatGraph>(&g, &dm));

                    let mut vns = VNS::new(g, 1203, ebc, dm, 0);
                    let d = vns.gvns_random_start_nonapprox_timeout(20.0);

                    println!("computed disto: {}", d.0);
                }
            }
        } else {
            println!("invalid profile, avalaible profiles are:");
            let mut keys: Vec<&String> = cfg.profiles.keys().collect();
            keys.sort();
            for k in keys {
                println!(" - <{}>", k);
            }
        }


    }



}


