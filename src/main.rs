use std::{collections::HashMap, fs::File};




use crate::graph::graph_serde::{Data, GraphData};
use crate::solver::greedy::multiple_greedy_bfs;
use crate::solver::{MultiBFSTree, Solver, VNSWithStart};
use rand::SeedableRng;

use crate::graph::compressed_graph::CompressedGraph;
use crate::utils::test_segment_tree;
use crate::my_rand::Prng;
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


pub fn save_result(gdt: &GraphData, label: &str, trace: (f64, u64, u64)) {
    let path = format!("{}_result-{}.json", gdt.label, label);
    println!("Done. Saving results in \"{}\"\n", &path);
    let mut file = File::create(&path).expect("bah");
    serde_json::to_writer_pretty(&mut file, &trace).expect("error");

}
 

const LARGE_GRAPH_DATASET: &[&str] = &[
    "binary_data/other-large-graphs.data",
    "binary_data/soc-LiveJournal1.data",
    "binary_data/soc-pokec-relationships.data",
    "binary_data/web-Google.data"
];

const LARGE_GRAPH_RAWDATA_DIRS: &[&str] = &[
    "data/other-large-graphs",
    "data/soc-LiveJournal1",
    "data/soc-pokec-relationships",
    "data/web-Google"
];

const SMALL_GRAPH_DATASET: &[&str] = &[
    "binary_data/graph-benchmark-samples.data"
];

const SMALL_GRAPH_RAWDATA_DIRS: &[&str] = &[
    "data/Graph Benchmark"
];



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

        for (&rawdata_dir, &bin_path) in SMALL_GRAPH_RAWDATA_DIRS.iter().zip(SMALL_GRAPH_DATASET) {
            let dt = Data::load_benchmark_directory(rawdata_dir);
            dt.save(bin_path);
        }


        let dt = Data::load_benchmark_directory("./data/social_network");
        dt.save("./binary_data/social-network-samples.data");

        for (&rawdata_dir, &bin_path) in LARGE_GRAPH_RAWDATA_DIRS.iter().zip(LARGE_GRAPH_DATASET) {
            let dt = Data::load_benchmark_directory(rawdata_dir);
            dt.save(bin_path);
        }

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
                Profile::ClusteringTest(_gi) => {
                },
                Profile::RegularGraph => {

                    let mut _prng = Prng::seed_from_u64(12);
                    println!("loading samples...");
                    let data = Data::load("binary_data/graph-benchmark-samples.data");
                    let gdt = &data.samples[0];
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
                    let mut tbfs = MultiBFSTree::<50>::auto_parameters_solve(g.clone(), vec![], vec![], 112, 1.0);
                    let mut t1 = VNSWithStart::<CompressedGraph, MultiBFSTree, 2, 0>::auto_parameters_solve(g.clone(), vec![], vec![], 134, 10.0);
                    // // //let t2 = VNS::<CompressedGraph>::auto_parameters_solve(g.clone(), vec![], vec![], 1234, 60.0);


                    println!("{:.2}%", t1.distance_sum() as f64 / tbfs.distance_sum() as f64 * 100.0);
                    
                    // let mut t1 = VNSWithStart::<CompressedGraph, MultiBFSTree<CompressedGraph>, 2, 1>::auto_parameters_solve(g.clone(), vec![], vec![], 134, 60.0);
                    // // // //let t2 = VNS::<CompressedGraph>::auto_parameters_solve(g.clone(), vec![], vec![], 1234, 60.0);


                    // println!("{:.2}%", t1.distance_sum() as f64 / tbfs.distance_sum() as f64 * 100.0)
                    
                    
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

                },

                Profile::Benchmark => {
                    for &bin_path in LARGE_GRAPH_DATASET {
                        let dt = Data::load(bin_path);
                        for gdt in dt.samples {
                            println!("{}", gdt.label);
                            let (g, _, _) = gdt.graph_ebc_dist_matrix::<CompressedGraph>();
                            let (bfsdist, _tbfs) = multiple_greedy_bfs(&g, 50);

                            {
                                let mut tvns1 = VNSWithStart::<CompressedGraph, MultiBFSTree, 2, 0>::auto_solve_no_ebcdm(g.clone(), 1234, 3600.0);
                                let vns1dist = tvns1.distance_sum();
                                save_result(&gdt, "super-vns1", (vns1dist as f64 / bfsdist as f64, vns1dist, bfsdist));

                            }{
                                let mut tvns2= VNSWithStart::<CompressedGraph, MultiBFSTree, 0, 0>::auto_solve_no_ebcdm(g.clone(), 1234, 3600.0);
                                let vns2dist = tvns2.distance_sum();
                                save_result(&gdt, "super-vns2", (vns2dist as f64 / bfsdist as f64, vns2dist, bfsdist));

                            }
                        }
                    }

                    for &bin_path in SMALL_GRAPH_DATASET {
                        let dt = Data::load(bin_path);
                        for gdt in dt.samples {
                            println!("{}", gdt.label);
                            let (g, _, _) = gdt.graph_ebc_dist_matrix::<CompressedGraph>();
                            let (bfsdist, _tbfs) = multiple_greedy_bfs(&g, 50);

                            {
                                let mut tvns1 = VNSWithStart::<CompressedGraph, MultiBFSTree, 2, 0>::auto_solve_no_ebcdm(g.clone(), 1234, 600.0);
                                let vns1dist = tvns1.distance_sum();
                                save_result(&gdt, "super-vns1", (vns1dist as f64 / bfsdist as f64, vns1dist, bfsdist));

                            }{
                                let mut tvns2= VNSWithStart::<CompressedGraph, MultiBFSTree, 0, 1>::auto_solve_no_ebcdm(g.clone(), 1234, 600.0);
                                let vns2dist = tvns2.distance_sum();
                                save_result(&gdt, "super-vns2", (vns2dist as f64 / bfsdist as f64, vns2dist, bfsdist));

                            }
                        }
                    }




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


