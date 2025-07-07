use std::time::Instant;
use std::{collections::HashMap, fs::File, io::Write};




use rand::{RngCore, SeedableRng};

use crate::aco2::ACO2;
use crate::annealing::SA;
use crate::compressed_graph::CompressedGraph;
use crate::graph::graph_core::GraphCore;
use crate::trace::{TraceData, TraceResult};
use crate::utils::{test_segment_tree, TarjanSolver};
use crate::vns::VNS;
use crate::my_rand::{random_permutation, Prng};
use crate::greedy::greedy_ebc_delete_no_recompute;
use crate::graph::graph_generator::{Data, GraphData, GraphRng};
use crate::graph::print_counters;
use crate::graph::RootedTree;
use crate::graph::MatGraph;
use crate::config::Profile;
use crate::config::Config;
use crate::config::AntColonyProfile;

pub mod graph;
pub mod my_rand;
pub mod greedy;
pub mod aco2;
pub mod utils;
pub mod config;
pub mod neighborhood;
pub mod annealing;
pub mod trace;
pub mod distorsion_heuristics;
pub mod counters;
pub mod compressed_graph;
pub mod vns;

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
        dt.save("./data/graph-benchmark-samples.data");


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

        profiles.insert(format!("vns-vs-aco"), Profile::VNSvsACO(
            AntColonyProfile {c: 8000.0, evap: 0.4, seed: 123, w: 0.5, k: 10, ic: 600}
        ));

        profiles.insert("clustering_test".to_string(), Profile::ClusteringTest);


        let cfg = Config {profiles};

        serde_json::to_writer_pretty(File::create("config.json").expect("beuh"), &cfg).expect("bouuh");
        
    } else {

        let cfg: Config = serde_json::from_reader(File::open("config.json").expect("wee")).expect("waa");
        
        if let Some(profile) = cfg.profiles.get(mode) {
            println!("% launching profile <{}>:", mode);

            match profile {
                Profile::ClusteringTest => {
                    let mut prng = Prng::seed_from_u64(1671);
                    let g = CompressedGraph::random_graph(100000, 2000000, &mut prng);
                    let d = g.clustering();
                    println!("{:?}", d.1.len());
                    println!("{}", d.1.iter().map(|x|{x*x}).sum::<usize>())

                },
                Profile::NewDistoApprox => {

                    println!("loading samples...");
                    let data = Data::load("data/graph-benchmark-samples.data");
                    let mut prng = Prng::seed_from_u64(1671);

                    
                    let g = MatGraph::random_graph(1000, 10000, &mut prng);
                    let dm = g.get_dist_matrix();

                    let mut dists = [0_u32; 10];
                    let mut d_mean = 0.0;
                    let mut d_max = 0;
                    for v in dm.iter() {
                        dists[*v as usize] += 1;
                    }

                    for v in 0..g.n {
                        let deg = g.get_neighboor_count_unchecked(v as usize);
                        d_max = d_max.max(deg);
                        d_mean += deg as f64 / g.n as f64;
                    }

                    println!("{} {}", d_max, d_mean);
                    println!("{:?}", dists);


                    let s: u32 = dm.iter().sum();
                    let m = s as f64 / g.n as f64 / (g.n - 1) as f64;
                    println!("{}", m);

                    let mut rm = 0.0;
                    let mut rmax: f64 = 0.0;
                    let p = 0.37480394605905987;
                    for _ in 0..0 {
                        let t = g.random_subtree(&mut prng);
                        let dhu = t.new_disto_approx();
                        let d = t.distorsion::<MatGraph>(&g, &dm);

                        let dh = dhu as f64 / (g.n as f64 * (g.n - 1) as f64) / m * 2.0;// * 1.0537940243214428; 
                        
                        let adh2 = dhu as f64 / (g.n as f64 * (g.n - 1) as f64) * 2.0;
                        let dh2 = adh2 / 3.0 + adh2 *p / (2.0 * 3.0);
                        
                        let rh = d / dh2;
                        println!("{} {} {}", rh, dh, d);
                        rm += dh2 / d;
                        rmax = rmax.max((d - dh).abs() / d); 
                    }

                    println!("{}% {}%" , rm / 10.0 * 100.0, rmax * 100.0);
                    


                    let gdt = &data.samples[6];
                    let (g, ebc, dm) = gdt.graph_ebc_dist_matrix::<CompressedGraph>();
                    println!("label={}", gdt.label);

                    // let t = g.random_subtree(&mut prng);
                    // println!("{}", t.new_disto_approx());
                    // println!("{}", t.s22_slow());
                    // {
                    //     let mut prng = Prng::seed_from_u64(1234);
                    //     println!("generating graph");
                    //     let g = CompressedGraph::random_graph(1000000, 50000000, &mut prng);
                    //     let ebc = vec![]; //g.get_edge_betweeness_centrality();
                    //     println!("ts");

                    //     let mut ts = TarjanSolver::new(g.n, &g);
                    //     println!("m {}", g.get_edges().len());
                    //     //let dm = g.get_dist_matrix();
                        
                    //     for _ in 0..10 {
            
                    //         let t = g.random_subtree(&mut prng);
                    //         println!("computing heuristic");
                    //         let now = Instant::now();
                    //         let heur = t.heuristic(&g, &vec![], &mut ts, &ebc, &vec![]);
                    //         println!("heuristique: {}, elapsed: {:?}", heur, now.elapsed());
                    //         //println!("disto: {}", t.distorsion(&g, &dm));
                    //     }
                    // }

                    let mut vns = VNS::new(g, 1239, ebc, dm, 2);
                    let d = vns.gvns_random_start_nonapprox_timeout(20.0);
                    
                    println!("{}", d.0);
                    counters::print_counters();
                    //g.to_dot("tree.dot");
                    //std::process::Command::new(".\\gen_tree_png.cmd").spawn().expect("bah");

                    //t.to_graph().to_dot("tree2.dot");
                    //std::process::Command::new(".\\gen_tree_png.cmd").spawn().expect("bah");
                },

                Profile::Benchmark => {



                    println!("loading samples...");
                    let data = Data::load("data/graph-benchmark-samples.data");

                    let gdt = &data.samples[6];
                    let mut per_distances = [0; 10];

                    let (_g, _ebc, dm) = gdt.graph_ebc_dist_matrix::<MatGraph>();

                    println!("{}", &gdt.label);
                    println!("max dist: {}", dm.iter().max().unwrap());
                    println!("mean dist: {}", dm.iter().map(|x| {*x as f64}).sum::<f64>() / dm.len() as f64);
                    dm.iter().for_each(|x| {per_distances[*x as usize] += 1});
                    println!("{:?}", per_distances);
                    test_with_multiple_algos(5 as u64, &data.samples[5]);

                    
                    println!("Launching tests on random graphs:");

                    for (i, gdt) in data.samples.iter().enumerate() {
                        println!("{}/{}", i+1, data.n_samples);
                        test_with_multiple_algos(i as u64, gdt);
                        println!("\n\n");
                    }

                    println!("Launching tests on clique cycles:");
                    let mut prng = Prng::seed_from_u64(8989898);
                    for (k, l) in [(30, 30), (10, 100), (100, 10)] {
                        let n = k * l;
                        let perm = random_permutation(n, &mut prng);
                        let g = MatGraph::clique_cycle(k, l).renumber(&perm);
                        let t = MatGraph::clique_cycle_mindisto_tree(k, l).renumber(&perm);
                        let trooted = RootedTree::from_graph(&t, 0);

                        let mut gdt = GraphData::from_graph(&g, false, false);
                        let dm = g.get_dist_matrix();
                        let d = trooted.distorsion::<MatGraph>(&g, &dm);

                        gdt.label = format!("data/clique_cycle{}-{}", k, l);

                        save_result(&gdt, "exact", d, vec![]);

                        test_with_multiple_algos(k as u64, &gdt);  
                    }


                },

                Profile::AntColony(dt) => {
                    println!("loading samples...");
                    let data = Data::load("data/samples1000-20000.data");
                    println!("launching test.");
                    test_on_graph(&data.samples[0], dt.c,dt.evap, dt.seed, dt.w);
                },
                Profile::DistoApprox => {

                    println!("loading samples...");
                    let data = Data::load("data/samples1000-20000-3.data");
                    println!("launching test.");
                    let mut _cool = 0;
                    let mut _pas_cool = 0;
                    let gdt = &data.samples[0];

                    println!("n={}, m={}", gdt.n, gdt.m);
                    let (g, ebc, dm) = gdt.graph_ebc_dist_matrix::<MatGraph>();
                    assert!(g.is_connected());

                    let mut prng = Prng::seed_from_u64(987);
                    let edges = &g.get_edges();
                    let tarjan_solver= &mut TarjanSolver::new(g.n, &g);
                    
                    for _ in 0..1000 {
                        let t1 = g.random_subtree(&mut prng);
                        let t2 = g.random_subtree(&mut prng);
                        let da1 = t1.heuristic(&g, edges, tarjan_solver, &ebc, &vec![]);
                        let da2 = t2.heuristic(&g, edges, tarjan_solver, &ebc, &vec![]);

                        let d1 = t1.distorsion::<MatGraph>(&g, &dm);
                        let d2 = t2.distorsion::<MatGraph>(&g, &dm);

                        if (da1 < da2 && d1 > d2) || (da1 > da2 && d1 < d2) {
                            _pas_cool += 1;
                        } else {
                            _cool += 1;
                        }

                        println!("err={:.2}%   ({}/{})", 100.0 * _pas_cool as f64 / (_cool + _pas_cool) as f64, _pas_cool, _pas_cool + _cool);
                    }

                },

                Profile::NeighborhoodTest => {
                    let mut prng = Prng::seed_from_u64(123);
                    let g = MatGraph::random_graph(15, 80, &mut prng);

                    g.to_dot("graph.dot");

                    let mut t = g.random_subtree(&mut prng);
                    t.update_parents();
                    t.to_graph(&g).to_dot("tree.dot");

                    // while !t.edge_swap_random(&mut prng, &g.get_edges()) {};

                    // t.to_graph().to_dot("tree2.dot");

                    let mut tbuf = MatGraph::new_empty(g.n);
                    // while !t.subtree_swap_with_random_edge(&mut prng, &g.get_edges(), &g, &mut tbuf) {}

                    t.subtree_swap_with_random_critical_path(&mut prng, &g, &mut tbuf);
                    tbuf.to_dot("tree2.dot");

                    std::process::Command::new(".\\gen_tree_png.cmd").spawn().expect("bah");
                },

                Profile::VNSvsACO(_dt) => {
                    println!("loading samples...");
                    let data = Data::load("data/samples1000-20000-2.data");
                    let gdt = &data.samples[0];

                    println!("launching test: VNS");
                    let (g, ebc, dm) = gdt.graph_ebc_dist_matrix();

                    let mut vns: VNS<MatGraph> = VNS::new(g, 1203, ebc, dm, 0);
                    let d = vns.gvns_random_start_nonapprox_timeout(60.0);

                    println!("vns result: {}", d.0);


                    println!("launching test: Annealing");
                    let (g, ebc, dm) = gdt.graph_ebc_dist_matrix();

                    let mut sa: SA<MatGraph> = SA::new(g, 1203, ebc, dm);
                    let d = sa.beuh(60.0);
                    println!("{}", d.0);

                    println!("launching test: ACO");
                    //test_on_graph(&data.samples[0], dt.c,dt.evap, dt.seed, dt.w);
                    let (g, ebc, dm) = gdt.graph_ebc_dist_matrix();

                    let max_tau = 80.0;
                    let min_tau = 0.2;
                    let tau_init = 76.;

                    let mut aco2: ACO2<MatGraph> = ACO2::new(g, 10, _dt.c, _dt.evap, min_tau, max_tau, tau_init, 121, None, ebc, dm);
                    aco2.vnd_hybrid = true;
                    let d = aco2.launch(1000000, 0.5, 6.0, 2.0);
                    println!("d: {}", d.0);

                    print_counters();

                },
                Profile::VNSFullTest => {

                    println!("loading samples...");
                    let data = Data::load("data/graph-benchmark-samples.data");
                    //test_on_graph(&data.samples[0], dt.c,dt.evap, dt.seed, dt.w);

                    let gdt = &data.samples[0];
                    let (g, ebc, dm) = gdt.graph_ebc_dist_matrix();


                    let (disto, _) = greedy_ebc_delete_no_recompute(&g, &ebc, &dm);
                    println!("greedy: {}", disto);

                    // let gdt = &data.samples[15];
                    // println!("sample name: <{}>", gdt.label);
                    // let (g, ebc, dm) = gdt.graph_ebc_dist_matrix();

                    // let mut vns = VNS::new(g, 1203, ebc, dm);
                    // let _d = vns.gvns_random_start_nonapprox_timeout(1.0);

                    println!("launching test: VNS");

                    for gdt in data.samples.iter() {
                        println!("sample name: <{}>", gdt.label);
                        let (g, ebc, dm) = gdt.graph_ebc_dist_matrix();

                        let mut vns: VNS<MatGraph> = VNS::new(g, 1203, ebc, dm, 0);
                        let d = vns.gvns_random_start_nonapprox_timeout(20.0);

                        println!("disto={}", d.0);

                        println!("saving..");
                        let mut file = File::create(format!("{}_result.json", gdt.label)).expect("bah");

                        serde_json::to_writer_pretty(&mut file, &d.1).expect("error");

                        println!("\n");
                    }



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


