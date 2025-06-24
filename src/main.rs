use std::{fs::File, io::{Read, Write}, time::Instant};



use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::{aco1::ACO, aco2::{test_segment_tree, ACO2}, graph::Graph, greedy::{greedy_algo, greedy_degree_bfs}, random_graph_generator::{Data, GraphData}, utils::Par};

pub mod graph;
pub mod my_rand;
pub mod greedy;
pub mod aco1;
pub mod aco2;
pub mod utils;
pub mod random_graph_generator;


pub fn test_on_facebook(c: f64, evap: f64, seed: u64) {
    let mut s = String::new();

    let mut file = File::open("data/facebook_converted.graph").expect("error");
    let mut v = vec![];
    file.read_to_string(&mut s);



    for (i, gs) in s.split("::").enumerate() {
        let gstring = gs.trim().to_string();
        if gstring.len() > 0 {

            let (disto, g) = Graph::from_string(gstring).expect("welp");

            // println!("serializing...");
            // let s = serde_json::to_string(&g.get_dist_matrix()).expect("welp");
            // // println!("{:?}", g.get_edge_betweeness_centrality());
            // println!("saving...");
            // let mut output = File::create("data/facebook_distmat.json").expect("welp2");
            // write!(output, "{}", s);

            // println!("done.");

            v.push(g);

            //if i == 0 {break}
        }
    }
    let mut s2 = String::new();
    let mut file2 = File::open("data/facebook_ebc.json").expect("welp");
    file2.read_to_string(&mut s2).expect("weeelp");
    let ebc: Vec<f64> = serde_json::from_str(&s2).expect("wlpe2");

    let mut s2 = String::new();
    let mut file2 = File::open("data/facebook_distmat.json").expect("welp");
    file2.read_to_string(&mut s2).expect("weeelp");
    let dm: Vec<u32> = serde_json::from_str(&s2).expect("wlpe2");

    // let c = 100.0;
    // let evap = 0.01;
    let k = 10;
    let ic = 100;


    let max_tau = 16.0;
    let min_tau = 0.2;
    let tau_init = 15.0;

    let g = &v[0];

    // println!("launch greedy1");
    // let (disto, _) = greedy_algo(g, &dm);
    // println!("{}", disto);


    // println!("launch greedy2");
    // let mut prng = Xoshiro256PlusPlus::seed_from_u64(189);
    // let (disto, _) = greedy_degree_bfs(g, &mut prng, &dm);
    // println!("{}", disto);


    // let now = Instant::now();
    // println!("init aco1");
    // let mut aco = ACO::new(g.clone(), k,
    //  Par::new_free(0.0), Par::new_free(0.0), Par::new_free(c),
    //   Par::new_free(evap), Par::new_free(max_tau), Par::new_free(15.8), ebc.clone(),
    // dm.clone());
    // println!("launch aco1");

    let (dist1, _) = (0.0, ());//aco.launch(ic);

    // println!("{:?}", now.elapsed());

    // let now = Instant::now();

    let mut dist2_sum = 0.0;
    let mut counter = 0;

    let mut vecs = vec![];

    for i in 0..5 {
        println!("init aco2");

        let mut aco2 = ACO2::new(g.clone(), k, c, evap, min_tau, max_tau, tau_init, seed + 212*i, None, ebc.clone(),
        dm.clone());
            println!("launch aco2");

        let dist2 = aco2.launch(ic, 1.0);
        dist2_sum += dist2;
        counter += 1;

        vecs.push(aco2.trace)
    }


    // println!("{:?}", now.elapsed());

    let s = serde_json::to_string(&vecs).expect("welp");
    // println!("{:?}", g.get_edge_betweeness_centrality());
    println!("saving...");
    let mut output = File::create("trace.json").expect("welp2");
    output.write_fmt(core::format_args!("{}",s)).expect("weee");

    println!("aco1={}, aco2={}", dist1, dist2_sum / counter as f64);

}



pub fn test_on_graph(gdt: &GraphData, c: f64, evap: f64, seed: u64, w: f64) {
    println!("n={}, m={}", gdt.n, gdt.m);
    let g = gdt.to_graph();
    assert!(g.is_connected());
    let ebc = gdt.ebc.as_ref().unwrap();
    let dm = gdt.dist_matrix.as_ref().unwrap();

    // let c = 100.0;
    // let evap = 0.01;
    let k = 10;
    let ic = 600;


    let max_tau = 80.0;
    let min_tau = 0.2;
    let tau_init = 76.;

    // println!("launch greedy1");
    // let (disto, _) = greedy_algo(g, &dm);
    // println!("{}", disto);


    // println!("launch greedy2");
    // let mut prng = Xoshiro256PlusPlus::seed_from_u64(189);
    // let (disto, _) = greedy_degree_bfs(g, &mut prng, &dm);
    // println!("{}", disto);


    // let now = Instant::now();
    // println!("init aco1");
    // let mut aco = ACO::new(g.clone(), k,
    //  Par::new_free(0.0), Par::new_free(0.0), Par::new_free(c),
    //   Par::new_free(evap), Par::new_free(max_tau), Par::new_free(15.8), ebc.clone(),
    // dm.clone());
    // println!("launch aco1");

    let (dist1, _) = (0.0, ());//aco.launch(ic);

    // println!("{:?}", now.elapsed());

    // let now = Instant::now();

    let mut dist2_sum = 0.0;
    let mut counter = 0;

    let mut vecs = vec![];

    for i in 0..2 {
        println!("init aco2");

        // let (disto, _) = greedy_algo(&g, &dm);
        // println!("{}", disto);

        let mut aco2 = ACO2::new(g.clone(), k, c, evap, min_tau, max_tau, tau_init, seed + 212*i, None, ebc.clone(),
        dm.clone());
            println!("launch aco2");

        let dist2 = aco2.launch(ic, w);
        dist2_sum += dist2;
        counter += 1;

        vecs.push(aco2.trace)
    }


    // println!("{:?}", now.elapsed());

    let s = serde_json::to_string(&vecs).expect("welp");
    // println!("{:?}", g.get_edge_betweeness_centrality());
    println!("saving...");
    let mut output = File::create("trace.json").expect("welp2");
    output.write_fmt(core::format_args!("{}",s)).expect("weee");

    println!("aco1={}, aco2={}", dist1, dist2_sum / counter as f64);

}




pub fn get_graphs() -> Vec<Graph> {
    let mut s = String::new();
    let mut file = File::open("data/facebook_converted.graph").expect("error");
    let mut v = vec![];
    file.read_to_string(&mut s);

    let coef_s = 0.0;
    let coef_c = 0;

    for (i, gs) in s.split("::").enumerate() {
        let gstring = gs.trim().to_string();
        if gstring.len() > 0 {

            let (disto, g) = Graph::from_string(gstring).expect("welp");

            println!("serializing...");
            let s = serde_json::to_string(&g.get_edge_betweeness_centrality()).expect("welp");
            // println!("{:?}", g.get_edge_betweeness_centrality());
            println!("saving...");
            let mut output = File::create("data/facebook_ebc.json").expect("welp2");
            output.write_fmt(core::format_args!("{}",s)).expect("weee");

            println!("done.");

            v.push(g);

            //if i == 0 {break}
        }
    }

    v
}

// pub fn evaluate_score(grs: &Vec<Graph>, i: usize, j: usize, k: usize,
//     alpha: Par<f64>, beta: Par<f64>, c: Par<f64>, evap: Par<f64>, max_tau: Par<f64>,
//     interval_tau: Par<f64>, iter_count: usize) -> f64 {
    
//     unsafe {CHEPA = 0.0};

//     let mut s = 0.0;
//     let mut sg = 0.0;
//     let mut sgs = 0.0;
//     for (o, g) in grs[i..j].iter().enumerate() {


//         let g2 = g.clone();
//         let (disto2, tgreedy) = greedy_algo(&g);

//         let bt2 = Some(RootedTree::from_graph(&tgreedy, 0));

//         let mut aco = ACO::new(g2, k, alpha, beta, c, evap, max_tau, interval_tau);
//         aco.base_dist = disto2;
//         aco.base_tree = tgreedy;
//         //let disto = 0.0;
//         //sleep(Duration::from_secs(1));
//         let (disto, t) = aco.launch(iter_count);

//         let mut aco2 = ACO2::new(g.clone(), k, c.val / 20.0, evap.val,
//          max_tau.val - interval_tau.val, max_tau.val, max_tau.val, 11212, bt2 );

//         let disto3 = aco2.launch(iter_count);
//         // let mut aco_dummy = ACO::new_dummy(g.clone());
//         // let disto3 = aco_dummy.greedy_stretch();
//         //println!("{:?}  {} {}", aco.get_tau_tab_info(), disto, disto2);

        

//         //println!("{:?}", aco.tau_matrix);
//         //println!("{:?}", t.get_edges());

//         s += disto;
//         sg += disto3;
//         // sgs += disto3;
//         //sleep(Duration::from_secs(2));
//     }
//     println!("{};{};{};{}", unsafe{graph::CHEPA}, s / (j-i) as f64, s / sg, sg / (j-i) as f64);

//     s / (j-i) as f64
// }




// pub fn test_on_graphs(grs: &Vec<Graph>, i: usize, j: usize) {
//     let c = Par { val: 100.0, bounds: [0.1, 1000.0], std: 0.0 };
//     let evap = Par { val: 0.01, bounds: [0.0, 0.5], std: 0.0 };
//     let k = 20;
//     let ic = 30* 40 * 20 / 10;

//     let alpha = Par { val: 1.2, bounds: [0.0, 5.0], std: 0.0 };
//     let beta = Par { val: 1.5, bounds: [0.0, 5.0], std: 0.0 };

//     let max_tau = Par { val: 16.0, bounds: [0.5, 1000.0], std: 0.0 };
//     let interval_tau = Par { val: 15.8, bounds: [0.5, 1000.0], std: 0.0 };


//     let current_best_score = evaluate_score(grs, i, j,
//         k, alpha, beta, c, evap,
//     max_tau, interval_tau, ic);
//     return;


// }

// pub fn evaluate_score2(grs: &Vec<Graph>, i: usize, j: usize, k: usize,
//     alpha: Par<f64>, beta: Par<f64>, c: Par<f64>, evap: Par<f64>, max_tau: Par<f64>,
//     interval_tau: Par<f64>, iter_count: usize) -> f64 {
    
//     unsafe {CHEPA = 0.0};

//     let mut s = 0.0;
//     let mut sg = 0.0;
//     let mut sgs = 0.0;
//     for (o, g) in grs[i..j].iter().enumerate() {


//         let g2 = g.clone();
//         // let (disto2, tgreedy) = greedy_algo(&g);
//         println!("greedy done");

//         let bt2 = None; //RootedTree::from_graph(&tgreedy, 0);

//         let now = Instant::now();

//         let mut aco = ACO::new(g2, k, alpha, beta, c, evap, max_tau, interval_tau);
//         //aco.base_dist = disto2;
//         //aco.base_tree = tgreedy;
//         //let disto = 0.0;
//         //sleep(Duration::from_secs(1));
//         let (disto, t) = aco.launch(iter_count);

//         println!("aco 1 done {:?}", now.elapsed());

//         let now = Instant::now();

//         let mut aco2 = ACO2::new(g.clone(), k, c.val / 20.0, evap.val,
//          max_tau.val - interval_tau.val, max_tau.val, max_tau.val, 11212, bt2 );

//         let disto3 = aco2.launch(iter_count);

//         println!("aco 2 done {:?}", now.elapsed());
//         // let mut aco_dummy = ACO::new_dummy(g.clone());
//         // let disto3 = aco_dummy.greedy_stretch();
//         //println!("{:?}  {} {}", aco.get_tau_tab_info(), disto, disto2);

        

//         //println!("{:?}", aco.tau_matrix);
//         //println!("{:?}", t.get_edges());

//         s += disto;
//         sg += 0.0; //disto2;
//         sgs += disto3;
//         //sleep(Duration::from_secs(2));
//     }
//     println!("{};{};{};{}", unsafe{graph::CHEPA}, s / (j-i) as f64, sgs / s, sgs / (j-i) as f64);
    


//     s / (j-i) as f64
// }




// pub fn test_on_graphs2(grs: &Vec<Graph>, i: usize, j: usize) {
//     let c = Par { val: 100.0, bounds: [0.1, 1000.0], std: 0.0 };
//     let evap = Par { val: 0.01, bounds: [0.0, 0.5], std: 0.0 };
//     let k = 2;
//     let ic = 2;

//     let alpha = Par { val: 1.2, bounds: [0.0, 5.0], std: 0.0 };
//     let beta = Par { val: 1.5, bounds: [0.0, 5.0], std: 0.0 };

//     let max_tau = Par { val: 16.0, bounds: [0.5, 1000.0], std: 0.0 };
//     let interval_tau = Par { val: 15.8, bounds: [0.5, 1000.0], std: 0.0 };


//     let current_best_score = evaluate_score2(grs, i, j,
//         k, alpha, beta, c, evap,
//     max_tau, interval_tau, ic);
//     return;


// }



fn main() {
    test_segment_tree();

    let args: Vec<String> = std::env::args().collect();
    //println!("{:?}", args);
    if let Some(fg) = args.get(1) {
        let c = 
            if let Some(Some(i_)) = args.get(2).map(|s| {s.parse::<f64>().ok()}) {
                i_
            } else {
                100.0
            };

        let evap = 
            if let Some(Some(j_)) = args.get(3).map(|s| {s.parse::<f64>().ok()}) {
                j_
            } else {
                0.01
            };

        let seed = 
            if let Some(Some(j_)) = args.get(4).map(|s| {s.parse::<u64>().ok()}) {
                j_
            } else {
                1771
            };

        let w = 
            if let Some(Some(j_)) = args.get(5).map(|s| {s.parse::<f64>().ok()}) {
                j_
            } else {
                0.5
            };
        //println!("{} {}", i, j);
                let data = Data::load("data/samples1000-20000.data");

        println!("Process\nc={}, evap={}, w={}", c, evap, w);
        test_on_graph(&data.samples[0], c, evap, seed, w);

        //let now = Instant::now();
        //test_on_graphs(&grs, i, j);
        //println!("{:#?}", now.elapsed());
    } else {
        //let grs = get_graphs();
        // let mut prng = Xoshiro256PlusPlus::seed_from_u64(898);
        // let t = Graph::random_graph(30, 300, &mut prng);
        // t.to_dot();
        // return;
        //Data::generate_samples(1, 1000, 20000, 123).save("data/samples1000-20000.data");
        //return;

        println!("loading samples...");
        let data = Data::load("data/samples1000-20000.data");
        println!("launching test.");
        test_on_graph(&data.samples[0], 8000.0, 0.4_f64, 181, 0.5);

        //test_on_facebook(800.0, 0.09, 171);
        //test_on_graphs2(&grs, 0, 1);
        return;
        // let mut prng: Xoshiro256PlusPlus = Xoshiro256PlusPlus::seed_from_u64(890);
        // let mut dist;
        // let mut dist2;
        // let mut dist_approx;
        // let mut dist_approx2;
        // let mut c = 0;
        // let mut inv = 0;

        // let mut ecs = 0.0;
        // let mut ecm: f64 = 0.0;

        // for (_i, g) in grs[0..1].iter().enumerate() {
        //     for _l in 0..20 {
        //         println!("computing {}...", _i * 20 + (_l+1));

        //         {
        //         let mut aco_dummy = ACO::new_dummy(g.clone());
        //         // let disto3 = aco_dummy.greedy_stretch();
        //         //println!("{:?}  {} {}", aco.get_tau_tab_info(), disto, disto2);
        //         aco_dummy.random_tree(&mut prng);
        //         dist = aco_dummy.tree.distorsion(&mut aco_dummy.tree_dist_matrix, &aco_dummy.dist_matrix);
        //         dist_approx = aco_dummy.tree.distorsion_approx(&mut aco_dummy.tree_dist_matrix, &aco_dummy.g.get_edges(), &aco_dummy.edge_betweeness_centrality);
        //         //println!("{} {}", dist, dist_approx);
        //         }
        //         {
        //         let mut aco_dummy2 = ACO::new_dummy(g.clone());
        //         // let disto3 = aco_dummy.greedy_stretch();
        //         //println!("{:?}  {} {}", aco.get_tau_tab_info(), disto, disto2);
        //         aco_dummy2.random_tree(&mut prng);
        //         dist2 = aco_dummy2.tree.distorsion(&mut aco_dummy2.tree_dist_matrix, &aco_dummy2.dist_matrix);
        //         dist_approx2 = aco_dummy2.tree.distorsion_approx(&mut aco_dummy2.tree_dist_matrix, &aco_dummy2.g.get_edges(), &aco_dummy2.edge_betweeness_centrality);
        //         //println!("{} {}", dist2, dist_approx2);
        //         }
        //         c += 1;
        //         let ec = if dist < dist2 && dist_approx > dist_approx2 {
        //             //println!("{}", (_i+1) * 40 + (_l+1));
        //             inv += 1;

        //             let e1 = (dist2-dist)/dist2;
        //             let e2 = (dist_approx - dist_approx2)/dist_approx;

        //             e2 + e1

        //         }
        //         else if dist > dist2 && dist_approx < dist_approx2 {
        //             //println!("{}", (_i+1) * 40 + (_l+1));
        //             inv += 1;
        //             let e1 = (dist-dist2)/dist;
        //             let e2 = (dist_approx2 - dist_approx)/dist_approx2;

        //             e2 + e1

        //         } else {
        //             0.0
        //         };
        //         ecs += ec;
        //         ecm = ecm.max(ec);
        //         //println!("{} {}    {} {}", dist, dist2, dist_approx, dist_approx2);

        //     }

        // }

        // println!("{}%, c={}, ecs={}%, ecm={}%", inv as f64 / c as f64 * 100.0, c, ecs / inv as f64 * 100.0, ecm * 100.);


        // //let now = Instant::now();
        //test_on_graphs2(&grs, 134, 135);

    }


}
