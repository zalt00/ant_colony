use std::{fs::File, io::{Read, Write}, thread::sleep, time::Duration};


use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::{graph::{repr, Graph, Par, ACO, CHEPA}, greedy::greedy_algo};

pub mod graph;
pub mod my_rand;
pub mod greedy;
pub mod aco2;

fn test_graph() {
    let mut s = String::new();
    let mut file = File::open("test.graph").expect("error");

    file.read_to_string(&mut s);

    let coef_s = 0.0;
    let coef_c = 0;

    for (i, gs) in s.split("::").enumerate() {
        let gstring = gs.trim().to_string();
        if gstring.len() > 0 {

            let (disto, g) = Graph::from_string(gstring).expect("welp");

            if i == 0 && false {
                println!("{}",repr(&g.get_dist_matrix()));
                return
            }


            //let mut aco = ACO::new(g, 10, Par::new_free(1.0), Par::new_free(1.0), Par::new_free(0.01), Par::new_free(0.1));
            // let (disto2, tree) = aco.launch(30);

            // println!("{}:   {}   |   {}",  disto2 / disto, disto, disto2);
            // coef_c += 1;
            // coef_s += disto2 / disto;
            //if i == 0 {break}
        }
    }

    println!("{}", coef_s / coef_c as f64);
}

pub fn get_graphs() -> Vec<Graph> {
    let mut s = String::new();
    let mut file = File::open("test.graph").expect("error");
    let mut v = vec![];
    file.read_to_string(&mut s);

    let coef_s = 0.0;
    let coef_c = 0;

    for (i, gs) in s.split("::").enumerate() {
        let gstring = gs.trim().to_string();
        if gstring.len() > 0 {

            let (disto, g) = Graph::from_string(gstring).expect("welp");

            // println!("{:?}", g.get_edge_betweeness_centrality());

            v.push(g);

            //if i == 0 {break}
        }
    }

    v
}

pub fn evaluate_score(grs: &Vec<Graph>, i: usize, j: usize, k: usize,
    alpha: Par<f64>, beta: Par<f64>, c: Par<f64>, evap: Par<f64>, max_tau: Par<f64>,
    interval_tau: Par<f64>, iter_count: usize) -> f64 {
    
    unsafe {CHEPA = 0.0};

    let mut s = 0.0;
    let mut sg = 0.0;
    let mut sgs = 0.0;
    for (o, g) in grs[i..j].iter().enumerate() {


        let g2 = g.clone();
        let (disto2, tgreedy) = greedy_algo(&g);

        let mut aco = ACO::new(g2, k, alpha, beta, c, evap, max_tau, interval_tau);
        aco.base_dist = disto2;
        aco.base_tree = tgreedy;
        //let disto = 0.0;
        //sleep(Duration::from_secs(1));
        let (disto, t) = aco.launch(iter_count);

        // let mut aco_dummy = ACO::new_dummy(g.clone());
        // let disto3 = aco_dummy.greedy_stretch();
        //println!("{:?}  {} {}", aco.get_tau_tab_info(), disto, disto2);

        

        //println!("{:?}", aco.tau_matrix);
        //println!("{:?}", t.get_edges());

        s += disto;
        sg += disto2;
        // sgs += disto3;
        //sleep(Duration::from_secs(2));
    }
    println!("{};{};{};{}", unsafe{graph::CHEPA}, s / (j-i) as f64, s / sg, sg / (j-i) as f64);
    


    s / (j-i) as f64
}




pub fn test_on_graphs(grs: &Vec<Graph>, i: usize, j: usize) {
    let c = Par { val: 100.0, bounds: [0.1, 1000.0], std: 0.0 };
    let evap = Par { val: 0.01, bounds: [0.0, 0.5], std: 0.0 };
    let k = 20;
    let ic = 30* 40 * 20 / 10;

    let alpha = Par { val: 1.2, bounds: [0.0, 5.0], std: 0.0 };
    let beta = Par { val: 1.5, bounds: [0.0, 5.0], std: 0.0 };

    let max_tau = Par { val: 16.0, bounds: [0.5, 1000.0], std: 0.0 };
    let interval_tau = Par { val: 15.8, bounds: [0.5, 1000.0], std: 0.0 };


    let current_best_score = evaluate_score(grs, i, j,
        k, alpha, beta, c, evap,
    max_tau, interval_tau, ic);
    return;


}

pub fn evaluate_score2(grs: &Vec<Graph>, i: usize, j: usize, k: usize,
    alpha: Par<f64>, beta: Par<f64>, c: Par<f64>, evap: Par<f64>, max_tau: Par<f64>,
    interval_tau: Par<f64>, iter_count: usize) -> f64 {
    
    unsafe {CHEPA = 0.0};

    
    let mut s = 0.0;
    let mut sg = 0.0;
    let mut sgs = 0.0;
    for (o, g) in grs[i..j].iter().enumerate() {

        let g2 = g.clone();
        let (disto2, tgreedy) = greedy_algo(&g);

        let mut aco = ACO::new(g2, k, alpha, beta, c, evap, max_tau, interval_tau);
        aco.base_dist = disto2;
        aco.base_tree = tgreedy;
        //let disto = 0.0;
        //sleep(Duration::from_secs(1));
        let (disto, t) = aco.launch(iter_count);
        //sleep(Duration::from_secs(1));

        //let disto2 = 0.0;
        let mut output = File::create("trace.json").expect("welp");
        writeln!(output, "{:?}", aco.trace);



        //println!("{:?}", aco.tau_matrix);
        //println!("{:?}", t.get_edges());

        s += disto;
        sg += disto2;
        // sgs += disto3;
        //sleep(Duration::from_secs(2));
    }
    println!("{};{};{};{}", unsafe{graph::CHEPA}, s / (j-i) as f64, s / sg, sg / (j-i) as f64);
    


    s / (j-i) as f64
}




pub fn test_on_graphs2(grs: &Vec<Graph>, i: usize, j: usize) {
    let c = Par { val: 100.0, bounds: [0.1, 1000.0], std: 0.0 };
    let evap = Par { val: 0.01, bounds: [0.0, 0.5], std: 0.0 };
    let k = 20;
    let ic = 30* 40 * 20 / 10;

    let alpha = Par { val: 1.2, bounds: [0.0, 5.0], std: 0.0 };
    let beta = Par { val: 1.5, bounds: [0.0, 5.0], std: 0.0 };

    let max_tau = Par { val: 16.0, bounds: [0.5, 1000.0], std: 0.0 };
    let interval_tau = Par { val: 15.8, bounds: [0.5, 1000.0], std: 0.0 };


    let current_best_score = evaluate_score2(grs, i, j,
        k, alpha, beta, c, evap,
    max_tau, interval_tau, ic);
    return;


}



fn main() {
    let args: Vec<String> = std::env::args().collect();
    //println!("{:?}", args);
    if let Some(fg) = args.get(1) {
        let i = 
            if let Some(Some(i_)) = args.get(2).map(|s| {s.parse::<usize>().ok()}) {
                i_
            } else {
                0
            };

        let j = 
            if let Some(Some(j_)) = args.get(3).map(|s| {s.parse::<usize>().ok()}) {
                j_
            } else {
                10
            };
        //println!("{} {}", i, j);
        let grs = get_graphs();
        //let now = Instant::now();
        test_on_graphs(&grs, i, j);
        //println!("{:#?}", now.elapsed());
    } else {
        let grs = get_graphs();

        let mut prng: Xoshiro256PlusPlus = Xoshiro256PlusPlus::seed_from_u64(890);
        let mut dist;
        let mut dist2;
        let mut dist_approx;
        let mut dist_approx2;
        let mut c = 0;
        let mut inv = 0;

        let mut ecs = 0.0;
        let mut ecm: f64 = 0.0;

        for (_i, g) in grs[0..40].iter().enumerate() {
            for _l in 0..20 {
                println!("computing {}...", _i * 20 + (_l+1));

                {
                let mut aco_dummy = ACO::new_dummy(g.clone());
                // let disto3 = aco_dummy.greedy_stretch();
                //println!("{:?}  {} {}", aco.get_tau_tab_info(), disto, disto2);
                aco_dummy.random_tree(&mut prng);
                dist = aco_dummy.tree.distorsion(&mut aco_dummy.tree_dist_matrix, &aco_dummy.dist_matrix);
                dist_approx = aco_dummy.tree.distorsion_approx(&mut aco_dummy.tree_dist_matrix, &aco_dummy.g.get_edges(), &aco_dummy.edge_betweeness_centrality);
                //println!("{} {}", dist, dist_approx);
                }
                {
                let mut aco_dummy2 = ACO::new_dummy(g.clone());
                // let disto3 = aco_dummy.greedy_stretch();
                //println!("{:?}  {} {}", aco.get_tau_tab_info(), disto, disto2);
                aco_dummy2.random_tree(&mut prng);
                dist2 = aco_dummy2.tree.distorsion(&mut aco_dummy2.tree_dist_matrix, &aco_dummy2.dist_matrix);
                dist_approx2 = aco_dummy2.tree.distorsion_approx(&mut aco_dummy2.tree_dist_matrix, &aco_dummy2.g.get_edges(), &aco_dummy2.edge_betweeness_centrality);
                //println!("{} {}", dist2, dist_approx2);
                }
                c += 1;
                let ec = if dist < dist2 && dist_approx > dist_approx2 {
                    println!("{}", (_i+1) * 40 + (_l+1));
                    inv += 1;

                    let e1 = (dist2-dist)/dist2;
                    let e2 = (dist_approx - dist_approx2)/dist_approx;

                    e2 + e1

                }
                else if dist > dist2 && dist_approx < dist_approx2 {
                    println!("{}", (_i+1) * 40 + (_l+1));
                    inv += 1;
                    let e1 = (dist-dist2)/dist;
                    let e2 = (dist_approx2 - dist_approx)/dist_approx2;

                    e2 + e1

                } else {
                    0.0
                };
                ecs += ec;
                ecm = ecm.max(ec);
                println!("{} {}    {} {}", dist, dist2, dist_approx, dist_approx2);

            }

        }

        println!("{}%, c={}, ecs={}%, ecm={}%", inv as f64 / c as f64 * 100.0, c, ecs / inv as f64 * 100.0, ecm * 100.);


        //let now = Instant::now();
        //test_on_graphs2(&grs, 134, 135);

    }


}
