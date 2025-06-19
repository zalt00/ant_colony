use std::{fs::File, io::Read, thread::sleep, time::{Duration, Instant}};

use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::graph::{repr, Graph, Par, ACO};

pub mod graph;
pub mod my_rand;

fn test_graph() {
    let mut s = String::new();
    let mut file = File::open("test.graph").expect("error");

    file.read_to_string(&mut s);

    let mut coef_s = 0.0;
    let mut coef_c = 0;

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

    let mut coef_s = 0.0;
    let mut coef_c = 0;

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
    
    let mut s = 0.0;
    for (o, g) in grs[i..j].iter().enumerate() {

        let g2 = g.clone();
        let mut aco = ACO::new(g2, k, alpha, beta, c, evap, max_tau, interval_tau);
        let (disto, t) = aco.launch(iter_count);
        println!("{:?}  {}", aco.get_tau_tab_info(), disto);

        //println!("{:?}", aco.tau_matrix);
        //println!("{:?}", t.get_edges());

        s += disto;
        //sleep(Duration::from_secs(2));
    }
    println!("{}", s / (j-i) as f64);


    s / (j-i) as f64
}




pub fn basic_hyperparameter_search(grs: &Vec<Graph>) {
    let tot_iter = 200;
    let mut glob_best_c = Par { val: 100.0, bounds: [0.1, 1000.0], std: 0.0 };
    let mut glob_best_evap = Par { val: 0.01, bounds: [0.0, 0.5], std: 0.0 };
    let mut glob_best_k = 50;
    let mut glob_best_ic = 10;

    let mut glob_best_alpha = Par { val: 1.2, bounds: [0.0, 5.0], std: 0.0 };
    let mut glob_best_beta = Par { val: 1.5, bounds: [0.0, 5.0], std: 0.0 };

    let mut glob_best_max_tau = Par { val: 1000.0, bounds: [0.5, 1000.0], std: 0.0 };
    let mut glob_best_interval_tau = Par { val: 999.9, bounds: [0.5, 1000.0], std: 0.0 };

    let mut prng = Xoshiro256PlusPlus::seed_from_u64(121242);

    let mut current_best_score = evaluate_score(grs, 0, 30,
        glob_best_k, glob_best_alpha, glob_best_beta, glob_best_c, glob_best_evap,
    glob_best_max_tau, glob_best_interval_tau, glob_best_ic);
    sleep(Duration::from_secs(20));
    println!("halo");
    for k in [10, 15, 10, 20, 10, 15] {
        let n_iter = tot_iter / k;

        let mut c = glob_best_c.derive(&mut prng, 1.0);
        let mut evap = glob_best_evap.derive(&mut prng, 1.0);
        let mut alpha = glob_best_alpha.derive(&mut prng, 1.0);
        let mut beta = glob_best_beta.derive(&mut prng, 1.0);

        let mut max_tau = glob_best_max_tau.derive(&mut prng, 1.0);
        let mut interval_tau = glob_best_interval_tau.derive(&mut prng, 1.0);
        //println!("{} {} {} {} {} {} {}", alpha.val, beta.val, c.val, evap.val, glob_best_ic, glob_best_k, current_best_score);


        let mut score_prev = evaluate_score(grs, 0, 10, k, alpha, beta, c, evap,
            max_tau, interval_tau, n_iter);
        if score_prev < current_best_score {
            current_best_score = score_prev;
            glob_best_c = c;
            glob_best_evap = evap;
            glob_best_ic = n_iter;
            glob_best_k = k;
            glob_best_alpha = alpha;
            glob_best_beta = beta;
            glob_best_max_tau = max_tau;
            glob_best_interval_tau = interval_tau;
        }
        for _ in 0..5 {
            for p in 1..6 {
                let c2 = c.derive(&mut prng, p as f64);
                let evap2 = evap.derive(&mut prng, p as f64);
                let alpha2 = alpha.derive(&mut prng, p as f64);
                let beta2 = beta.derive(&mut prng, p as f64);

                let interval_tau2 = interval_tau.derive(&mut prng, p as f64);
                let max_tau2 = max_tau.derive(&mut prng, p as f64);

                let score = evaluate_score(grs, 0, 10, k, alpha2, beta2, c2, evap2,
                    max_tau2, interval_tau2, n_iter);
                if score < current_best_score {
                    current_best_score = score;
                    glob_best_c = c2;
                    glob_best_evap = evap2;
                    glob_best_ic = n_iter;
                    glob_best_k = k;
                    glob_best_alpha = alpha2;
                    glob_best_beta = beta2;

                    glob_best_max_tau = max_tau2;
                    glob_best_interval_tau = interval_tau2;
                }
                if score < score_prev {
                    score_prev = score;
                    c = c2;
                    evap = evap2;
                    alpha = alpha2;
                    beta = beta2;

                    max_tau = max_tau2;
                    interval_tau = interval_tau2;
                }

                println!("{} {} a={:.3} b={:.3} c={:.3} e={:.3} ic={:.3} k={:.3}    score={:.3}",
                glob_best_max_tau.val, glob_best_interval_tau.val, glob_best_alpha.val, glob_best_beta.val, glob_best_c.val, glob_best_evap.val, glob_best_ic, glob_best_k, current_best_score);

            }

        }
    }


}




fn main() {
    let grs = get_graphs();
    let now = Instant::now();
    basic_hyperparameter_search(&grs);
    println!("{:#?}", now.elapsed());
}
