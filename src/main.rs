use std::{fs::File, io::{Read, Write}, time::Instant};

use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::graph::{repr, Graph, ACO};

pub mod graph;

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


            let mut aco = ACO::new(g, 10, 1, 0.01, 0.1);
            let (disto2, tree) = aco.launch(30);

            println!("{}:   {}   |   {}",  disto2 / disto, disto, disto2);
            coef_c += 1;
            coef_s += disto2 / disto;
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
            v.push(g);
            //if i == 0 {break}
        }
    }

    v
}

pub fn evaluate_score(grs: &Vec<Graph>, i: usize, j: usize, k: usize,
    alpha: u32, c: f64, evap: f64, iter_count: usize) -> f64 {
    
    let mut s = 0.0;
    for (o, g) in grs[i..j].iter().enumerate() {

        let g2 = g.clone();
        let mut aco = ACO::new(g2, k, alpha, c, evap);
        let (disto, _) = aco.launch(iter_count);
        s += disto;
    }
    println!("");


    s / (j-i) as f64
}

pub fn basic_hyperparameter_search(grs: &Vec<Graph>) {
    let tot_iter = 100;
    let alpha = 2;
    let mut glob_best_c = 0.01;
    let mut glob_best_evap = 0.1;
    let mut glob_best_k = 10;
    let mut glob_best_ic = 10;

    let mut prng = Xoshiro256PlusPlus::seed_from_u64(121242);
    fn rand(prng: &mut Xoshiro256PlusPlus) -> f64 {
        prng.next_u64() as f64 / u64::MAX as f64 - 0.5
    }

    let mut current_best_score = evaluate_score(grs, 0, 10,
        glob_best_k, alpha, glob_best_c, glob_best_evap, glob_best_ic);

    println!("halo");
    for ksq in 1..10 {
        let k = ksq * ksq;
        let n_iter = tot_iter / k;

        let mut c = (glob_best_c + rand(&mut prng) / 2.0).clamp(0.0, 5.0);
        let mut evap = (glob_best_evap + rand(&mut prng) / 2.0).clamp(0.0, 0.5);

        let mut score_prev = evaluate_score(grs, 0, 10, k, alpha, c, evap, n_iter);
        if score_prev < current_best_score {
            current_best_score = score_prev;
            glob_best_c = c;
            glob_best_evap = evap;
            glob_best_ic = n_iter;
            glob_best_k = k;
        }
        for _ in 0..5 {
            for p in 10..13 {
                let c2 = (c + rand(&mut prng) / p as f64).clamp(0.0, 5.0);
                let evap2 = (evap + rand(&mut prng) / p as f64).clamp(0.0, 0.5);

                let score = evaluate_score(grs, 0, 10, k, alpha, c2, evap2, n_iter);
                if score < current_best_score {
                    current_best_score = score;
                    glob_best_c = c2;
                    glob_best_evap = evap2;
                    glob_best_ic = n_iter;
                    glob_best_k = k;
                }
                if score < score_prev {
                    score_prev = score;
                    c = c2;
                    evap = evap2;
                }

                println!("{} {} {} {} {}", glob_best_c, glob_best_evap, glob_best_ic, glob_best_k, current_best_score);

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
