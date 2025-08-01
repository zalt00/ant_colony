use std::collections::{HashMap, HashSet};

use crate::{graph::{compressed_graph::CompressedGraph, graph_core::GraphCore, RootedTree}, utils::{IterExt, PairExt}};



impl RootedTree {
    pub fn wiener(&mut self, additional_edges: &[[usize; 2]], g: &CompressedGraph) -> u64 {
        let mut size = vec![0; self.n];

        self.precalcul_sizes(self.root, &mut size);
        let mut covered_edges = HashSet::new();
        let mut inter_count = 0;
        for e in additional_edges {
            let cycle = self.path_between(e[0], e[1]);
            println!("{}", cycle[0].len() + cycle[1].len() - 1);
            let mut s = 0;
            for &u in cycle[0][..cycle[0].len() - 1].iter() {
                s += size[u] * (self.n as u64 - size[u]);
            }

            for &u in cycle[1][..cycle[1].len() - 1].iter() {
                s += size[u] * (self.n as u64 - size[u]);
            }


            let mut node_sizes = [vec![0; cycle[0].len()], vec![0; cycle[1].len()]];
            for ci in [0,1] {
                for (val, &u) in node_sizes[ci].iter_mut().zip(cycle[ci].iter()) {
                    *val = size[u];
                    //println!("{}", size[u]);
                }
            }
            //println!("nd {:?}", node_sizes);
            for ci in [0,1] {
                for i in 1..cycle[ci].len() {
                    let u_prev = cycle[ci][i - 1];
                    node_sizes[ci][i] -= size[u_prev];
                    if ci == 1 && i == cycle[ci].len() - 1 {
                        println!("ya");
                        *node_sizes[0].last_mut().unwrap() -= size[u_prev];
                    }
                }
            }
            *node_sizes[0].last_mut().unwrap() += self.n as u64 - size[*cycle[0].last().unwrap()];


            let mut full_cycle = Vec::new();
            let mut full_cycle_sizes = Vec::new();
            for (&u, &s) in cycle[0].iter().zip(node_sizes[0].iter()) {
                full_cycle_sizes.push(s);
                full_cycle.push(u);
            }

            for (&v, &s) in cycle[1].iter().zip(node_sizes[1].iter()).rev().skip(1) {
                full_cycle.push(v);
                full_cycle_sizes.push(s);
            }

            {
                let u = full_cycle[0];
                let v = full_cycle[full_cycle.len() - 1];
                assert!([u, v] == *e || [v, u] == *e);
            }

            //let mm = if full_cycle.len() % 2 == 1 {full_cycle.len() / 2;
            let mut cumulative_sums = vec![0; full_cycle.len()*2];
            let mut cumulative_weighted_sums = vec![0; full_cycle.len()*2];

            cumulative_sums[0] = full_cycle_sizes[0]; // a_0 + a_1 + ...
            cumulative_weighted_sums[0] = 0;          // 0*a_0 + 1*a_1 + ...
            for i in 1..cumulative_sums.len() {
                cumulative_sums[i] = cumulative_sums[i-1] 
                    + full_cycle_sizes[i % full_cycle_sizes.len()];
                cumulative_weighted_sums[i] = cumulative_weighted_sums[i-1] 
                    + full_cycle_sizes[i % full_cycle_sizes.len()] * i as u64;
            }
            let mut s2 = 0;
            let mm = full_cycle.len() / 2;

            if full_cycle.len() % 2 == 1 {
                for i in 0..full_cycle.len() {
                    // somme des poids pour tous les chemins de taille
                    // <= mm partant de i, en multipliant par le nombre d'arete du chemin
                    // pour compter la contribution du chemin à la ebc de chaque arete

                    // a_{i+1} + ... + a_{mm+i} 
                    let unw_sum = cumulative_sums[mm + i] - cumulative_sums[i];

                    // (i+1)a_{i+1} + ... + (mm+i)a_{mm+i} 
                    let w_sum = cumulative_weighted_sums[mm + i] - cumulative_weighted_sums[i];
                    
                    // a_{i+1} + 2a_{i+2} + ...
                    let w_sum_corrected = w_sum - i as u64 * unw_sum;


                    s2 += full_cycle_sizes[i] * w_sum_corrected;
                }
            } else {
                for i in 0..full_cycle.len() {
                    // somme des poids pour tous les chemins de taille
                    // <= mm-1 partant de i, en multipliant par le nombre d'arete du chemin
                    // pour compter la contribution du chemin à la ebc de chaque arete

                    // a_{i+1} + ... + a_{mm+i-1} 
                    let unw_sum = cumulative_sums[mm + i - 1] - cumulative_sums[i];

                    // (i+1)a_{i+1} + ... + (mm+i)a_{mm+i-1} 
                    let w_sum = cumulative_weighted_sums[mm + i - 1] - cumulative_weighted_sums[i];
                    
                    // a_{i+1} + 2a_{i+2} + ...
                    let w_sum_corrected = w_sum - i as u64 * unw_sum;


                    s2 += full_cycle_sizes[i] * w_sum_corrected;
                }

                // on ajoute a part les chemins de taille mm pour eviter de les compter 2 fois
                for i in 0..mm {
                    s2 += full_cycle_sizes[i] * full_cycle_sizes[i + mm] * mm as u64;
                }
            

            }
            // println!("{:?}, {:?}", full_cycle, full_cycle_sizes);
            // for u in full_cycle.iter() {
            //     println!("arity={}", self.arity[*u]);
            //     println!("u: {}, size={}", *u, size[*u]);

            // }
            println!("cycle length={}", full_cycle.len());
            println!("cycle: {:?}", full_cycle);

            let wiener_calc = self.distance_sum() - s + s2;
            println!("wiener_calc = {}", wiener_calc);
            // let mut t_g = self.to_graph(g);
            // t_g.add_edge_unckecked(e[0], e[1]);
            // let wiener_slow = t_g.wiener(&mut vec![u32::MAX; self.n*self.n]);
            // println!("wiener 2 =    {}", wiener_slow);

            // assert_eq!(wiener_calc, wiener_slow);
            let mut intersecting = false;
            let mut prev = *full_cycle.last().unwrap();
            for &u in full_cycle.iter() {
                if covered_edges.contains(&[prev, u].sorted()) {
                    intersecting = true;
                    println!("{:?}", [prev, u]);
                } else {
                    covered_edges.insert([prev, u].sorted());
                }
                prev = u;
            }

            if intersecting {
                println!("intersecting");
                inter_count += 1;
            }

        }

        println!("inter_count: {}, ratio: {}%", inter_count, inter_count as f64 / additional_edges.len() as f64 * 100.0);

        todo!()
    }
}

