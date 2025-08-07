use std::collections::{HashMap, HashSet};

use crate::{graph::{compressed_graph::CompressedGraph, graph_core::GraphCore, RootedTree}, utils::{CompressedVecVec, IterExt, PairExt}};



impl RootedTree {
    pub fn wiener(&mut self, additional_edges: &[[usize; 2]], g: &CompressedGraph) -> u64 {
        let mut size = vec![0; self.n];

        self.precalcul_sizes(self.root, &mut size);

        let mut ebc_hmap = HashMap::new();
        for e in self.edges() {
            ebc_hmap.insert(e.sorted(), size[e[0]] * (self.n as u64 - size[e[0]]));
        }

        let base_ebc_hmap = ebc_hmap.clone();

        for e in additional_edges {
            let cycle = self.path_between(e[0], e[1]);

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
                        //println!("ya");
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

            let mut ebc = vec![0; full_cycle.len()];

            let half_len = full_cycle.len() / 2;
            let cycle_len = full_cycle.len();
            for i in 0..full_cycle.len() {
                for path_len in 1..=half_len {
                    if path_len * 2 == cycle_len && i >= half_len {
                        continue;
                    } else {
                        let w = full_cycle_sizes[i] * full_cycle_sizes[(i + path_len) % cycle_len];
                        for j in i..(i+path_len) {
                            ebc[j % cycle_len] += w;
                        }
                    }
                }
            }

            // update ebc
            for i in 0..full_cycle.len() {
                let u = full_cycle[i];
                let v = full_cycle[(i+1)%cycle_len];

                let e = [u, v].sorted();
                if base_ebc_hmap.contains_key(&e) {
                    ebc_hmap.entry(e).and_modify(|val| {
                        *val = *val + ebc[i] - base_ebc_hmap[&e];
                    });
                } else {
                    ebc_hmap.insert([u, v].sorted(), ebc[i]);
                }

            }


            // println!("cycle length={}", full_cycle.len());
            // println!("cycle: {:?}", full_cycle);

            //let wiener_calc = ebc_hmap.values().sum::<u64>();
            // println!("wiener_calc = {}", wiener_calc);
            // let wiener_slow = t_g.wiener(&mut vec![u32::MAX; self.n*self.n]);
            // println!("wiener 2 =    {}", wiener_slow);

            //assert_eq!(wiener_calc, wiener_slow);

        }
        let wiener_calc = ebc_hmap.values().sum::<u64>();

        // println!("inter_count: {}, ratio: {}%", inter_count, inter_count as f64 / additional_edges.len() as f64 * 100.0);

        wiener_calc
    }
}

