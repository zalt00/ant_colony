
use std::collections::HashMap;

use rand::{seq::SliceRandom, RngCore};

use crate::{graph::{graph_core::GraphCore, RootedTree}, my_rand::{sample_slow, Prng}, utils::Uf};

#[derive(Debug, Clone, Copy)]
pub enum NeighborhoodStrategies {
    EdgeSwap,
    EdgeSubtreeRelocation,
    CriticalPathSubtreeRelocation
}


impl RootedTree {
    pub fn edge_removable_for_swap(&mut self, ei: usize, edges: &Vec<[usize; 2]>) -> [Vec<usize>; 2]
    {

        // essentiellement, renvoie un chemin dans l'arbre entre les deux extremites de ei
        let [u, v] = edges[ei];

        if self.parent[u] != v && self.parent[v] != u {

            let mut resu = vec![];
            let mut resv = vec![];
            let mut wu = u;
            let mut wv = v;

            while self.depths[wu] > self.depths[wv] {
                resu.push(wu);
                wu = self.parent[wu];
            }

            while self.depths[wv] > self.depths[wu] {
                resv.push(wv);
                wv = self.parent[wv];
            }

            while wu != wv {
                // println!("w {} {} {}", wu, wv, self.root);
                // println!("{} {}", self.depths[wu], self.depths[wv]);
                resu.push(wu);
                resv.push(wv);

                wu = self.parent[wu];
                if wv == usize::MAX {
                    println!("{:?} {:?} {}", resu, resv, self.root);
                }
                wv = self.parent[wv];
            }

            resu.push(wu);
            resv.push(wv);

            [resu, resv]
        } else {
                        //println!("{} {} {:?}" , u, v, self.parents);

            [vec![], vec![]]
        }
    }

    pub fn do_the_edge_swap(&mut self, dt_rm: &Vec<usize>, dt_oth: &Vec<usize>, rmi: usize) {
        {
            let mut wi = rmi;
            // println!("rmi={}", rmi);

            while wi > 0 {
                self.change_parent(dt_rm[wi], dt_rm[wi-1]);
                //self.parent[dt_rm[wi]] = dt_rm[wi - 1]; // change le parent
                //self.children[dt_rm[wi-1]].push(dt_rm[wi]);

                // let mut rmj = usize::MAX;
                // for (j, &node) in self.children[dt_rm[wi + 1]].iter().enumerate() {
                //     if node == dt_rm[wi] {
                //         rmj = j
                //     }
                // }
                // self.children[dt_rm[wi + 1]].swap_remove(rmj); // supprime l'enfant du parent

                wi -= 1;
            }
        }
        self.change_parent(dt_rm[0], dt_oth[0]);
        // self.children[dt_oth[0]].push(dt_rm[0]);
        // let mut rmj = usize::MAX;
        // for (j, &node) in self.children[dt_rm[1]].iter().enumerate() {
        //     if node == dt_rm[0] {
        //         rmj = j
        //     }
        // }
        // self.children[dt_rm[1]].swap_remove(rmj); // supprime l'enfant du parent


        
        // aaaaarg j'avais oublie a ce truc -> dans l'idee pas forcement utile de tout calculer
        self.recompute_depths();
        self.update_leaves();

    }

    pub fn edge_swap_random(&mut self, prng: &mut Prng, edges: &Vec<[usize; 2]>) -> bool {
        // /!\ appeler update_parent avant

        // pour tester, essayer d'enlever plus tard
        //self.update_parents();

        let ei = (prng.next_u64() % edges.len() as u64) as usize;
        //println!("{:?}", edges[ei]);
        // println!("{:?}", edges[ei]);
        let dt = self.edge_removable_for_swap(ei, edges);
        // println!("{:?}", dt);

        let [resu, resv] = &dt;
        if resu.len() == 0 {return false;}
        let k = resu.len() + resv.len() - 2;
        let mut rk = (prng.next_u64() % k as u64) as usize;

        let mut dtrmi = 0;
        let mut dtothi = 1;
        // println!("{:?} k={}  {}", dt, k, rk);

        if rk >= resu.len() - 1 {
            rk -= resu.len() - 1;
            dtrmi = 1;
            dtothi = 0;
        }


        self.do_the_edge_swap(&dt[dtrmi], &dt[dtothi], rk);

        true
    }


    pub fn edge_swap_random_biaised(&mut self, prng: &mut Prng, proba: &[f64], edges: &Vec<[usize; 2]>) -> bool {
        // /!\ appeler update_parent avant

        // pour tester, essayer d'enlever plus tard
        //self.update_parents();

        let ei = sample_slow(proba
            .iter()
            .zip(edges)
            .map(| (&p, &[u, v]) | {if self.has_edge(u, v) {0.0} else {p}} ), prng);
        // println!("{:?}", edges[ei]);
        let dt = self.edge_removable_for_swap(ei, edges);
        // println!("{:?}", dt);

        let mut edge_cycle_set = HashMap::new();

        let [resu, resv] = &dt;
                if resu.len() == 0 {println!("welp"); return false;}

        for i in 0..(resu.len() - 1) {
            let u = resu[i];
            let v = resu[i+1];
            edge_cycle_set.insert((u.min(v), u.max(v)), i);
        }
        for i in 0..(resv.len() - 1) {
            let u = resv[i];
            let v = resv[i+1];
            edge_cycle_set.insert((u.min(v), u.max(v)), i + resu.len() - 1);
        }



        let k = resu.len() + resv.len() - 2;
        let mut probk = vec![f64::NAN; k];

        for (&[u, v], &p) in edges.iter().zip(proba) {
            if let Some(entry) = edge_cycle_set.get(&(u.min(v), u.max(v))) {
                probk[*entry] = (1.0 - p).max(0.0);

            }
        }

        let mut rk = sample_slow(probk.iter().cloned(), prng);

        let mut dtrmi = 0;
        let mut dtothi = 1;
        // println!("{:?} k={}  {}", dt, k, rk);

        if rk >= resu.len() - 1 {
            rk -= resu.len() - 1;
            dtrmi = 1;
            dtothi = 0;
        }


        self.do_the_edge_swap(&dt[dtrmi], &dt[dtothi], rk);

        true
    }



    pub fn edge_swap(&mut self, prng: &mut Prng, edges: &Vec<[usize; 2]>, ei: usize) -> ([usize; 2], bool) {
        // /!\ appeler update_parent avant

        // pour tester, essayer d'enlever plus tard
        //self.update_parents();

        // println!("{:?}", edges[ei]);
        let dt = self.edge_removable_for_swap(ei, edges);
        // println!("{:?}", dt);

        let [resu, resv] = &dt;
        if resu.len() == 0 {return ([0, 0], false);}
        let k = resu.len() + resv.len() - 2;
        let mut rk = (prng.next_u64() % k as u64) as usize;

        let mut dtrmi = 0;
        let mut dtothi = 1;
        // println!("{:?} k={}  {}", dt, k, rk);

        if rk >= resu.len() - 1 {
            rk -= resu.len() - 1;
            dtrmi = 1;
            dtothi = 0;
        }


        self.do_the_edge_swap(&dt[dtrmi], &dt[dtothi], rk);

        ([dt[dtrmi][rk], dt[dtrmi][rk + 1]], true)
    }


    pub fn subtree_swap_with_edge<T: GraphCore>(&mut self, ei: usize, prng: &mut Prng,
        edges: &Vec<[usize; 2]>, g: &T, tree_buf: &mut T) -> bool
    {

        let dt = self.edge_removable_for_swap(ei, edges);
        if dt[0].len() == 0 {
            return false;
        }
        let mut vertices = vec![];

        for &v in dt[0].iter() {
            vertices.push(v);
        }
        for &v in dt[1][0..(dt[1].len() - 1)].iter() {
            vertices.push(v);
        }

        self.subtree_swap_with_vertices(prng, &vertices, g, tree_buf);

        true

    }

    pub fn subtree_swap_with_random_edge<T: GraphCore>(&mut self, prng: &mut Prng, 
        edges: &Vec<[usize; 2]>, g: &T, tree_buf: &mut T) -> bool {
        
        let ei = (prng.next_u64() % edges.len() as u64) as usize;
        self.subtree_swap_with_edge(ei, prng, edges, g, tree_buf)
    }

    // critical path subtree swap
    pub fn shuffle_tree_and_random_leaf(&mut self, prng: &mut Prng) -> usize {
        self.leaves.shuffle(prng);

        self.leaves[0]
    }

    pub fn get_critical_path<T: GraphCore>(&mut self, prng: &mut Prng, tree_buf: &mut T) -> Vec<usize> {
        let mut ans = vec![];

        let leaf = self.shuffle_tree_and_random_leaf(prng);
        //println!("{}", leaf);
        self.fill_graph(tree_buf);

        let leaf2 = tree_buf.bfs_further_vertex(leaf, false, &mut vec![]);
        //println!("{}", leaf2);
        let _ = tree_buf.bfs_further_vertex(leaf2, true, &mut ans);

        ans
    }

    pub fn subtree_swap_with_random_critical_path<T: GraphCore>(&mut self, prng: &mut Prng, g: &T, tree_buf: &mut T) {
        let cp = self.get_critical_path(prng, tree_buf);
        self.subtree_swap_with_vertices(prng, &cp, g, tree_buf)
    }
    
    fn subtree_swap_with_vertices<T: GraphCore>(&self, prng: &mut Prng, vertices: &Vec<usize>,
        g: &T, tree_buf: &mut T)
    {
        //println!("{:?}", vertices);
        tree_buf.reset();

        let mut covered_vertices = vec![false; self.n];
        //let vertices = self.get_critical_path(prng, tree_buf);
        for &v in vertices {
            covered_vertices[v] = true;
        }

        //println!("{:?}", vertices);

        let mut possible_edges = vec![];

        for &v in vertices.iter() {
            for &w in g.get_neighbors(v) {
                if v < w && covered_vertices[w] {
                    possible_edges.push([v, w])
                }
            }
        }

        possible_edges.shuffle(prng);

        let n2 = vertices.len();
        let mut uf = Uf::init(self.n);

        let mut m = 0;
        let mut i = 0;
        while m < n2 - 1 {
            let [u, v] = possible_edges[i];

            if uf.find(u) != uf.find(v) {
                uf.union(u, v);
                tree_buf.add_edge_unckecked(u, v);
                m += 1;
            }

            i += 1;
        }

        for [u, v] in self.edges() {
            if !covered_vertices[u] || !covered_vertices[v] {
                tree_buf.add_edge_unckecked(u, v);
            }
            
        }



        

    }


}








