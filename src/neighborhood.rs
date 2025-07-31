

use crate::graph::graph_generator::GraphRng;
use std::collections::HashMap;

use rand::{seq::SliceRandom, RngCore};

use crate::{graph::{graph_core::GraphCore, RootedTree}, my_rand::{sample_slow, Prng}, utils::Uf};

#[derive(Debug, Clone, Copy)]
pub enum NSVal {
    Sqrt(usize, usize),
    N(usize, usize)
}

impl NSVal {
    pub fn to_n2(self, n: usize) -> usize {
        match self {
            Self::Sqrt(mul, div) => {n.isqrt() * mul / div},
            Self::N(mul, div) => {n * mul / div}
        }.max(10)
    }
}


#[derive(Debug, Clone, Copy)]
pub enum NeighborhoodStrategies {
    EdgeSwap,
    EdgeSubtreeRelocation,
    CriticalPathSubtreeRelocation,
    CriticalPathSubtreeVNS,
    SpiderSubtreeVNS(NSVal),
    SpiderSubtreeSwap(NSVal),
    SubtreeSubtreeVNS(NSVal),
    SubtreeSubtreeSwap(NSVal)
}


impl RootedTree {
    pub fn edge_removable_for_swap(&mut self, ei: usize, edges: &Vec<[usize; 2]>) -> [Vec<usize>; 2]
    {
        // let ar_bkp = self.arity.clone();
        // self.arity.fill(0);
        // self.recompute_arity();
        // if ar_bkp != self.arity {
        //     println!("{:?}", ar_bkp);
        //     println!("{:?}", &self.arity);
        //     println!("");
        // }

        // self.update_leaves();

        // self.recompute_depths();

        // essentiellement, renvoie un chemin dans l'arbre entre les deux extremites de ei
        let [u, v] = edges[ei];

        if self.parent[u] != v && self.parent[v] != u {

            let mut resu = vec![];
            let mut resv = vec![];
            let mut wu = u;
            let mut wv = v;
            // println!("la1");
            while self.depths[wu] > self.depths[wv] {
                resu.push(wu);
                wu = self.parent[wu];
            }

            while self.depths[wv] > self.depths[wu] {
                resv.push(wv);
                wv = self.parent[wv];
            }
            // println!("la2");

            while wu != wv {
                // println!("w {} {} {}", wu, wv, self.root);
                // println!("{} {}", self.depths[wu], self.depths[wv]);
                resu.push(wu);
                resv.push(wv);

                // println!("{} {} ", wu, wv);
                // println!("depth {} {} ", self.depths[wu], self.depths[wv]);
                // println!("root {}", self.root);
                // println!("{:?}", &self.parent);
                // println!("{:?}", &self.depths);
                // for i in 0..self.n {
                //     if i == self.root {
                //         print!("root, ");
                //     } else {
                //         print!("{}, ", self.depths[self.parent[i]]);
                //     }
                // }
                // print!("\n");

                // println!("{:?}", &self.leaves);

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
        self.update_leaves();
        self.recompute_depths();

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
    pub fn shuffle_tree_and_random_leaf(&self, prng: &mut Prng) -> usize {
        loop {
            let u = (prng.next_u64() % self.n as u64) as usize;
            if self.arity[u] == 0 {
                return u
            }
        }
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

        //println!("subtree swap");
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

    pub fn random_spider(&mut self, n2: usize, prng: &mut Prng) -> Vec<usize> {
        let mut ans = vec![];
        //self.shuffle_tree_and_random_leaf(prng);

        let mut node_order = (0..self.n).collect::<Vec<usize>>();
        node_order.shuffle(prng);

        let mut visited = vec![false; self.n];
        visited[self.root] = true;
        ans.push(self.root);
        let mut i = 0;
        while ans.len() < n2 {
            let u = node_order[i];
            if self.arity[u] > 0 {i+=1; continue;}
            let mut v = u;
            while !visited[v] {
                ans.push(v);
                visited[v] = true;
                v = self.parent[v];
            }
            i += 1;
            
        }
        //println!("ans length {}, n={}, n2={}", ans.len(), self.n, n2);
        ans

    }

    pub fn random_subtree_incomplete<T: GraphCore>(&self, prng: &mut Prng, n2: usize, tree_buf: &mut T) -> Vec<usize> {
        self.fill_graph(tree_buf);

        let mut stack = Vec::new();
        let mut ans = Vec::new();
        let mut visited = vec![false; self.n];
        let u = (prng.next_u64() % self.n as u64) as usize;

        stack.push(u);

        while ans.len() < n2 {
            let i = (prng.next_u64() % stack.len() as u64) as usize;
            let v = stack.swap_remove(i);

            if !visited[v] {
                visited[v] = true;
                ans.push(v);

                for &w in tree_buf.get_neighbors(v) {
                    if !visited[w] {
                        stack.push(w);
                    }
                }
            }

        }
        ans
    }


    #[cfg(not(feature="mean_path_heuristic"))]
    pub fn subtree_vns_with_vertices<T: GraphCore>(&self, _prng: &mut Prng, _vertices: &Vec<usize>,
        _g: &T, _tree_buf: &mut T)
    {panic!()} 

    #[cfg(feature="mean_path_heuristic")]
    pub fn subtree_vns_with_vertices<T: GraphCore+GraphRng>(&self, prng: &mut Prng, vertices: &Vec<usize>,
        g: &T, tree_buf: &mut T, old2new: &mut [usize], new2old: &mut [usize])
    {
        //println!("{:?}", vertices);

        use crate::{solver::{RandomStartBFSTree, Solver}, utils::{renumber_edges, renumber_edges2, HashMapExt}};

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
        //println!("halo ? {}", vertices.len());
        renumber_edges2(&mut possible_edges, old2new, new2old);
        let g2 = T::from_edges(vertices.len(), &possible_edges);
        //println!("{:?}", possible_edges);
        let seed = prng.next_u64();
        let t_better = //VNSWithStartMode1::<T, BFSTree<T>>
        RandomStartBFSTree::<T>::auto_parameters_solve(g2, vec![], vec![], seed, 0.5);
        //println!("euh ? {}", vertices.len());

        for [unew, vnew] in t_better.edges() {
            tree_buf.add_edge_unckecked(new2old[unew], new2old[vnew]);
        }


        for [u, v] in self.edges() {
            if !covered_vertices[u] || !covered_vertices[v] {
                tree_buf.add_edge_unckecked(u, v);
            } 
        }
    }

    pub fn subtree_vns_with_random_critical_path<T: GraphCore+GraphRng>(&mut self, prng: &mut Prng, g: &T, tree_buf: &mut T,
    old2new: &mut [usize], new2old: &mut [usize]) {
        let cp = self.get_critical_path(prng, tree_buf);
        self.subtree_vns_with_vertices(prng, &cp, g, tree_buf, old2new, new2old)
    }

    pub fn subtree_vns_with_random_spider<T: GraphCore+GraphRng>(&mut self, prng: &mut Prng, n2: usize, g: &T, tree_buf: &mut T,
    old2new: &mut [usize], new2old: &mut [usize]) {
        let cp = self.random_spider(n2, prng);
        //println!("cp lenn {} {}", cp.len(), self.leaves.len());
        self.subtree_vns_with_vertices(prng, &cp, g, tree_buf, old2new, new2old)
    }
    pub fn subtree_swap_with_random_spider<T: GraphCore+GraphRng>(&mut self, prng: &mut Prng, n2: usize, g: &T, tree_buf: &mut T) {
        let cp = self.random_spider(n2, prng);
        self.subtree_swap_with_vertices(prng, &cp, g, tree_buf)
    }


    pub fn subtree_vns_with_random_subtree<T: GraphCore+GraphRng>(&mut self, prng: &mut Prng, n2: usize, g: &T, tree_buf: &mut T,
    old2new: &mut [usize], new2old: &mut [usize]) {
        let cp = self.random_subtree_incomplete(prng, n2, tree_buf);
        //println!("cp lenn {} {}", cp.len(), self.leaves.len());
        self.subtree_vns_with_vertices(prng, &cp, g, tree_buf, old2new, new2old)
    }
    pub fn subtree_swap_with_random_subtree<T: GraphCore+GraphRng>(&mut self, prng: &mut Prng, n2: usize, g: &T, tree_buf: &mut T) {
        let cp = self.random_subtree_incomplete(prng, n2, tree_buf);
        self.subtree_swap_with_vertices(prng, &cp, g, tree_buf)
    }
}








