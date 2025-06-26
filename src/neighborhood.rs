use rand::{seq::SliceRandom, RngCore};

use crate::{utils::{TarjanSolver, Uf}, graph::{Graph, RootedTree, N}, my_rand::Prng};


impl RootedTree {
    pub fn edge_removable_for_swap(&self, ei: usize,
        tarjan_solver: &TarjanSolver, edges: &Vec<[usize; 2]>) -> [Vec<usize>; 2]
    {
        // essentiellement, renvoie un chemin dans l'arbre entre les deux extremites de ei

        let [u, v] = edges[ei];

        if self.parents[u] != v && self.parents[v] != u {
            let lca = tarjan_solver.get_results()[u + self.n * v];

            let mut resu = vec![];
            let mut resv = vec![];
            let mut w = u;
            println!("root={}, lca={}, u={}, v={}", self.root, lca, u, v);
            while w != lca {
                resu.push(w);
                w = self.parents[w];
            }

            w = v;
            while w != lca {
                println!("w={}", w);
                resv.push(w);
                w = self.parents[w]
            }

            resu.push(lca);
            resv.push(lca);

            [resu, resv]
        } else {
            [vec![], vec![]]
        }
    }

    pub fn do_the_edge_swap(&mut self, dt_rm: &Vec<usize>, dt_oth: &Vec<usize>, rmi: usize) {
        {
            let mut wi = rmi;
            println!("rmi={}", rmi);

            while wi > 0 {

                self.parents[dt_rm[wi]] = dt_rm[wi - 1]; // change le parent
                self.children[dt_rm[wi-1]].push(dt_rm[wi]);

                let mut rmj = usize::MAX;
                for (j, &node) in self.children[dt_rm[wi + 1]].iter().enumerate() {
                    if node == dt_rm[wi] {
                        rmj = j
                    }
                }
                self.children[dt_rm[wi + 1]].swap_remove(rmj); // supprime l'enfant du parent

                wi -= 1;
            }
        }
        self.parents[dt_rm[0]] = dt_oth[0];
        self.children[dt_oth[0]].push(dt_rm[0]);
        let mut rmj = usize::MAX;
        for (j, &node) in self.children[dt_rm[1]].iter().enumerate() {
            if node == dt_rm[0] {
                rmj = j
            }
        }
        self.children[dt_rm[1]].swap_remove(rmj); // supprime l'enfant du parent

    }

    pub fn edge_swap_random(&mut self, prng: &mut Prng, tarjan_solver: &TarjanSolver, edges: &Vec<[usize; 2]>) -> bool {
        let ei = (prng.next_u64() % edges.len() as u64) as usize;
        println!("{:?}", edges[ei]);
        let dt = self.edge_removable_for_swap(ei, tarjan_solver, edges);
        let [resu, resv] = &dt;
        if resu.len() == 0 {return false;}
        let k = resu.len() + resv.len() - 2;
        let mut rk = (prng.next_u64() % k as u64) as usize;

        let mut dtrmi = 0;
        let mut dtothi = 1;
        println!("{:?} k={}  {}", dt, k, rk);

        if rk >= resu.len() - 1 {
            rk -= resu.len() - 1;
            dtrmi = 1;
            dtothi = 0;
        }


        self.do_the_edge_swap(&dt[dtrmi], &dt[dtothi], rk);

        true
    }

    pub fn subtree_swap_with_edge(&self, ei: usize, prng: &mut Prng,
        tarjan_solver: &TarjanSolver, edges: &Vec<[usize; 2]>, g: &Graph, tree_buf: &mut Graph) -> bool
    {
        let dt = self.edge_removable_for_swap(ei, tarjan_solver, edges);
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

    pub fn subtree_swap_with_random_edge(&self, prng: &mut Prng, 
        tarjan_solver: &TarjanSolver, edges: &Vec<[usize; 2]>, g: &Graph, tree_buf: &mut Graph) -> bool {
        
        let ei = (prng.next_u64() % edges.len() as u64) as usize;
        self.subtree_swap_with_edge(ei, prng, tarjan_solver, edges, g, tree_buf)
    }

    // critical path subtree swap
    pub fn shuffle_tree_and_random_leaf(&mut self, prng: &mut Prng) -> usize {
        for children in self.children.iter_mut() {
            children.shuffle(prng);
        }

        let mut v = self.root;

        loop {
            if self.children[v].is_empty() {
                return v
            }

            v = self.children[v][0];
        }
    }

    pub fn get_critical_path(&mut self, prng: &mut Prng, tree_buf: &mut Graph) -> Vec<usize> {
        let mut ans = vec![];

        let leaf = self.shuffle_tree_and_random_leaf(prng);
        //println!("{}", leaf);
        self.fill_graph(tree_buf);

        let leaf2 = tree_buf.bfs_further_vertex(leaf, false, &mut vec![]);
        //println!("{}", leaf2);
        let _ = tree_buf.bfs_further_vertex(leaf2, true, &mut ans);

        ans
    }

    pub fn subtree_swap_with_random_critical_path(&mut self, prng: &mut Prng, g: &Graph, tree_buf: &mut Graph) {
        let cp = self.get_critical_path(prng, tree_buf);
        self.subtree_swap_with_vertices(prng, &cp, g, tree_buf)
    }
    
    fn subtree_swap_with_vertices(&self, prng: &mut Prng, vertices: &Vec<usize>,
        g: &Graph, tree_buf: &mut Graph)
    {
        

        let mut covered_vertices = vec![false; self.n];
        //let vertices = self.get_critical_path(prng, tree_buf);
        tree_buf.clear();
        for &v in vertices {
            covered_vertices[v] = true;
        }

        println!("{:?}", vertices);

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

        for (u, children) in self.children.iter().enumerate() {
            for &v in children.iter() {
                if !covered_vertices[u] || !covered_vertices[v] {
                    tree_buf.add_edge_unckecked(u, v);
                }
            }
        }

        

    }


}


impl Graph {
    pub fn bfs_further_vertex(&self, u: usize, construct_path: bool, path: &mut Vec<usize>) -> usize {
        static mut QUEUE: [(usize, u32); N] = [(0, 0); N];
        
        unsafe{QUEUE[0] = (u, 0)};

        let mut previous = vec![-1; self.n];
        previous[u] = -2;

        let mut current_furthest = (u, 0);

        let mut i = 0;
        let mut j = 1;

        while i < j {
            unsafe{
                let (v, d) = QUEUE[i];
                //println!("{} {}", v, d);
                if current_furthest.1 < d {
                    current_furthest = (v, d);
                }

                i += 1;

                for &nv in self.get_neighbors(v) {
                    if previous[nv] == -1 {
                        previous[nv] = v as isize;
                        QUEUE[j] = (nv, d + 1);
                        j += 1;
                    }
                }
            }
        }

        if construct_path {
            let mut v = current_furthest.0 as isize;
            while v != -2 {
                assert!(v >= 0);
                path.push(v as usize);
                v = previous[v as usize];
            }
        }

        current_furthest.0
    }

}



