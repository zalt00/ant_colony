use crate::graph::{graph_core::GraphCore, N};







pub struct ParentTree {
    n: usize,
    parent: Vec<usize>,
    arity: Vec<usize>,
    leaves: Vec<usize>,
    depths: Vec<usize>,
    root: usize
}

impl ParentTree {
    pub fn new(n: usize, root: usize) -> ParentTree {
        let parent = vec![usize::MAX; n];

        let mut depths = vec![usize::MAX; n];
        depths[root] = 0;

        ParentTree { n, parent, leaves: Vec::with_capacity(n), arity: vec![0; n], depths, root }
    }

    pub fn add_child(&mut self, u: usize, v: usize) {
        self.parent[v] = u;
        self.arity[u] += 1;
        self.depths[v] = self.depths[u] + 1;
    }

    pub fn update_leaves(&mut self) {
        for u in (0..self.n).filter(|&x|{self.arity[x] == 0}) {
            self.leaves.push(u)
        }
    }

    pub fn recompute_arity(&mut self) {
        for u in 0..self.n {
            if u != self.root {
                self.arity[self.parent[u]] += 1;
            }
        }
    }

    pub fn from_graph<T: GraphCore>(g: &T, root: usize) -> ParentTree {
        let mut tree = Self::new(g.vertex_count(), root);

        let mut visited = vec![false; g.vertex_count()];

        fn dfs<T: GraphCore>(u: usize, g: &T, visited: &mut Vec<bool>, tree: &mut ParentTree) {
            visited[u] = true;

            for &v in g.get_neighbors(u) {
                if !visited[v] {
                    tree.add_child(u, v);
                    dfs(v, g, visited, tree);
                }
            }

        }
        dfs(root, g, &mut visited, &mut tree);
        tree.update_leaves();
        tree
    }

    pub fn precalcul_sizes(&mut self, u: usize, tab: &mut Vec<u64>) {
        static mut QUEUE: [usize; 50000000] = [0; 50000000];
        let mut i = 0;
        let mut j = self.leaves.len();
        for (t, &u) in self.leaves.iter().enumerate() {
            unsafe{QUEUE[t] = u;}
        }
        while i < j {
            unsafe{
                let u = QUEUE[i];
                tab[u] += 1;

                if u != self.root {
                    if self.arity[self.parent[u]] == 1 {
                        QUEUE[j] = self.parent[u];
                        j+= 1;
                    } else {
                        self.arity[self.parent[u]] -= 1;
                    }

                    tab[self.parent[u]] += tab[u];
                }
                i += 1;
            }
        }
        self.recompute_arity();
    }
    pub fn new_disto_approx2(&mut self) -> u64 {
        let mut size = vec![0; self.n];
        self.precalcul_sizes(self.root, &mut size);
        let mut s3 = 0;
        for u in 0..self.n {
            if self.root == u {
                s3 += self.depths[u] as u64 * (1 + self.n as u64)
            } else {
                let su = size[u];
                let spu = size[self.parent[u]];
                s3 += self.depths[u] as u64 * (1 + self.n as u64 - su * (spu - su + 1))
                        + su * (spu - su - 1);
            }
        }
        s3
    }

}












