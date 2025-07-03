
use crate::{compressed_graph::init_compressed_vecvec, graph::{MatGraph, RootedTree}, graph_core::GraphCore};

pub struct SegmentTree (Vec<f64>);

impl SegmentTree {
    pub fn new(n: usize) -> SegmentTree {
        let size = n * 2;
        SegmentTree(vec![0.0; size])
    }

    pub fn get_leaves(&self) -> &[f64] {
        &self.0[self.intern_node_count()..]
    }

    pub const fn size(&self) -> usize {
        self.0.len()
    }

    pub const fn leaf_count(&self) -> usize {
        self.0.len() / 2
    }

    pub const fn intern_node_count(&self) -> usize {
        self.0.len() / 2
    }

    const fn children_count(&self, vi: usize) -> u8 {
        if vi * 2 + 2 < self.size() {
            2
        } else if vi * 2 + 2 == self.size() {
            1
        } else {
            0
        }
    }

    const fn parent(&self, vi: usize) -> usize {
        (vi-1) / 2
    }

    pub fn global_sum(&self) -> f64 {
        self.0[0]
    }

    pub fn update(&mut self, v: usize, val: f64) {
        let mut vi = v + self.intern_node_count();
        let dv = val - self.0[vi];
        self.0[vi] = val;
        while vi != 0 {
            vi = self.parent(vi);
            self.0[vi] += dv;
        }
    }

    pub fn get(&self, v: usize) -> f64 {
        let vi = v + self.intern_node_count();
        self.0[vi]
    }

    fn _smallest_above_from(&self, vi: usize, val: f64) -> usize {
        //println!("{} {}", vi, val);
        let children_count = self.children_count(vi);
        if children_count == 2 {
            //println!("g{} d{}", self.0[vi * 2 + 1], self.0[vi * 2 + 2]);

            let val_left = self.0[vi * 2 + 1];
            if val_left < val {
                self._smallest_above_from(vi * 2 + 2, val - val_left)
            } else {
                self._smallest_above_from(vi * 2 + 1, val)
            }

        } else if children_count == 1 {
            self._smallest_above_from(vi * 2 + 1, val)
        } else {
            debug_assert!(val <= self.0[vi]);
            debug_assert!(vi >= self.intern_node_count());
            vi
        }
    }

    pub fn smallest_above(&self, val: f64) -> usize {
        self._smallest_above_from(0, val) - self.intern_node_count()
    }

    pub fn reset(&mut self) {
        self.0.fill(0.0);
    }
}

pub fn test_segment_tree() {
    let mut sg = SegmentTree::new(5);  // ordre 2, 3, 4, 0, 1

    sg.update(0, 0.2);
    sg.update(1, 0.2);
    sg.update(2, 0.2);
    sg.update(3, 0.2);
    sg.update(4, 0.2);

    assert_eq!(sg.smallest_above(0.1), 2);
    assert_eq!(sg.smallest_above(0.2), 2);
    assert_eq!(sg.smallest_above(0.3), 3);
    assert_eq!(sg.smallest_above(1.0), 1);
    assert_eq!(sg.smallest_above(0.9), 1);
    assert_eq!(sg.smallest_above(0.5), 4);



    sg.update(0, 0.0);
    sg.update(1, 0.0);
    sg.update(2, 0.6);
    sg.update(3, 0.2);
    sg.update(4, 0.2);

    assert_eq!(sg.smallest_above(0.1), 2);
    assert_eq!(sg.smallest_above(0.2), 2);
    assert_eq!(sg.smallest_above(0.3), 2);
    assert_eq!(sg.smallest_above(1.0), 4);
    assert_eq!(sg.smallest_above(0.9), 4);
    assert_eq!(sg.smallest_above(0.5), 2);


}


pub struct Uf(Vec<isize>);

impl Uf {
    pub fn find(&mut self, i: usize) -> Option<isize> {
        let v = self.0.get(i)?;
        if *v < 0 {
            Some(i as isize)
        } else {
            let v2 = *v as usize;
            let c = self.find(v2)?;
            self.0[i] = c;
            Some(c)
        }
    }

    pub fn union(&mut self, i1: usize, i2: usize) -> Option<()> {
        let c1 = self.find(i1)?;
        let c2 = self.find(i2)?;
        let s1 = self.0[c1 as usize];
        let s2 = self.0[c2 as usize];
        if s1 < s2 {  // |c1| > |c2|
            self.0[c2 as usize] = c1;
            self.0[c1 as usize] = s1 + s2;
        } else {
            self.0[c1 as usize] = c2;
            self.0[c2 as usize] = s1 + s2;
        }
        Some(())
    }

    pub fn init(n: usize) -> Uf {
        Uf(vec![-1; n])
    }
    pub fn reset(&mut self) {
        self.0.fill(-1);
    }
}

pub struct TarjanSolver {
    n: usize,
    uf: Uf,
    mark: Vec<bool>,
    ancestors: Vec<usize>,
    results: Vec<usize>,
    results_idx: Vec<usize>
}

impl TarjanSolver {

    #[cfg(feature = "need_tarjan")]
    pub fn new<T: GraphCore>(n: usize, g: &T) -> TarjanSolver {
        let (results_idx, results) = g.get_edges_compressed_vecvec(usize::MAX);
 
        TarjanSolver { n, uf: Uf::init(n), mark: vec![false; n], ancestors: vec![0; n], results, results_idx }
    }

    #[cfg(not(feature = "need_tarjan"))]
    pub fn new<T: GraphCore>(n: usize, _g: &T) -> TarjanSolver {
        TarjanSolver { n, uf: Uf::init(0), mark: vec![], ancestors: vec![], results: vec![], results_idx: vec![] }
    }

    fn reset(&mut self) {
        self.uf.reset();
        self.mark.fill(false);
        self.results.fill(usize::MAX);
    }

    fn _launch_from<T: GraphCore>(&mut self, u: usize, tree: &RootedTree, g: &T) {
        self.ancestors[u] = u;
        for v in tree.get_children(u) {
            self._launch_from(*v, tree, g);
            self.uf.union(u, *v);
            if let Some(c) = self.uf.find(u) {
                self.ancestors[c as usize] = u;
            } else {
                // welp
                println!("uf: {:?}", self.uf.0);
                println!("\n\n");
                println!("{} {}", u, *v);
                
                panic!()
            }
        }
        self.mark[u] = true;

        for (i, v) in g.get_neighbors(u).iter().enumerate() {
            if self.mark[*v] {
                let lca = self.ancestors[self.uf.find(*v).unwrap() as usize];
                self.results[self.results_idx[u] + i] = lca;
            }
        }
    }

    pub fn launch<T: GraphCore>(&mut self, tree: &RootedTree, g: &T) -> (&Vec<usize>, &Vec<usize>) {
        if cfg!(not(feature = "need_tarjan")) {panic!()};

        self.reset();
        self._launch_from(tree.get_root(), tree, g);

        (&self.results_idx, &self.results)
    }

    pub fn get_results(&self) -> (&Vec<usize>, &Vec<usize>) {
        (&self.results_idx, &self.results)
    }


}





