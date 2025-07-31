
use std::{collections::HashMap, hash::Hash, ops::Deref};

use crate::{graph::RootedTree, graph::graph_core::GraphCore};

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
    pub n: usize,
    uf: Uf,
    mark: Vec<bool>,
    ancestors: Vec<usize>,
    results: CompressedVecVec<usize>,
}

impl TarjanSolver {

    #[cfg(feature = "need_tarjan")]
    pub fn new<T: GraphCore>(n: usize, g: &T) -> TarjanSolver {
        let results = g.get_edges_compressed_vecvec(usize::MAX);
 
        TarjanSolver { n, uf: Uf::init(n), mark: vec![false; n], ancestors: vec![0; n], results}
    }

    #[cfg(not(feature = "need_tarjan"))]
    pub fn new<T: GraphCore>(n: usize, _g: &T) -> TarjanSolver {
        TarjanSolver { n, uf: Uf::init(0), mark: vec![], ancestors: vec![], results: Default::default() }
    }

    fn reset(&mut self) {
        self.uf.reset();
        self.mark.fill(false);
        self.results.fill(usize::MAX);
    }

    fn _launch_from<T: GraphCore>(&mut self, u: usize, tree: &RootedTree, g: &T, children: &CompressedVecVec<usize>) {
        self.ancestors[u] = u;
        for v in children.get_slice(u).iter() {
            self._launch_from(*v, tree, g, children);
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
                self.results.get_slice_mut(u)[i] = lca;
            }
        }
    }

    pub fn launch<T: GraphCore>(&mut self, tree: &RootedTree, g: &T, children: &CompressedVecVec<usize>) -> &CompressedVecVec<usize> {
        if cfg!(not(feature = "need_tarjan")) {panic!()};

        self.reset();
        self._launch_from(tree.get_root(), tree, g, children);

        &self.results
    }

    pub fn get_results(&self) -> &CompressedVecVec<usize> {
        &self.results
    }


}

pub trait PairIterExt: Iterator<Item = [Self::It; 2]> 
    where <Self as PairIterExt>::It: Copy+Deref<Target = Self::Tar>,
          <Self as PairIterExt>::Tar: Copy+Ord 
{
    type It;
    type Tar;
    fn sorted_pairs(&mut self) -> PairIterator<Self> where Self: Sized {
        PairIterator { iterator: self }
    }
}

pub struct PairIterator<'a, T: PairIterExt> {
    iterator: &'a mut T
}

impl<'a, T: PairIterExt> Iterator for PairIterator<'a, T> {
    type Item = [T::Tar; 2];

    fn next(&mut self) -> Option<Self::Item> {
        if let Some([u, v]) = self.iterator.next() {
            Some([u.min(*v), u.max(*v)])
        } else {
            None
        }
    }
}

impl<I2: Copy+Ord, I: Copy+Deref<Target = I2>, T: Iterator<Item = [I; 2]>> PairIterExt for T {
    type It = I;
    type Tar = I2;
}

pub trait PairExt where Self::T: Ord+Copy {
    type T;
    fn sorted(self) -> Self;
}

impl<T2: Ord+Copy> PairExt for [T2; 2] {
    type T = T2;
    fn sorted(self) -> Self {
        if self[0] < self[1] {
            self
        } else {
            [self[1], self[0]]
        }
    }
}

pub trait HashMapExt {
    type T;
    fn inverse(self) -> Self::T;
}

impl<A, B> HashMapExt for HashMap<A, B> where B: Hash+Eq {
    type T = HashMap<B, A>;

    fn inverse(mut self) -> Self::T {
        let mut hmap2 = HashMap::with_capacity(self.len());
        for (k, v) in self.drain() {
            hmap2.insert(v, k);
        }
        hmap2
    }
}

pub struct Iter2Elements<'a, I: Iterator> {
    i: &'a mut I,
    prev: Option<I::Item>
}

impl<'a, I: Iterator> Iterator for Iter2Elements<'a, I> where I::Item: Copy+Clone{
    type Item = (I::Item, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let elem = self.i.next()?;
        if let Some(prev) = self.prev {
            self.prev = Some(elem);
            Some((prev, elem))
        } else {
            self.prev = Some(elem);
            self.next()
        }
    }
}

pub trait IterExt {
    type Iter: Iterator;
    fn iter2(&mut self) -> Iter2Elements<Self::Iter>;
}

impl<I: Iterator> IterExt for I {
    type Iter = I;

    fn iter2(&mut self) -> Iter2Elements<Self::Iter> {
        Iter2Elements { i: self, prev: None }
    }
}

pub trait IterCountExt {
    type Iter: Iterator;

    fn count_unique_elements(&mut self) -> HashMap<<Self::Iter as Iterator>::Item, usize> where <<Self as IterCountExt>::Iter as Iterator>::Item: Hash+Eq;
}

impl <I: Iterator> IterCountExt for I where <I as Iterator>::Item: Hash {
    type Iter = I;

    fn count_unique_elements(&mut self) -> HashMap<<Self::Iter as Iterator>::Item, usize> where <<Self as IterCountExt>::Iter as Iterator>::Item: Hash+Eq {
        let mut hmap = HashMap::new();

        for elem in self {
            hmap.entry(elem).and_modify(|x| {*x += 1}).or_insert(1);
        }

        hmap
    }
}


#[derive(Clone, Default)]
pub struct CompressedVecVec<T: Copy+Clone> {
    idx: Vec<usize>,
    data: Vec<T>
}

impl<T: Copy+Clone> CompressedVecVec<T> {
    pub const fn len(&self) -> usize {self.data.len()}
    pub fn new(init_value: T, n: usize, degrees: &Vec<usize>) -> Self {

        let mut idx = vec![0; n+1];
        for i in 1..=n {
            idx[i] = idx[i-1] + degrees[i-1];
        }

        let data = vec![init_value; idx[n]];

        Self { idx, data }
    }

    #[cfg(feature = "disable_vecvec_bound_check")]
    pub fn get_slice(&self, i: usize) -> &[T] {
        &self.data[self.idx[i]..]//self.idx[i+1]]
    }

    #[cfg(not(feature = "disable_vecvec_bound_check"))]
    pub fn get_slice(&self, i: usize) -> &[T] {
        &self.data[self.idx[i]..self.idx[i+1]]
    }


    #[cfg(feature = "disable_vecvec_bound_check")]
    pub fn get_slice_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.data[self.idx[i]..]//self.idx[i+1]]
    }

    #[cfg(not(feature = "disable_vecvec_bound_check"))]
    pub fn get_slice_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.data[self.idx[i]..self.idx[i+1]]
    }
    pub fn fill(&mut self, val: T) {
        self.data.fill(val);
    }
    

}


pub fn renumber_edges(edges: &mut Vec<[usize; 2]>) -> HashMap<usize, usize> {
    // old vertex id -> new vertex id
    let mut hmap = HashMap::with_capacity(edges.len() * 2);
    let mut i = 0;
    for [u, v] in edges {
        hmap.entry(*u).or_insert_with(|| {let j = i; i += 1; j});
        hmap.entry(*v).or_insert_with(|| {let j = i; i += 1; j});

        *u = hmap[u];
        *v = hmap[v];
    }
    hmap
}
pub fn renumber_edges2(edges: &mut [[usize; 2]], old2new: &mut [usize], new2old: &mut [usize]) {
    // old vertex id -> new vertex id
    let mut i = 0;
    for &[u, v] in edges.iter() {
        old2new[u] = usize::MAX;
        old2new[v] = usize::MAX;
    }

    for e in edges {
        let [u, v] = *e;
        if old2new[u] == usize::MAX {
            old2new[u] = i;
            i += 1;
        }

        if old2new[v] == usize::MAX {
            old2new[v] = i;
            i += 1;
        }

        e[0] = old2new[u];
        e[1] = old2new[v];

        new2old[old2new[u]] = u;
        new2old[old2new[v]] = v;

    }
}


