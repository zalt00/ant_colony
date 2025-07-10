use pyo3::prelude::*;

pub mod graph;
pub mod my_rand;
pub mod greedy;
pub mod aco2;
pub mod utils;
pub mod config;
pub mod neighborhood;
pub mod annealing;
pub mod trace;
pub mod distorsion_heuristics;
pub mod counters;
pub mod vns;
use rand::SeedableRng;

use crate::distorsion_heuristics::constants;
use crate::graph::graph_core::GraphCore;
use crate::graph::graph_generator::GraphRng;
use crate::graph::MatGraph;
use crate::graph::RootedTree;
use crate::my_rand::Prng;
use crate::utils::TarjanSolver;


#[pyclass]
struct Helper {
    g: MatGraph,
    base_tree: RootedTree,
    tree: RootedTree,
    dataset: Vec<(usize, usize, usize)>,
    edges: Vec<[usize;2]>,
    prng: Prng
}

#[pymethods]
impl Helper {
    #[new]
    fn new() -> Helper {
        let mut h = Helper { g: MatGraph::new_really_empty(),
            base_tree: RootedTree::new_really_empty(),
            tree: RootedTree::new_really_empty(),
            dataset: vec![], edges: vec![], prng: Prng::seed_from_u64(12) };
        h.init();

        h
    }

    fn init(&mut self) {
        let m = 500;
        self.g = MatGraph::random_graph(100, m, &mut self.prng);
        self.tree = self.g.random_subtree(&mut self.prng);
        self.tree.update_parents();

        self.dataset = Vec::with_capacity(m);

        for &[u, v] in &self.g.get_edges() {
            let t = if self.tree.has_edge(u, v) {1} else {0};
            self.dataset.push((u, v, t));
        }
        self.edges = self.g.get_edges();
    }

    fn get_dataset(&self) -> &Vec<(usize, usize, usize)> {
        &self.dataset
    }

    fn recompense(&mut self, i: usize) -> (f64, [usize; 2]) {
        let dist1 = self.tree.heuristic(&self.g, &self.edges, &mut TarjanSolver::new(0, &self.g), &vec![], &vec![]);
        let dt = self.tree.edge_removable_for_swap(i, &self.edges);
        // println!("{:?}", dt);

        let [resu, resv] = &dt;
        if resu.len() == 0 {return (0.0, self.edges[i]);}
        let k = resu.len() + resv.len() - 2;

        let mut best_tree = None;
        let mut best_disto = constants::INF;
        let mut best_edge = [usize::MAX, usize::MAX];

        for mut rk in 0..k {

            let mut dtrmi = 0;
            let mut dtothi = 1;
            // println!("{:?} k={}  {}", dt, k, rk);

            if rk >= resu.len() - 1 {
                rk -= resu.len() - 1;
                dtrmi = 1;
                dtothi = 0;
            }

            let mut t2 = self.tree.clone();
            t2.do_the_edge_swap(&dt[dtrmi], &dt[dtothi], rk);
            let dist2 = t2.heuristic(&self.g, &self.edges, &mut TarjanSolver::new(0, &self.g), &vec![], &vec![]);
            if dist2 < best_disto {
                best_disto = dist2;
                best_edge = [dt[dtrmi][rk], dt[dtrmi][rk + 1]];
                best_tree = Some(t2)
            }
            
            
        }

        self.tree = best_tree.unwrap();
        
        let r = (dist1 as f64 - best_disto as f64) / (dist1 as f64);

        (r, best_edge)
    
    }

    fn reset(&mut self, iter_id: usize) {

        if iter_id > 0 && iter_id % 10 == 0 {
            self.init();
        } else {
            self.tree = self.g.random_subtree(&mut self.prng);
            self.tree.update_parents();


        }

    }



}




/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn mylib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Helper>()?;

    Ok(())
}