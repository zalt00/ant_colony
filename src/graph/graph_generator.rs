use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

use bincode::{Decode, Encode};
use rand::{seq::SliceRandom, RngCore, SeedableRng};

use crate::graph::graph_core::GraphCore;
use crate::my_rand::Prng;
use crate::{graph::{MatGraph, RootedTree}, utils::Uf};


pub trait GraphRng: GraphCore {

    fn random_tree(n: usize, prng: &mut Prng) -> Self where Self: Sized {
        let mut edges = Vec::with_capacity(n - 1);
        let mut covered_vertices = vec![];
        let mut uncovered_vertices: Vec<usize> = (0..n).collect();
        
        let u = (prng.next_u64() % n as u64) as usize;
        uncovered_vertices.swap_remove(u);
        covered_vertices.push(u);

        for _ in 0..(n-1) {
            let i = (prng.next_u64() % uncovered_vertices.len() as u64) as usize;
            let j = (prng.next_u64() % covered_vertices.len() as u64) as usize;

            let u = covered_vertices[j];
            let v = uncovered_vertices.swap_remove(i);
            covered_vertices.push(v);
            edges.push([u, v])
        }

        Self::from_edges(n, &edges)
    }

    fn random_graph(n: usize, m: usize, prng: &mut Prng) -> Self where Self: Sized {
        let t = Self::random_tree(n, prng);
        let mut edges = Vec::with_capacity(m);

        let mut adj_mat = HashSet::with_capacity(m);
        for [u, v] in t.get_edges() {
            adj_mat.insert([u.min(v), u.max(v)]);
            edges.push([u.min(v), u.max(v)])
        }
        for _iter_id in 0..m {
            if _iter_id % 1000000 == 0 && cfg!(feature="verbose") {
                println!("{}M/{}M", _iter_id / 1000000, m/1000000);
            }
            let u = (prng.next_u64() % n as u64) as usize;
            let v = (prng.next_u64() % n as u64) as usize;
            //println!("{} {}", u, v);
            let e = [u.min(v), u.max(v)];
            if !adj_mat.contains(&e) && u != v {
                edges.push(e);
                adj_mat.insert(e);
            }
        }

        Self::from_edges(n, &edges)

    }

    fn to_dot(&self, fname: &str) {
        // 2) Détection du nombre de nœuds nécessaire (max index + 1)
        let edges = self.get_edges();
        let node_count = self.vertex_count();

        // 3) Création d'un graphe non orienté, poids () sur les arêtes
        let mut g: petgraph::graph::Graph<usize, (), petgraph::Undirected> = petgraph::graph::Graph::new_undirected();

        // 4) Ajout des nœuds, étiquetés ici par leur index
        let node_indices: Vec<petgraph::graph::NodeIndex> = (0..node_count)
            .map(|i| g.add_node(i))
            .collect();

        // 5) Ajout des arêtes depuis `edges`
        for &[u, v] in &edges {
            // on suppose 0 <= u, v < node_count
            g.add_edge(node_indices[u], node_indices[v], ());
        }

        // 6) Génération et affichage du DOT (sans étiquette sur les arêtes)
        let dot = petgraph::dot::Dot::with_config(&g, &[petgraph::dot::Config::EdgeNoLabel]);
        let mut file = File::create(fname).expect("wee");
        writeln!(file, "{dot:?}").expect("beuh");
    }

    fn is_connected(&self) -> bool {
                let edges = self.get_edges();
        let node_count = self.vertex_count();

        // 3) Création d'un graphe non orienté, poids () sur les arêtes
        let mut g: petgraph::graph::Graph<usize, (), petgraph::Undirected> = petgraph::graph::Graph::new_undirected();

        // 4) Ajout des nœuds, étiquetés ici par leur index
        let node_indices: Vec<petgraph::graph::NodeIndex> = (0..node_count)
            .map(|i| g.add_node(i))
            .collect();

        // 5) Ajout des arêtes depuis `edges`
        for &[u, v] in &edges {
            // on suppose 0 <= u, v < node_count
            g.add_edge(node_indices[u], node_indices[v], ());
        }

        // 6) Génération et affichage du DOT (sans étiquette sur les arêtes)
        petgraph::algo::connected_components(&g) == 1
    }

    fn is_connected_without(&self, edge: [usize; 2]) -> bool {
                let edges = self.get_edges();
        let node_count = self.vertex_count();

        // 3) Création d'un graphe non orienté, poids () sur les arêtes
        let mut g: petgraph::graph::Graph<usize, (), petgraph::Undirected> = petgraph::graph::Graph::new_undirected();

        // 4) Ajout des nœuds, étiquetés ici par leur index
        let node_indices: Vec<petgraph::graph::NodeIndex> = (0..node_count)
            .map(|i| g.add_node(i))
            .collect();

        // 5) Ajout des arêtes depuis `edges`
        for &[u, v] in &edges {
            if u.min(v) != edge[0].min(edge[1]) || u.max(v) != edge[0].max(edge[1]) {
                g.add_edge(node_indices[u], node_indices[v], ());

            }
            // on suppose 0 <= u, v < node_count
        }

        // 6) Génération et affichage du DOT (sans étiquette sur les arêtes)
        petgraph::algo::connected_components(&g) == 1
    }

    fn random_subtree(&self, prng: &mut Prng) -> RootedTree where Self: Sized {
        let mut uf = Uf::init(self.vertex_count());

        let mut edges = self.get_edges();
        edges.shuffle(prng);

        let n = self.vertex_count();
        let mut tree_edges = Vec::with_capacity(n - 1);

        let mut m = 0;
        let mut i = 0;
        while m < n - 1 {
            let [u, v] = edges[i];

            if uf.find(u) != uf.find(v) {
                uf.union(u, v);

                tree_edges.push([u, v]);
                m += 1;
            }


            i += 1;
        }

        let root = (prng.next_u64() % n as u64) as usize;

        RootedTree::from_graph(&Self::from_edges(n, &tree_edges), root)
    }

    fn fill_clique(edges: &mut Vec<[usize; 2]>, i: usize, j: usize) {
        // [i, j[

        for s in i..(j-1) {
            for t in (s+1)..j {
                edges.push([s, t]);
            }
        }
    }

    fn clique_cycle(clique_count: usize, clique_size: usize) -> Self where Self: Sized {
        let n = clique_count * clique_size;
        let mut edges = Vec::new();

        for i in (0..n).step_by(clique_size) {
            Self::fill_clique(&mut edges, i, i + clique_size);
            edges.push([i, (i+clique_size) % n])
        }
        Self::from_edges(n, &edges)
    }

    fn clique_cycle_mindisto_tree(clique_count: usize, clique_size: usize) -> Self where Self: Sized {
        let n = clique_count * clique_size;
        let mut edges = Vec::with_capacity(n - 1);

        for i in (0..n).step_by(clique_size) {
            for j in (i+1)..(i+clique_size) {
                edges.push([i, j]);
            }
            if i > 0 {
                edges.push([i, (i + clique_size) % n]);
            }
        }
        Self::from_edges(n, &edges)
    }

    fn renumber(&self, permutation: &Vec<usize>) -> Self where Self: Sized {
        let base_edges = self.get_edges();
        let mut edges = Vec::with_capacity(base_edges.len());
        //println!("len: {}", self.n);
        for &[u, v] in &base_edges {
            //println!("{} {}", u, v);
            edges.push([permutation[u], permutation[v]]);
        } 

        Self::from_edges(self.vertex_count(), &edges)

    }




}


#[derive(Encode, Decode)]
pub struct Data {
    pub n_samples: usize,
    pub samples: Vec<GraphData>
}

impl Data {
    pub fn generate_samples(n_samples: usize, n: usize, m: usize, seed: u64) -> Data {
        let mut samples = vec![];
        let mut prng = Prng::seed_from_u64(seed);
        for sample_id in 1..=n_samples { 
            println!("generating sample {}", sample_id);
            let g = MatGraph::random_graph(n, m, &mut prng);
            samples.push(GraphData::from_graph(&g, false, false));
        }

        Data { n_samples, samples }
    }

    pub fn save(&self, path: &str) {
        
        bincode::encode_into_std_write(
            self, 
            &mut File::create(path).expect("welp"),
            bincode::config::standard()
        ).expect("welp2");
    }

    pub fn load(path: &str) -> Data {
        bincode::decode_from_std_read(&mut File::open(path).expect("beuh"), bincode::config::standard()).expect("wee")
    }

    pub fn load_benchmark_directory(path: &str) -> Data {
        let mut samples = vec![];
        for entry in glob::glob(&format!("{}/**/*.txt", path)).expect("wee") {
            println!("loading: <{:?}>", entry.as_ref().expect("baa"));
            samples.push(GraphData::from_text_file(entry.expect("bouuuh").to_str().unwrap()))
        }

        Data { n_samples: samples.len(), samples }
    }

}

#[derive(Encode, Decode)]
pub struct GraphData {
    pub label: String,
    pub n: usize,
    pub m: usize,
    pub edges: Vec<[usize; 2]>,
    ebc: Option<Vec<f64>>,
    dist_matrix: Option<Vec<u32>>

}

impl GraphData {
    pub fn from_graph(g: &MatGraph, compute_ebc: bool, compute_dm: bool) -> GraphData {

        let n = g.n;
        let edges = g.get_edges();
        let m = edges.len();
        println!("computing ebc..");
        let ebc = if compute_ebc {Some(g.get_edge_betweeness_centrality())} else {None};

        println!("computing dist matrix..");
        let dist_matrix = if compute_dm {Some(g.get_dist_matrix())} else {None};

        println!("done.");

        GraphData { label: "unlabeled".to_string(), n, m, edges, ebc, dist_matrix }
    }

    pub fn to_graph<T: GraphCore>(&self) -> T {
        T::from_edges(self.n, &self.edges)
    }

    pub fn from_text_file(path: &str) -> GraphData {
        let file = File::open(path).expect("welp");

        let mut edges = vec![];
        
        let reader = BufReader::new(file);
        for line_res in reader.lines() {
            if let Ok(line) = line_res {
                let values: Vec<usize> = line.split(' ').map(|xs| {xs.parse::<usize>().unwrap()}).collect();
                edges.push([values[0], values[1]])
            } else {
                panic!()
            }
        }

        let mut n = 0;
        for &[u, v] in &edges {
            n = n.max(u).max(v);
        }
        n += 1;
        let m = edges.len();

        let gdt = GraphData { label: String::new(), n, m, edges, ebc: None, dist_matrix: None };
        let g = gdt.to_graph();
          
        let mut gdt2 = GraphData::from_graph(&g, false, false);
        gdt2.label = path.to_string();
        gdt2
    }

    pub fn graph_ebc_dist_matrix<T: GraphCore>(&self) -> (T, Vec<f64>, Vec<u32>) {
        let g = self.to_graph::<T>();

        let ebc = if let Some(ebc) = &self.ebc {
            ebc.clone()
        } else {
            if cfg!(feature = "need_ebc") {
                println!("compute ebc");
                g.get_edge_betweeness_centrality()
            } else {
                println!("ignore ebc computation");
                vec![]
            }
        };

        let dm = if let Some(dm) = &self.dist_matrix {
            dm.clone()
        } else {
            println!("compute dm");
            g.get_dist_matrix()

        };

        (g, ebc, dm)

    }

    pub fn graph_ebc_dist_matrix_force<T: GraphCore>(&self) -> (T, Vec<f64>, Vec<u32>) {
        let g: T = self.to_graph();

        let ebc = if let Some(ebc) = &self.ebc {
            ebc.clone()
        } else {
            println!("compute ebc");
            g.get_edge_betweeness_centrality()
        };

        let dm = if let Some(dm) = &self.dist_matrix {
            dm.clone()
        } else {
            g.get_dist_matrix()
        };

        (g, ebc, dm)

    }


    pub fn graph_dist_matrix<T: GraphCore>(&self) -> (T, Vec<u32>) {
        let g: T = self.to_graph();

        let dm = if let Some(dm) = &self.dist_matrix {
            dm.clone()
        } else {
            g.get_dist_matrix()
        };

        (g, dm)
    }

}








