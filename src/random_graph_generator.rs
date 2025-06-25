use std::fs::File;

use bincode::{Decode, Encode};
use rand::{seq::SliceRandom, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::{aco2::Uf, graph::{Graph, RootedTree}};


impl Graph {

    pub fn random_tree(n: usize, prng: &mut Xoshiro256PlusPlus) -> Graph {

        let mut g = Graph::new_empty(n);

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
            g.add_edge_unckecked(u, v);
        }

        g
    }

    pub fn random_graph(n: usize, m: usize, prng: &mut Xoshiro256PlusPlus) -> Graph {
        let mut t = Self::random_tree(n, prng);

        let mut adj_mat = vec![false; n*n];
        for [u, v] in t.get_edges() {
            adj_mat[u + n * v] = true;
            adj_mat[v + n * u] = true;
        }

        for _ in 0..m {
            let u = (prng.next_u64() % n as u64) as usize;
            let v = (prng.next_u64() % n as u64) as usize;
            //println!("{} {}", u, v);
            if !adj_mat[u + n * v] && u != v {
                t.add_edge_unckecked(u, v);
                adj_mat[u + n * v] = true;
                adj_mat[v + n * u] = true;
            }
        }

        t

    }

    pub fn to_dot(&self) {
        // 2) Détection du nombre de nœuds nécessaire (max index + 1)
        let edges = self.get_edges();
        let node_count = self.n;

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
        println!("{dot:?}");
    }

    pub fn is_connected(&self) -> bool {
                let edges = self.get_edges();
        let node_count = self.n;

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

    pub fn random_tree2(&self, prng: &mut Xoshiro256PlusPlus) -> RootedTree {
        let mut uf = Uf::init(self.n);

        let mut edges = self.get_edges();
        edges.shuffle(prng);

        let mut t = Graph::new_empty(self.n);

        let mut m = 0;
        let mut i = 0;
        while m < self.n - 1 {
            let [u, v] = edges[i];

            if uf.find(u) != uf.find(v) {
                uf.union(u, v);

                t.add_edge_unckecked(u, v);
                m += 1;
            }


            i += 1;
        }

        let root = (prng.next_u64() % self.n as u64) as usize;

        RootedTree::from_graph(&t, root)
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
        let mut prng = Xoshiro256PlusPlus::seed_from_u64(seed);
        for sample_id in 1..=n_samples { 
            println!("generating sample {}", sample_id);
            let g = Graph::random_graph(n, m, &mut prng);
            samples.push(GraphData::from_graph(&g));
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
}

#[derive(Encode, Decode)]
pub struct GraphData {
    pub n: usize,
    pub m: usize,
    pub edges: Vec<[usize; 2]>,
    pub ebc: Option<Vec<f64>>,
    pub dist_matrix: Option<Vec<u32>>

}

impl GraphData {
    pub fn from_graph(g: &Graph) -> GraphData {
        let n = g.n;
        let edges = g.get_edges();
        let m = edges.len();
        println!("computing ebc..");
        let ebc = Some(g.get_edge_betweeness_centrality());

        println!("computing dist matrix..");
        let dist_matrix = Some(g.get_dist_matrix());

        println!("done.");

        GraphData { n, m, edges, ebc, dist_matrix }
    }

    pub fn to_graph(&self) -> Graph {
        let mut g = Graph::new_empty(self.n);
        for &e in &self.edges {
            g.add_edge_unckecked(e[0], e[1]);
        }

        g
    }
}








