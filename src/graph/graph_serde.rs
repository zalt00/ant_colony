use std::{fs::File, io::{BufRead, BufReader, BufWriter}};

use bincode::{Decode, Encode};
use rand::SeedableRng;

use crate::{graph::{compressed_graph::CompressedGraph, graph_core::GraphCore, graph_generator::GraphRng, MatGraph}, my_rand::Prng, utils::PairExt};


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
            &mut BufWriter::new( File::create(path).expect("welp")),
            bincode::config::standard()
        ).expect("welp2");
    }

    pub fn load(path: &str) -> Data {
        bincode::decode_from_std_read(&mut BufReader::new(File::open(path).expect("beuh")), bincode::config::standard()).expect("wee")
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
    pub fn from_graph<T: GraphCore>(g: &T, compute_ebc: bool, compute_dm: bool) -> GraphData {

        let n = g.vertex_count();
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
                if line.contains("#") {continue};
                let values: Vec<usize> = line.split(char::is_whitespace).map(|xs| {xs.parse::<usize>().unwrap()}).collect();
                edges.push([values[0], values[1]].sorted())
            } else {
                panic!()
            }
        }
        edges.sort();
        let mut n = 0;
        let mut prev = [usize::MAX, usize::MAX];
        let mut i = 0;
        while i < edges.len() {
            let e = edges[i];
            let [u, v] = e;
            if e == prev {
                edges.swap_remove(i);
            } else {
                i += 1;
            }

            n = n.max(u).max(v);
            prev = [u, v]
        }

        n += 1;
        let m = edges.len();

        let gdt = GraphData { label: String::new(), n, m, edges, ebc: None, dist_matrix: None };
        let g: CompressedGraph = gdt.to_graph();

        // connexify
        let g = g.connectify();
          
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
            if self.n > 20000 {
                println!("dm too big, ignoring computation.");
                vec![]
            } else {
                println!("compute dm");
                g.get_dist_matrix()
            }


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


