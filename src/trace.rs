use serde::{Deserialize, Serialize};




#[derive(Serialize, Deserialize)]
pub struct TraceData {
    best_disto_so_far: f64,
    iter_count: usize,
    elapsed: f64
}

impl TraceData {
    pub fn new(best_disto_so_far: f64, iter_count: usize, elapsed: f64) -> TraceData {
        TraceData { best_disto_so_far, iter_count, elapsed }
    }
}

#[derive(Serialize, Deserialize)]
pub struct TraceResult {
    distorsion: f64,
    trace: Vec<TraceData>
}

impl TraceResult {
    pub fn new(distorsion: f64, trace: Vec<TraceData>) -> Self {
        Self { distorsion, trace }
    }
}

