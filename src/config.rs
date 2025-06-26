use std::collections::HashMap;

use serde::{Deserialize, Serialize};



#[derive(Serialize, Deserialize)]
pub struct AntColonyProfile {
    pub c: f64,
    pub evap: f64,
    pub seed: u64,
    pub w: f64,
    pub k: u64,
    pub ic: u64
}

#[derive(Serialize, Deserialize)]
pub enum Profile {
    DistoApprox,
    AntColony(AntColonyProfile),
    NeighborhoodTest,
    VNSvsACO(AntColonyProfile)
}

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub profiles: HashMap<String, Profile>
}

#[derive(Serialize, Deserialize)]
pub enum Answer {
    DistoApprox(usize, usize, f64),            // bon, pas bon, ratio
    AntColony(f64, Vec<f64>, AntColonyProfile) // disto, trace
}




