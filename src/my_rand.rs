use rand::{seq::SliceRandom, RngCore};
use rand_xoshiro::Xoshiro256PlusPlus;

pub type Prng = Xoshiro256PlusPlus;


pub fn my_rand(prng: &mut Xoshiro256PlusPlus) -> f64 {
    prng.next_u64() as f64 / u64::MAX as f64
}

pub fn radamacher(prng: &mut Xoshiro256PlusPlus) -> f64 {
    prng.next_u64() as f64 / u64::MAX as f64 - 0.5
}

pub fn irwin_hall(prng: &mut Xoshiro256PlusPlus) -> f64 {
    let mut s = 0.0;
    for _ in 0..12 {
        s += my_rand(prng);
    }
    s - 6.0
}

pub fn random_permutation(n: usize, prng: &mut Prng) -> Vec<usize> {
    let mut perm = (0..n).collect::<Vec<usize>>();
    perm.shuffle(prng);
    perm
}




