use rand_xoshiro::Xoshiro256PlusPlus;

use crate::my_rand::irwin_hall;


#[derive(Debug, Clone, Copy)]
pub struct Par<T> {
    pub(crate) val: T,
    pub(super) bounds: [T; 2],
    pub(super) std: T
}

impl Par<f64> {
    pub fn derive(&self, prng: &mut Xoshiro256PlusPlus, p: f64) -> Par<f64> {
        let mut prev = self.clone();
        prev.val += irwin_hall(prng) * self.std / p;
        prev.val = prev.val.clamp(self.bounds[0], self.bounds[1]);

        prev
    }

    pub fn new_free(val: f64) -> Par<f64> {
        Par { val, bounds: [0.0, 10.0], std: 0.0 }
    }
}


