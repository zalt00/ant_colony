

#[cfg(feature = "enable_counters")]
static mut COUNTERS: [u64; 10] = [0; 10];

#[cfg(feature = "enable_counters")]
pub fn incr(i: usize) {
    unsafe {
        COUNTERS[i] += 1;
    }
}

#[cfg(feature = "enable_counters")]
pub fn print_counters() {
    unsafe {
        let vals = COUNTERS;
        println!("counters: {:?}", vals);
    }
}


#[cfg(not(feature = "enable_counters"))]
pub fn incr(_i: usize) {}

#[cfg(not(feature = "enable_counters"))]
pub fn print_counters() {
    println!("counters are disabled.");
}




