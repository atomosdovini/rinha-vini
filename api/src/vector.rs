// 14-dim query vector representation.

pub const D: usize = 14;

#[derive(Clone, Copy, Debug)]
pub struct Query {
    pub v: [f32; D],
    pub no_history: bool, // dims 5, 6 are sentinel -1
}

impl Default for Query {
    fn default() -> Self {
        Query { v: [0.0; D], no_history: false }
    }
}
