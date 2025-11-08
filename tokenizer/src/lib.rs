use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;

use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use rayon::prelude::*;

const PRE_TOKENIZER_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

type BytePair = (u32, u32);

#[pyclass]
pub struct Tokenizer {
    pub merges: StdHashMap<BytePair, u32>,
    pub pattern: String,
}

#[derive(Clone, Debug)]
struct Word {
    tokens : Vec<u32>,
}

impl Word {
    #[inline]
    fn new(tokens: Vec<u32>) -> Self {
        Self { tokens }
    }

    #[inline]
    fn pairs<'a>(&'a self) -> impl Iterator<Item = BytePair> + 'a {
        self.tokens.windows(2).map(|w| (w[0], w[1]))
    }   


    // TODO
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn dummy(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b +2).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dummy, m)?)?;
    Ok(())
}

