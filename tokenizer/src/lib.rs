use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;

use dary_heap::OctonaryHeap;
use fancy_regex::Regex;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use rayon::prelude::*;
use std::time::Instant;


pub mod merger;

#[pyfunction]
fn train_bpe(
    py: pyo3::Python<'_>,
    iterator: &pyo3::Bound<'_, pyo3::PyAny>,
) -> PyResult<()> {
    // python iterator reference
    let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
        pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
    };

    // container to store strings yielded by python iterator
    let buf_size = 1024;
    let mut buf: Vec<String> = Vec::with_capacity(buf_size);

    let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
        pyo3::Python::attach(|py| {
            buf.clear();

            // rust interface to python iterator, while it holds the GIL token
            let rs_iter = py_iter.bind(py);

            loop {
                if buf.len() >= buf_size {
                    return Ok(false);
                }

                // calls next on the iterator
                let next_str = unsafe {
                    pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(rs_iter.as_ptr()))
                };

                match next_str {
                    Some(text) => {
                        let t: String = text.extract()?;
                        buf.push(t);
                    }
                    None => {
                        if pyo3::PyErr::occurred(py) {
                            return Err(pyo3::PyErr::fetch(py));
                        } else {
                            return Ok(true); // end of iter
                        }
                    }
                }
            }
        })
    };

    log::info!("running pretokenization");
    let mut start = Instant::now();
    let mut num_strings_processed: u64 = 0;

    // prepare regex for pre-tokenization
    let regex_pat = Regex::new(r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+").expect("Regex compilation failed, invalid pattern.");

    let mut pretoken_count_table: AHashMap<CompactString, i32> = AHashMap::new();
    loop {
        let end_of_iter = refill(&mut buf)?;

        if buf.is_empty() && end_of_iter {
            // note even if we reached the end of iterator
            // we still need to process it before termainating
            break;
        }

        num_strings_processed += buf.len() as u64;

        let loop_local_counts: AHashMap<CompactString, i32> = py.detach(|| {
            buf.par_iter()
                .map(|str_seq| {
                    let mut local_count_table: AHashMap<CompactString, i32> = AHashMap::new();
                    for chunk in regex_pat.find_iter(str_seq) {
                        let pretoken = CompactString::from(chunk.expect("Regex Error").as_str());
                        *local_count_table.entry(pretoken).or_default() += 1;
                    }
                    local_count_table
                })
                .reduce(AHashMap::new, |mut hashmap_a, hashmap_b| {
                    for (pretoken, count) in hashmap_b {
                        *hashmap_a.entry(pretoken).or_default() += count;
                    }
                    hashmap_a
                })
        });

        for (pretoken, count) in loop_local_counts {
            *pretoken_count_table.entry(pretoken).or_default() += count;
        }

        if end_of_iter {
            break;
        }
    }

    log::info!("finished pretokenization in {:.3} seconds", start.elapsed().as_secs_f64());
    // by now we have pretoken count in pretoken_count_table
    let num_merges = 2000;

    log::info!("initializing byte pair encoding training");
    let mut account = merger::Account::new(pretoken_count_table);
    account.merge(num_merges);
    Ok(())
}

/// LLM Tokenizer: Byte Pair Encoder in Rust
#[pymodule]
mod bpe_tokenizer {
    use log::info;
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::train_bpe;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();
        Ok(())
    }
}
