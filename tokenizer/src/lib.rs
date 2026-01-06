use compact_str::CompactString;
use fancy_regex::Regex;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::fs;

fn main() {
    // IO
    let file_path = "./dummy.txt";
    let contents = fs::read_to_string(file_path).expect("File error");

    // // word count
    // let mut dict: HashMap<CompactString, u8> = HashMap::new();

    // const PAT: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
    // let _s = "I've done my homework!";
    // let re = Regex::new(&PAT).unwrap();

    // for word in re.find_iter(&contents) {
    //     let w = CompactString::new(word.unwrap().as_str());
    //     let w_count = dict.entry(w).or_insert(0);
    //     *w_count += 1;

    // }

    // let mut unique_words: Vec<CompactString> = Vec::new();

    // for (word, count) in dict {
    //     // println!("|{word}| {count}");
    //     unique_words.push(word.clone());
    //     // println!(r"{word} {:?}", word.as_bytes());
    //     println!("{:?}", word.as_bytes());
    // }

    // println!("Unique words {}", unique_words.len());

    type BytePair = (u8, u8);

    #[derive(Debug, Eq, PartialEq)]
    struct PairCount {
        pair: BytePair,
        count: u32,
    }

    impl Ord for PairCount {
        fn cmp(&self, other: &Self) -> Ordering {
            // self.count.cmp(&other.count)
            (self.count, self.pair).cmp(&(other.count, other.pair))
        }
    }

    impl PartialOrd for PairCount {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut pair_count_dict: HashMap<BytePair, u32> = HashMap::new();
    let mut byte_sequence: Vec<u8> = vec![
        32, 115, 111, 112, 104, 105, 115, 116, 105, 99, 97, 116, 101, 100,
    ];
    byte_sequence = contents.as_bytes().to_vec();
    // println!("Text Length {}", contents.len());
    // byte_sequence = byte_sequence[..10].to_vec();

    let mut i = 0;
    while i < byte_sequence.len() - 1 {
        // println!("{}, {}", byte_sequence[i], byte_sequence[i+1]);
        let pair = (byte_sequence[i], byte_sequence[i + 1]);
        let pair_count = pair_count_dict.entry(pair).or_insert(0);
        *pair_count += 1;
        i += 1;
    }

    let mut pair_count_heap: BinaryHeap<PairCount> = BinaryHeap::new();

    for (pair, count) in pair_count_dict {
        // println!("{:?}, {}", pair, count);
        pair_count_heap.push(PairCount { pair, count })
    }

    let l = pair_count_heap.len() / 2;
    for i in 0..l {
        let pair_count = pair_count_heap.pop().unwrap();
        // println!("{:?}, {}", pair_count.pair, pair_count.count);
    }
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

