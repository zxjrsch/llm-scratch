use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;

use dary_heap::OctonaryHeap;
use fancy_regex::Regex;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use rayon::prelude::*;
use std::time::Instant;


type Token = u32;
type TokenPair = (u32, u32);
type TokenDB = AHashMap<TokenPair, TokenPairInfo>;
type Dictionary = Vec<TokenPair>;

#[derive(Debug)]
struct TokenSequenceInfo {
    seq: Vec<Token>,
    count: u32,
}

impl TokenSequenceInfo {
    #[inline]
    fn pairs(&self) -> impl Iterator<Item = TokenPair> + '_ {
        self.seq.windows(2).map(|bytes| (bytes[0], bytes[1]))
    }
}

#[derive(Debug)]
struct TokenPairInfo {
    count: u32,
    locations: AHashSet<usize>, // index of Account.pretok_seqs which contains pair
}

#[derive(Debug, Eq, PartialEq)]
struct TBMerged {
    count: u32,
    idx: usize, // index to external_pair_count, used to decrease count change after merges
    pair: TokenPair,
    locations: AHashSet<usize>,
}

impl Ord for TBMerged {
    fn cmp(&self, other: &Self) -> Ordering {
        // self.count.cmp(&other.count)
        (self.count, self.pair).cmp(&(other.count, other.pair))
    }
}

impl PartialOrd for TBMerged {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
pub struct Account {
    pretok_seqs: Vec<TokenSequenceInfo>,
    priority_queue: OctonaryHeap<TBMerged>,
    pair_to_idx: AHashMap<TokenPair, usize>, // pair to index of external_pair_count vec
    external_pair_count: Vec<u32>,
    dictionary: Option<Dictionary>,
}

impl Account {
    pub fn new(pretoken_counts: AHashMap<CompactString, i32>) -> Self {
        // let mut pretok_seqs: Vec<TokenSequenceInfo> = Vec::with_capacity(pretoken_counts.len());

        // for (pretok_str, count) in pretoken_counts {
        //     let seq: Vec<Token> = pretok_str
        //         .as_bytes()
        //         .iter()
        //         .map(|&byte| byte as Token)
        //         .collect();

        //     pretok_seqs.push(TokenSequenceInfo{
        //         seq,
        //         count: count as u16
        //     });
        // }

        // Step 1: turn pretokens into token sequences
        let pretok_seqs: Vec<TokenSequenceInfo> = pretoken_counts
            .par_iter()
            .map(|(pretok_str, count)| {
                let seq: Vec<Token> = pretok_str
                    .as_bytes()
                    .iter()
                    .map(|&byte| byte as Token)
                    .collect();

                TokenSequenceInfo {
                    seq,
                    count: *count as u32,
                }
            })
            .collect();

        // Step 2: count token pairs
        log::info!("counting byte pairs");
        let mut start = Instant::now();
        let token_db: TokenDB = pretok_seqs
            .par_iter()
            .enumerate()
            .map(|(idx, info)| {
                let mut local_db: TokenDB = AHashMap::new();
                if info.seq.len() > 1 && info.count > 0 {
                    for pair in info.pairs() {
                        let db_entry = local_db.entry(pair).or_insert_with(|| TokenPairInfo {
                            locations: AHashSet::new(),
                            count: 0,
                        });
                        db_entry.locations.insert(idx);
                        db_entry.count += info.count;
                    }
                }
                local_db
            })
            .reduce(AHashMap::new, |mut db_1, db_2| {
                for (pair, info) in db_2 {
                    let db_entry = db_1.entry(pair).or_insert_with(|| TokenPairInfo {
                        locations: AHashSet::new(),
                        count: 0,
                    });
                    db_entry.locations.extend(info.locations);
                    db_entry.count += info.count;
                }
                db_1
            });
        log::info!("counted byte pairs in {:.3} seconds", start.elapsed().as_secs_f64());

        // Step 3: setup priority queue
        log::info!("setting up merge queue");
        start = Instant::now();
        let mut priority_queue = OctonaryHeap::new();
        let mut external_pair_count = Vec::new();
        let mut pair_to_idx = AHashMap::new(); // index into external pair count vec
        for (pair, info) in token_db {
            let idx = external_pair_count.len();
            external_pair_count.push(info.count);

            assert!(pair_to_idx.len() == idx);
            pair_to_idx.insert(pair, idx);

            priority_queue.push(TBMerged {
                pair,
                idx,
                count: info.count,
                locations: info.locations,
            });
        }
        log::info!("heapified in {:.3} seconds, initialization complete!", start.elapsed().as_secs_f64());

        Account {
            pretok_seqs,
            priority_queue,
            external_pair_count,
            pair_to_idx,
            dictionary: None,
        }
    }

    pub fn merge(&mut self, num_merges: usize) {
        let init_vocab_size = 256;
        assert!(
            num_merges > 0,
            "Invalid values: num_merges {} should be positive",
            num_merges,
        );

        let mut token_dictionary = Vec::with_capacity(num_merges);

        let mut start = Instant::now();
        log::info!("starting byte pair merging");

        let mut num_completed_merges = 0;
        while num_completed_merges < num_merges {
            // next_merge has type TBMerged
            let Some(next_merge) = self.priority_queue.pop() else {
                log::info!(
                    "No more merges remaining, completed {num_completed_merges} out of {num_merges} required merges"
                );
                break;
            };

            // check if count is stale and needs to be updated
            // this is valid since the priority of elements already enqueued can only be decremented due to merges
            // and therefore will be re-enqueued with lower priority than present
            if self.external_pair_count[next_merge.idx] == 0 {
                continue;
            } else if next_merge.count > self.external_pair_count[next_merge.idx] {
                self.priority_queue.push(TBMerged {
                    pair: next_merge.pair,
                    idx: next_merge.idx,
                    count: self.external_pair_count[next_merge.idx],
                    locations: next_merge.locations,
                });
                continue;
            }

            // mint & save new token
            token_dictionary.push(next_merge.pair);
            let new_token = (init_vocab_size + token_dictionary.len()) as u32;
            num_completed_merges += 1;

            if num_completed_merges % 500 == 0 {
                log::info!("Performing merge no. {num_completed_merges}");
            }
            
            let mut new_queue_elements: AHashMap<TokenPair, TBMerged> = AHashMap::new();

            // iterate over all pretokens where the pair occurrs to merge and update counts and heap
            for pretok_idx in next_merge.locations {
                let pretok: &mut TokenSequenceInfo = &mut self.pretok_seqs[pretok_idx];

                let mut merged_seq = Vec::new();
                let mut j = 0;
                while j < pretok.seq.len() {
                    if pretok.seq[j] != next_merge.pair.0
                        || j + 1 == pretok.seq.len()
                        || pretok.seq[j + 1] != next_merge.pair.1
                    {
                        // does not match pair
                        merged_seq.push(pretok.seq[j]);
                        j += 1;
                        continue;
                    }

                    // destroy old token
                    self.external_pair_count[next_merge.idx] = 0;
                    merged_seq.push(new_token);

                    if j > 0 {
                        // a (p q) -> a <|new_tok|> case, need to decrement (a p) and add (a <|new_tok|>) count to be added to heap later
                        let backward_pair: TokenPair = (pretok.seq[j-1], pretok.seq[j]);

                        if let Some(&backward_pair_idx) = self.pair_to_idx.get(&backward_pair) {
                            self.external_pair_count[backward_pair_idx] -= pretok.count;
                        } 
                        
                        // else if backward_pair.0 != new_token && backward_pair.1 != new_token {
                        //     assert!(false, 
                        //         "failed in line \n {:?} \n  while merging pair {:?} \n searching backward_pair {:?} \n {:?} \n j={:?}, new token {:?} \n token dictionary {:?}",
                        //         "failed in self.external_pair_count[backward_pair_idx] -= pretok.count;",
                        //         next_merge.pair,
                        //         backward_pair,
                        //         pretok.seq,
                        //         j,
                        //         new_token,
                        //         &token_dictionary
                        //     );
                        // }

                        let new_pair: TokenPair = (pretok.seq[j - 1], new_token);

                        let mut entry = new_queue_elements.entry(new_pair).or_insert_with(|| {
                            let idx = self.external_pair_count.len();
                            self.external_pair_count.push(0);
                            TBMerged {
                                pair: new_pair,
                                idx,
                                count: 0,
                                locations: AHashSet::new(),
                            }
                        });

                        entry.count += pretok.count;
                        entry.locations.insert(pretok_idx);
                    }

                    if j + 2 < pretok.seq.len() {
                        // (p q) b -> <|new_tok|> b case
                        let fwd_pair: TokenPair = (pretok.seq[j + 1], pretok.seq[j + 2]);

                        if let Some(&fwd_pair_idx) = self.pair_to_idx.get(&fwd_pair) {
                            self.external_pair_count[fwd_pair_idx] -= pretok.count;
                        } 
                        
                        // else {
                        //     assert!(false, 
                        //         "failed in line \n {:?} \n  while merging pair {:?} \n {:?} \n {:?} \n j={:?}, new token {:?} \n token dictionary {:?}",
                        //         "self.external_pair_count[fwd_pair_idx] -= pretok.count;",
                        //         next_merge.pair,
                        //         fwd_pair,
                        //         pretok.seq,
                        //         j,
                        //         new_token,
                        //         &token_dictionary
                        //     );
                        // }

                        let new_pair: TokenPair = (new_token, pretok.seq[j + 2]);

                        let mut entry = new_queue_elements.entry(new_pair).or_insert_with(|| {
                            let idx = self.external_pair_count.len();
                            self.external_pair_count.push(0);

                            TBMerged {
                                pair: new_pair,
                                idx,
                                count: 0,
                                locations: AHashSet::new(),
                            }
                        });

                        entry.count += pretok.count;
                        entry.locations.insert(pretok_idx);
                    }
                    j += 2;
                }

                // update the pretoken sequence with merged sequence
                pretok.seq = merged_seq
            }
            // at this point we have examined all pretokens containing the merge pair
            for (new_pair, tb_merged) in new_queue_elements {
                self.pair_to_idx.insert(new_pair, tb_merged.idx);
                self.external_pair_count[tb_merged.idx] = tb_merged.count;
                self.priority_queue.push(tb_merged);
            }
        }

        self.dictionary = Some(token_dictionary);
        log::info!("finished merging in {:.3} seconds", start.elapsed().as_secs_f64());
    }

}
