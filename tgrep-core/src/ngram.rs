/// Sparse n-gram extraction and hashing.
///
/// Instead of indexing every overlapping 3-byte window (trigrams), we select
/// variable-length n-grams (2–8 bytes) using bigram frequency analysis.
/// This produces ~2N n-grams per N-character document (similar to trigrams)
/// but with longer, more selective tokens that dramatically reduce false
/// positives at query time.
///
/// The algorithm assigns priority values to each 2-byte pair (bigram) using
/// a frequency table. An n-gram boundary occurs where both the starting and
/// ending bigrams have lower priority than all bigrams in between. Two
/// independent priority maps provide redundancy.
///
/// At index time, we extract ALL sparse grams (every possible n-gram the
/// algorithm produces). At query time, we prefer the longest n-grams that
/// cover the search string, minimizing the number of posting lists to
/// intersect.
///
/// Based on the sparse n-gram algorithm from blackbird.
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;

pub type NGramHash = u32;

/// Maximum length of a sparse n-gram in bytes.
pub const MAX_SPARSE_GRAM_SIZE: u32 = 8;

/// Number of most-frequent bigrams to use from the frequency table.
const NUM_FREQUENT_BIGRAMS: usize = 200;

/// Per-ngram masks for a single file (reusing the posting entry format).
#[derive(Debug, Clone, Copy, Default)]
pub struct NGramMasks {
    /// Positional mask: bit i is set if the ngram occurs at offset where offset % 8 == i.
    pub loc_mask: u8,
    /// 8-bit Bloom filter of bytes that immediately follow this ngram in the file.
    pub next_mask: u8,
}

// ---------------------------------------------------------------------------
// NGram encoding
// ---------------------------------------------------------------------------

/// Encode a variable-length byte slice into a u32 NGram hash.
///
/// - For n-grams ≤ 3 bytes: bytes are stored directly in bits 8–31 (no collision).
/// - For n-grams > 3 bytes: FNV-1a hash in bits 8–31 (24-bit hash space per length).
/// - Bits 0–7 always store the length, so different-length n-grams never collide.
pub fn ngram_from_bytes(src: &[u8]) -> NGramHash {
    debug_assert!(!src.is_empty() && src.len() <= MAX_SPARSE_GRAM_SIZE as usize);
    let hash = if src.len() <= 3 {
        let mut h: u32 = 0;
        for &byte in src {
            h = h.wrapping_add(byte as u32);
            h <<= 8;
        }
        h
    } else {
        // FNV-1a hash, keep upper 24 bits
        let mut h: u32 = 0x811c_9dc5;
        for &b in src {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
        h << 8
    };
    hash | src.len() as u32
}

// ---------------------------------------------------------------------------
// Character normalization
// ---------------------------------------------------------------------------

/// Convert a byte to its indexable form: ASCII lowercase, non-ASCII → 0x80.
///
/// This ensures one-byte-per-character representation and case-insensitive
/// indexing. The sparse gram algorithm operates on these normalized bytes.
#[inline]
pub fn byte_to_indexable(b: u8) -> u8 {
    if b.is_ascii() {
        b.to_ascii_lowercase()
    } else {
        0x80
    }
}

/// Convert a byte slice to indexable bytes (in-place semantics, returns new vec).
pub fn text_to_indexable(data: &[u8]) -> Vec<u8> {
    data.iter().map(|&b| byte_to_indexable(b)).collect()
}

// ---------------------------------------------------------------------------
// Bloom filter for next-byte masks
// ---------------------------------------------------------------------------

/// Map a byte to one of 8 Bloom bits for next_mask.
#[inline]
fn next_bit(byte: u8) -> u8 {
    1u8 << (byte.wrapping_mul(0x9E) >> 5 & 0x07)
}

/// Compute the Bloom filter bit for a byte (public, for query-time checks).
#[inline]
pub fn bloom_hash(byte: u8) -> u8 {
    next_bit(byte)
}

// ---------------------------------------------------------------------------
// Murmur1 hash (for bigram tiebreaking)
// ---------------------------------------------------------------------------

fn murmur1_hash(bytes: &[u8]) -> u32 {
    const M: u32 = 0xc6a4_a793;
    let mut h: u32 = (bytes.len() as u32).wrapping_mul(M);

    let chunks_len = bytes.len() / 4 * 4;
    for chunk in bytes[..chunks_len].chunks_exact(4) {
        let k = u32::from_le_bytes(chunk.try_into().unwrap());
        h = h.wrapping_add(k).wrapping_mul(M);
        h ^= h >> 16;
    }
    let tail = &bytes[chunks_len..];
    if !tail.is_empty() {
        let mut tail_bytes = [0u8; 4];
        tail_bytes[..tail.len()].copy_from_slice(tail);
        h = h
            .wrapping_add(u32::from_le_bytes(tail_bytes))
            .wrapping_mul(M);
        h ^= h >> 16;
    }
    h = h.wrapping_mul(M);
    h ^= h >> 10;
    h = h.wrapping_mul(M);
    h ^= h >> 17;
    h
}

fn hash_bigram(gram: (u8, u8)) -> u32 {
    let bytes = [gram.0, gram.1];
    murmur1_hash(&bytes)
}

// ---------------------------------------------------------------------------
// Bigram frequency map
// ---------------------------------------------------------------------------

/// The bigrams file contains null-separated bigram strings sorted by frequency
/// (most frequent first). Generated from large-scale code corpus analysis.
static BIGRAMS_STR: &str = include_str!("bigrams.bin");

static BIGRAM_MAP: OnceLock<HashMap<(u8, u8), (u32, u32)>> = OnceLock::new();

/// Returns a map from bigrams to two priority values:
/// - First: frequency-based (higher = more common, more likely to be inside longer grams)
/// - Second: murmur hash (deterministic tiebreaker)
pub fn get_bigram_map() -> &'static HashMap<(u8, u8), (u32, u32)> {
    BIGRAM_MAP.get_or_init(|| {
        BIGRAMS_STR
            .split('\0')
            .take(NUM_FREQUENT_BIGRAMS)
            .enumerate()
            .filter_map(|(idx, s)| {
                let mut chars = s.chars();
                let gram = match (chars.next(), chars.next()) {
                    (Some(a), Some(b)) => (byte_to_indexable(a as u8), byte_to_indexable(b as u8)),
                    _ => return None,
                };
                // Higher index in frequency table = more common = higher priority value
                // High-priority bigrams are encompassed by longer grams more often
                Some((
                    gram,
                    ((NUM_FREQUENT_BIGRAMS - idx) as u32, hash_bigram(gram)),
                ))
            })
            .collect()
    })
}

/// Look up priority values for a bigram at position `pair[0..2]`.
fn hash_pair(pair: &[u8]) -> (u32, u32) {
    *get_bigram_map().get(&(pair[0], pair[1])).unwrap_or(&(0, 0))
}

// ---------------------------------------------------------------------------
// Internal queue for monotone stack algorithm
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PosState {
    /// 1-based bigram index position.
    index: u32,
    /// Priority value from the bigram map.
    value: u32,
}

// ---------------------------------------------------------------------------
// Sparse gram extraction for INDEXING
// ---------------------------------------------------------------------------

/// Extract all sparse n-grams from a byte slice for indexing.
///
/// The input should already be in indexable form (lowercase ASCII).
/// Returns a deduplicated list of (ngram_hash, masks) pairs.
pub fn extract_sparse_grams_for_indexing(data: &[u8]) -> Vec<(NGramHash, NGramMasks)> {
    if data.len() < 2 {
        return Vec::new();
    }

    let mut result: HashMap<NGramHash, NGramMasks> = HashMap::new();

    // Phase 1: Extract all bigrams
    for idx in 1..data.len() {
        emit_ngram_for_indexing(data, idx as u32, idx as u32, &mut result);
    }

    // Phase 2: For each of the two bigram maps, use monotone queue to find sparse grams
    for is_first in [true, false] {
        let mut queue: VecDeque<PosState> = VecDeque::new();

        for idx in 1..data.len() {
            let (v1, v2) = hash_pair(&data[idx - 1..]);
            let value = if is_first { v1 } else { v2 };

            // Pop front if n-gram would exceed MAX_SPARSE_GRAM_SIZE
            if let Some(front) = queue.front()
                && idx as u32 - front.index + 1 >= MAX_SPARSE_GRAM_SIZE
            {
                queue.pop_front();
            }

            // Pop back elements with >= value, emitting n-grams for each
            while let Some(back) = queue.back() {
                if is_first || back.index + 1 < idx as u32 {
                    emit_ngram_for_indexing(data, back.index, idx as u32, &mut result);
                }
                if back.value < value {
                    break;
                }
                queue.pop_back();
            }

            queue.push_back(PosState {
                index: idx as u32,
                value,
            });
        }
    }

    result.into_iter().collect()
}

/// Emit a single n-gram during indexing, computing masks.
///
/// The n-gram spans `data[begin_index-1 .. end_index+1]`.
fn emit_ngram_for_indexing(
    data: &[u8],
    begin_index: u32,
    end_index: u32,
    result: &mut HashMap<NGramHash, NGramMasks>,
) {
    let start = (begin_index as usize).saturating_sub(1);
    let end = (end_index as usize + 1).min(data.len());
    if end <= start || end - start < 2 {
        return;
    }

    let gram_bytes = &data[start..end];
    if gram_bytes.len() > MAX_SPARSE_GRAM_SIZE as usize {
        return;
    }

    let hash = ngram_from_bytes(gram_bytes);

    // Compute masks
    let loc_mask = 1u8 << (start % 8);
    let next_mask = if end < data.len() {
        next_bit(data[end])
    } else {
        0
    };

    match result.entry(hash) {
        Entry::Occupied(mut e) => {
            let m = e.get_mut();
            m.loc_mask |= loc_mask;
            m.next_mask |= next_mask;
        }
        Entry::Vacant(e) => {
            e.insert(NGramMasks {
                loc_mask,
                next_mask,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse gram extraction for QUERYING
// ---------------------------------------------------------------------------

/// Internal queue for query-time sparse gram selection.
///
/// Maintains a deque of (index, value) pairs with strictly increasing values.
/// When a new element can't maintain the invariant, it produces an n-gram
/// boundary (preferring longer n-grams over shorter ones).
#[derive(Clone)]
struct QueryQueue {
    inner: VecDeque<PosState>,
}

impl QueryQueue {
    fn new() -> Self {
        Self {
            inner: VecDeque::new(),
        }
    }

    fn front_idx(&self) -> u32 {
        self.inner[0].index
    }

    fn pop_front(&mut self) -> std::ops::Range<u32> {
        let res = self.inner[0].index..self.inner[1].index;
        self.inner.pop_front();
        res
    }

    fn peek_front(&self) -> Option<std::ops::Range<u32>> {
        if self.inner.len() > 1 {
            Some(self.inner[0].index..self.inner[1].index)
        } else {
            None
        }
    }

    /// Push a new element, maintaining the strictly-increasing-value invariant.
    /// Returns a range if an n-gram was produced by popping the back.
    fn push(&mut self, next: PosState) -> Option<std::ops::Range<u32>> {
        while let Some(back) = self.inner.back() {
            if back.value < next.value {
                break;
            }
            if self.inner.len() == 1 {
                let range = self.inner[0].index..next.index;
                self.inner[0] = next;
                return Some(range);
            }
            self.inner.pop_back();
        }
        self.inner.push_back(next);
        None
    }
}

/// Query-time sparse gram extractor.
///
/// Selects the longest possible n-grams that cover the query string,
/// minimizing the number of posting lists to intersect.
pub struct QueryGrams {
    queue: QueryQueue,
    queue2: QueryQueue,
    content: Vec<u8>,
    len: u32,
    buffered: u32,
    produced: u32,
    first_content_idx: u32,
}

impl Default for QueryGrams {
    fn default() -> Self {
        Self {
            produced: 1,
            buffered: 0,
            len: 0,
            content: Vec::new(),
            queue: QueryQueue::new(),
            queue2: QueryQueue::new(),
            first_content_idx: 0,
        }
    }
}

impl QueryGrams {
    /// Append a single byte to the n-gram state.
    /// May produce n-grams reported via the consumer callback.
    pub fn append_byte<F>(&mut self, b: u8, mut consumer: F)
    where
        F: FnMut(&[u8]),
    {
        self.content.push(b);
        self.buffered += 1;
        if self.buffered > 1 {
            self.consume_buffered(&mut consumer);
        }
    }

    /// Consume one buffered byte, potentially producing n-grams.
    fn consume_buffered<F>(&mut self, consumer: &mut F)
    where
        F: FnMut(&[u8]),
    {
        if self.buffered == 0 {
            return;
        }
        let idx = self.len;
        self.len += 1;
        self.buffered -= 1;

        if idx == 0 {
            return;
        }

        let content_offset = idx as usize - self.first_content_idx as usize;
        let (value, value2) = hash_pair(&self.content[content_offset - 1..]);

        // Ensure n-grams don't exceed MAX_SPARSE_GRAM_SIZE
        if idx >= MAX_SPARSE_GRAM_SIZE {
            let q1_over = idx - self.queue.front_idx() + 1 == MAX_SPARSE_GRAM_SIZE;
            let q2_over = idx - self.queue2.front_idx() + 1 == MAX_SPARSE_GRAM_SIZE;
            match (q1_over, q2_over) {
                (true, true) => {
                    let range = self.queue.pop_front();
                    let range2 = self.queue2.pop_front();
                    self.extract_gram(range.start, range.end.max(range2.end), consumer);
                }
                (true, false) => {
                    let range = self.queue.pop_front();
                    if self.produced < self.queue2.front_idx() {
                        self.extract_gram(range.start, range.end, consumer);
                    }
                }
                (false, true) => {
                    let range2 = self.queue2.pop_front();
                    if self.produced < self.queue.front_idx() {
                        self.extract_gram(range2.start, range2.end, consumer);
                    }
                }
                (false, false) => {}
            }
        }

        let range = self.queue.push(PosState { index: idx, value });
        let range2 = self.queue2.push(PosState {
            index: idx,
            value: value2,
        });

        match (range, range2) {
            (Some(range), Some(range2)) => {
                self.extract_gram(range.start.min(range2.start), range.end, consumer);
            }
            (Some(range), None) => {
                if self.produced < self.queue2.front_idx() {
                    self.extract_gram(range.start, range.end, consumer);
                }
            }
            (None, Some(range2)) => {
                if self.produced < self.queue.front_idx() {
                    self.extract_gram(range2.start, range2.end, consumer);
                }
            }
            (None, None) => {}
        }

        self.shrink_content();
    }

    /// Emit one n-gram to the consumer.
    fn extract_gram<F>(&mut self, begin_index: u32, end_index: u32, consumer: &mut F)
    where
        F: FnMut(&[u8]),
    {
        self.produced = end_index;
        let end = (end_index - self.first_content_idx) as usize;
        let begin = (begin_index - self.first_content_idx) as usize;
        if begin >= 1 && end < self.content.len() {
            consumer(&self.content[begin - 1..end + 1]);
        }
    }

    fn shrink_content(&mut self) {
        let new_first = self.queue.front_idx().min(self.queue2.front_idx()) - 1;
        let drain_count = (new_first - self.first_content_idx) as usize;
        if drain_count > 0 {
            self.content.drain(0..drain_count);
            self.first_content_idx = new_first;
        }
    }

    /// Flush remaining buffered n-grams. Must be called after all bytes are appended.
    pub fn flush<F>(mut self, mut consumer: F)
    where
        F: FnMut(&[u8]),
    {
        self.consume_buffered(&mut consumer);
        match self.len {
            3.. => {
                while self.produced < self.len - 1 {
                    self.consume_next(&mut consumer);
                }
            }
            2 => {
                if self.produced == 1 {
                    self.extract_gram(1, 1, &mut consumer);
                }
            }
            _ => {}
        }
    }

    fn consume_next<F>(&mut self, consumer: &mut F)
    where
        F: FnMut(&[u8]),
    {
        let range = self.queue.peek_front();
        let range2 = self.queue2.peek_front();
        match (range, range2) {
            (Some(range), Some(range2)) if range.end < range2.end => {
                self.queue.pop_front();
                if self.produced < self.queue2.front_idx() {
                    self.extract_gram(range.start, range.end, consumer);
                }
            }
            (Some(range), Some(range2)) if range.end > range2.end => {
                self.queue2.pop_front();
                if self.produced < self.queue.front_idx() {
                    self.extract_gram(range2.start, range2.end, consumer);
                }
            }
            (Some(range), Some(range2)) => {
                debug_assert_eq!(range.end, range2.end);
                self.queue.pop_front();
                self.queue2.pop_front();
                self.extract_gram(range.start.min(range2.start), range.end, consumer);
            }
            (Some(range), None) => {
                self.queue.pop_front();
                self.extract_gram(range.start, range.end, consumer);
            }
            (None, Some(range2)) => {
                self.queue2.pop_front();
                self.extract_gram(range2.start, range2.end, consumer);
            }
            (None, None) => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Public extraction API
// ---------------------------------------------------------------------------

/// Extract sparse n-grams from raw file content for indexing.
///
/// Normalizes to indexable bytes, extracts both original and case-normalized
/// grams, and returns deduplicated (hash, masks) pairs.
pub fn extract_for_indexing(data: &[u8]) -> Vec<(NGramHash, NGramMasks)> {
    let normalized = text_to_indexable(data);
    extract_sparse_grams_for_indexing(&normalized)
}

/// Extract sparse n-grams from a literal string for querying.
///
/// Returns the optimal set of n-gram hashes that cover the query string,
/// preferring longer (more selective) n-grams.
pub fn extract_for_querying(literal: &str) -> Vec<NGramHash> {
    let normalized = text_to_indexable(literal.as_bytes());
    if normalized.len() < 2 {
        return Vec::new();
    }

    let mut grams = QueryGrams::default();
    let mut results: Vec<NGramHash> = Vec::new();

    for &b in &normalized {
        grams.append_byte(b, |gram_bytes| {
            results.push(ngram_from_bytes(gram_bytes));
        });
    }
    grams.flush(|gram_bytes| {
        results.push(ngram_from_bytes(gram_bytes));
    });

    results.sort_unstable();
    results.dedup();
    results
}

/// Check if a file is likely binary by scanning the first 8KB for NUL bytes.
pub fn is_binary(data: &[u8]) -> bool {
    let check_len = data.len().min(8192);
    data[..check_len].contains(&0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_from_bytes_bigram() {
        let h = ngram_from_bytes(b"ab");
        // Length should be 2
        assert_eq!(h & 0xFF, 2);
    }

    #[test]
    fn test_ngram_from_bytes_trigram() {
        let h = ngram_from_bytes(b"abc");
        assert_eq!(h & 0xFF, 3);
    }

    #[test]
    fn test_ngram_from_bytes_longer() {
        let h = ngram_from_bytes(b"abcde");
        assert_eq!(h & 0xFF, 5);
        // Different content should produce different hashes
        let h2 = ngram_from_bytes(b"vwxyz");
        assert_ne!(h >> 8, h2 >> 8);
    }

    #[test]
    fn test_different_lengths_never_collide() {
        let h2 = ngram_from_bytes(b"ab");
        let h3 = ngram_from_bytes(b"abc");
        let h4 = ngram_from_bytes(b"abcd");
        assert_ne!(h2, h3);
        assert_ne!(h3, h4);
        assert_ne!(h2, h4);
    }

    #[test]
    fn test_byte_to_indexable() {
        assert_eq!(byte_to_indexable(b'A'), b'a');
        assert_eq!(byte_to_indexable(b'z'), b'z');
        assert_eq!(byte_to_indexable(b'0'), b'0');
        assert_eq!(byte_to_indexable(0xFF), 0x80);
    }

    #[test]
    fn test_bigram_map_loaded() {
        let map = get_bigram_map();
        assert!(!map.is_empty(), "bigram map should have entries");
        // The most frequent bigrams should have high priority values
        for &(priority, _) in map.values().take(5) {
            assert!(priority > 0, "priorities should be positive");
        }
    }

    #[test]
    fn test_extract_for_indexing_basic() {
        let data = text_to_indexable(b"hello world");
        let grams = extract_sparse_grams_for_indexing(&data);
        assert!(!grams.is_empty(), "should extract n-grams");
        // Should have at least the bigrams (10 for 11 chars)
        assert!(
            grams.len() >= 10,
            "should have at least bigrams, got {}",
            grams.len()
        );
    }

    #[test]
    fn test_extract_for_indexing_short() {
        assert!(extract_sparse_grams_for_indexing(b"a").is_empty());
        assert!(!extract_sparse_grams_for_indexing(b"ab").is_empty());
    }

    #[test]
    fn test_extract_for_querying_basic() {
        let grams = extract_for_querying("hello world");
        assert!(!grams.is_empty(), "should extract query n-grams");
        // Query should produce fewer n-grams than indexing (prefers longer ones)
        let index_grams = extract_for_indexing(b"hello world");
        assert!(
            grams.len() <= index_grams.len(),
            "query should produce ≤ index grams: {} vs {}",
            grams.len(),
            index_grams.len()
        );
    }

    #[test]
    fn test_extract_for_querying_short() {
        assert!(extract_for_querying("a").is_empty());
        let grams = extract_for_querying("ab");
        assert_eq!(grams.len(), 1, "bigram query should produce 1 n-gram");
    }

    #[test]
    fn test_query_grams_subset_of_index_grams() {
        // Every n-gram produced by querying must also appear in the index grams
        let text = "self.reset_states(";
        let normalized = text_to_indexable(text.as_bytes());

        let index_grams: HashMap<NGramHash, NGramMasks> =
            extract_sparse_grams_for_indexing(&normalized)
                .into_iter()
                .collect();
        let query_grams = extract_for_querying(text);

        for qg in &query_grams {
            assert!(
                index_grams.contains_key(qg),
                "query gram {:08x} not found in index grams",
                qg
            );
        }
    }

    #[test]
    fn test_is_binary() {
        assert!(!is_binary(b"hello world"));
        assert!(is_binary(b"hello\0world"));
    }

    #[test]
    fn test_masks_next_byte() {
        let data = text_to_indexable(b"abcde");
        let grams = extract_sparse_grams_for_indexing(&data);
        // At least some grams should have non-zero next_mask (since there are follow bytes)
        let has_next = grams.iter().any(|(_, m)| m.next_mask != 0);
        assert!(has_next, "some grams should have next_mask set");
    }
}
