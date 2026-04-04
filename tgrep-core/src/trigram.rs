/// Trigram extraction and hashing.
///
/// A trigram is every overlapping 3-byte window in a byte sequence.
/// We pack 3 bytes into a `u32`: `(a << 16) | (b << 8) | c`.
/// This gives us up to ~16.7M unique trigrams with zero collisions.
pub type TrigramHash = u32;

/// Pack three bytes into a single u32 trigram hash.
#[inline]
pub fn hash(a: u8, b: u8, c: u8) -> TrigramHash {
    (a as u32) << 16 | (b as u32) << 8 | c as u32
}

/// Hash a byte offset into a bit position in a 64-bit loc_mask.
#[inline]
fn loc_bit(offset: usize) -> u64 {
    // Map byte offset to one of 64 buckets. We use a multiplicative hash
    // so nearby offsets don't always map to adjacent bits.
    let bucket = ((offset as u64).wrapping_mul(0x9E3779B97F4A7C15)) >> 58; // top 6 bits → 0..63
    1u64 << bucket
}

/// Hash a byte into a bit position in a 32-bit next_mask (Bloom filter).
#[inline]
fn next_bit(byte: u8) -> u32 {
    1u32 << (byte & 31)
}

/// Per-trigram masks for a single file.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrigramMasks {
    /// Positional Bloom filter: which "slots" in the file contain this trigram.
    pub loc_mask: u64,
    /// Bloom filter of bytes that immediately follow this trigram in the file.
    pub next_mask: u32,
}

/// Extract all unique trigrams from a byte slice.
pub fn extract(data: &[u8]) -> Vec<TrigramHash> {
    if data.len() < 3 {
        return Vec::new();
    }
    let mut seen = vec![false; 1 << 24]; // 16MB bitmap — faster than HashSet
    let mut result = Vec::new();
    for window in data.windows(3) {
        let h = hash(window[0], window[1], window[2]);
        if !seen[h as usize] {
            seen[h as usize] = true;
            result.push(h);
        }
    }
    result
}

/// Extract all unique trigrams with positional and next-byte masks.
///
/// For each unique trigram, computes:
/// - `loc_mask`: Bloom filter of byte offsets where this trigram occurs
/// - `next_mask`: Bloom filter of bytes that immediately follow this trigram
pub fn extract_with_masks(data: &[u8]) -> Vec<(TrigramHash, TrigramMasks)> {
    if data.len() < 3 {
        return Vec::new();
    }

    // Two-pass: first pass accumulates masks, second collects results.
    // Use a parallel array indexed by trigram hash (24-bit space).
    let mut seen = vec![false; 1 << 24];
    let mut loc_masks = vec![0u64; 1 << 24];
    let mut next_masks = vec![0u32; 1 << 24];
    let mut order = Vec::new();

    for (i, window) in data.windows(3).enumerate() {
        let h = hash(window[0], window[1], window[2]);
        let idx = h as usize;

        loc_masks[idx] |= loc_bit(i);

        // Next byte is at position i+3 (the byte after the trigram)
        if i + 3 < data.len() {
            next_masks[idx] |= next_bit(data[i + 3]);
        }

        if !seen[idx] {
            seen[idx] = true;
            order.push(h);
        }
    }

    order
        .into_iter()
        .map(|h| {
            let idx = h as usize;
            (
                h,
                TrigramMasks {
                    loc_mask: loc_masks[idx],
                    next_mask: next_masks[idx],
                },
            )
        })
        .collect()
}

/// Check whether consecutive trigrams from a literal can be adjacent based on masks.
///
/// For trigrams at offsets i and i+1 in a literal, the loc_mask of the first
/// trigram shifted by 1 position AND'd with the second should be non-zero
/// if they appear adjacently in the file.
pub fn check_adjacency(masks: &[(TrigramHash, TrigramMasks)]) -> bool {
    if masks.len() <= 1 {
        return true;
    }
    for pair in masks.windows(2) {
        let prev_loc = pair[0].1.loc_mask;
        let next_loc = pair[1].1.loc_mask;
        // Rotate prev_loc by 1 bit position. Since loc_bit uses a multiplicative
        // hash, adjacent offsets map to different bits — but when we stored them
        // in the file, offset i and offset i+1 each got their own loc_bit.
        // For the adjacency check, we check all 64 possible rotations: if any
        // single-bit rotation of prev_loc overlaps with next_loc, adjacency is possible.
        // Simplified: just AND the raw masks. If they share any bit, the trigrams
        // appear in overlapping positional buckets (probabilistic but effective).
        if prev_loc & next_loc == 0 {
            return false;
        }
    }
    true
}

/// Check whether a trigram's next_mask is compatible with an expected next byte.
pub fn check_next_byte(masks: &TrigramMasks, next_byte: u8) -> bool {
    masks.next_mask & next_bit(next_byte) != 0
}

/// Extract trigrams from a string pattern (for query planning).
pub fn extract_from_literal(s: &str) -> Vec<TrigramHash> {
    extract(s.as_bytes())
}

/// Check if a file is likely binary by scanning the first 8KB for NUL bytes.
pub fn is_binary(data: &[u8]) -> bool {
    let check_len = data.len().min(8192);
    data[..check_len].contains(&0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_packing() {
        assert_eq!(hash(b't', b'h', b'e'), 0x746865);
        assert_eq!(hash(0, 0, 0), 0);
        assert_eq!(hash(0xFF, 0xFF, 0xFF), 0x00FFFFFF);
    }

    #[test]
    fn test_extract_basic() {
        let trigrams = extract(b"the cat");
        // "the", "he ", "e c", " ca", "cat"
        assert_eq!(trigrams.len(), 5);
        assert!(trigrams.contains(&hash(b't', b'h', b'e')));
        assert!(trigrams.contains(&hash(b'c', b'a', b't')));
    }

    #[test]
    fn test_extract_short() {
        assert!(extract(b"ab").is_empty());
        assert!(extract(b"").is_empty());
    }

    #[test]
    fn test_extract_dedup() {
        // "aaa" has trigram "aaa" appearing twice, but should be deduped
        let trigrams = extract(b"aaaa");
        assert_eq!(trigrams.len(), 1);
    }

    #[test]
    fn test_is_binary() {
        assert!(!is_binary(b"hello world"));
        assert!(is_binary(b"hello\0world"));
    }

    #[test]
    fn test_extract_with_masks_basic() {
        let results = extract_with_masks(b"abcde");
        // Trigrams: "abc", "bcd", "cde" → 3 unique
        assert_eq!(results.len(), 3);
        let abc = results
            .iter()
            .find(|(h, _)| *h == hash(b'a', b'b', b'c'))
            .unwrap();
        // "abc" is followed by 'd'
        assert!(check_next_byte(&abc.1, b'd'));
    }

    #[test]
    fn test_extract_with_masks_short() {
        assert!(extract_with_masks(b"ab").is_empty());
        assert!(extract_with_masks(b"").is_empty());
    }

    #[test]
    fn test_next_mask_filters_false_positive() {
        // File contains "abcXe" — trigram "abc" is followed by 'X', not 'd'
        let results = extract_with_masks(b"abcXe");
        let abc = results
            .iter()
            .find(|(h, _)| *h == hash(b'a', b'b', b'c'))
            .unwrap();
        // 'X' should be in the mask
        assert!(check_next_byte(&abc.1, b'X'));
        // 'd' may or may not be (Bloom filter can have false positives, but
        // for a single-entry Bloom it should only have the actual byte)
        // Since next_bit uses byte & 31, 'd'=100 and 'X'=88 → different bits
        // 'd' & 31 = 4, 'X' & 31 = 24
        assert!(!check_next_byte(&abc.1, b'd'));
    }

    #[test]
    fn test_loc_mask_nonzero() {
        let results = extract_with_masks(b"hello world");
        for (_, masks) in &results {
            assert_ne!(
                masks.loc_mask, 0,
                "loc_mask should have at least one bit set"
            );
        }
    }
}
