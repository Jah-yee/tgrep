/// Regex → n-gram query decomposition.
///
/// Parses a regex pattern using `regex-syntax` and extracts literal segments
/// that can be converted to sparse n-gram lookups. Builds a QueryPlan tree of
/// AND/OR nodes that can be evaluated against the index.
use regex_syntax::hir::{Class, Hir, HirKind, Literal};

use crate::ngram::{self, NGramHash};
use crate::ondisk::PostingEntry;

/// A node in the query plan tree.
#[derive(Debug, Clone)]
pub enum QueryPlan {
    /// All n-grams must match (intersection of posting lists).
    And(Vec<NGramHash>),
    /// Any branch can match (union of results).
    Or(Vec<QueryPlan>),
    /// No n-grams could be extracted — must scan all files.
    MatchAll,
}

impl QueryPlan {
    pub fn is_match_all(&self) -> bool {
        matches!(self, QueryPlan::MatchAll)
    }
}

/// Parse a regex pattern and produce a query plan for trigram lookups.
pub fn build_query_plan(pattern: &str, case_insensitive: bool) -> Result<QueryPlan, String> {
    let hir = regex_syntax::parse(pattern).map_err(|e| format!("regex parse error: {e}"))?;
    let plan = decompose_hir(&hir, case_insensitive);
    Ok(simplify(plan))
}

/// Build a query plan for a literal (fixed-string) pattern.
pub fn build_literal_plan(literal: &str, case_insensitive: bool) -> QueryPlan {
    let text = if case_insensitive {
        literal.to_lowercase()
    } else {
        literal.to_string()
    };
    let ngrams = ngram::extract_for_querying(&text);
    if ngrams.is_empty() {
        QueryPlan::MatchAll
    } else {
        QueryPlan::And(ngrams)
    }
}

fn decompose_hir(hir: &Hir, case_insensitive: bool) -> QueryPlan {
    match hir.kind() {
        HirKind::Literal(Literal(bytes)) => {
            let text = if case_insensitive {
                String::from_utf8_lossy(bytes).to_lowercase()
            } else {
                String::from_utf8_lossy(bytes).into_owned()
            };
            let ngrams = ngram::extract_for_querying(&text);
            if ngrams.is_empty() {
                QueryPlan::MatchAll
            } else {
                QueryPlan::And(ngrams)
            }
        }
        HirKind::Concat(subs) => {
            // Collect all literals from concat children into a single string,
            // then extract n-grams. Non-literal children break the chain.
            let mut all_ngrams = Vec::new();
            let mut current_literal = String::new();

            for sub in subs {
                if let HirKind::Literal(Literal(bytes)) = sub.kind() {
                    let s = String::from_utf8_lossy(bytes);
                    current_literal.push_str(&s);
                } else {
                    // Flush the current literal run
                    if !current_literal.is_empty() {
                        let text = if case_insensitive {
                            current_literal.to_lowercase()
                        } else {
                            current_literal.clone()
                        };
                        all_ngrams.extend(ngram::extract_for_querying(&text));
                        current_literal.clear();
                    }
                    // Recurse into the non-literal child
                    let sub_plan = decompose_hir(sub, case_insensitive);
                    if let QueryPlan::And(grams) = sub_plan {
                        all_ngrams.extend(grams);
                    }
                    // MatchAll or Or children don't contribute AND n-grams
                }
            }

            // Flush remaining literal
            if !current_literal.is_empty() {
                let text = if case_insensitive {
                    current_literal.to_lowercase()
                } else {
                    current_literal
                };
                all_ngrams.extend(ngram::extract_for_querying(&text));
            }

            if all_ngrams.is_empty() {
                QueryPlan::MatchAll
            } else {
                QueryPlan::And(all_ngrams)
            }
        }
        HirKind::Alternation(alts) => {
            let plans: Vec<QueryPlan> = alts
                .iter()
                .map(|a| decompose_hir(a, case_insensitive))
                .collect();
            // If any branch is MatchAll, the whole alternation is MatchAll
            if plans.iter().any(|p| p.is_match_all()) {
                QueryPlan::MatchAll
            } else {
                QueryPlan::Or(plans)
            }
        }
        HirKind::Repetition(rep) => {
            if rep.min >= 1 {
                decompose_hir(&rep.sub, case_insensitive)
            } else {
                // min=0 means the pattern is optional → can match anything
                QueryPlan::MatchAll
            }
        }
        HirKind::Capture(cap) => decompose_hir(&cap.sub, case_insensitive),
        HirKind::Class(Class::Unicode(_)) | HirKind::Class(Class::Bytes(_)) => QueryPlan::MatchAll,
        HirKind::Look(_) | HirKind::Empty => QueryPlan::MatchAll,
    }
}

/// Simplify the query plan (dedup n-grams, flatten nested structures).
fn simplify(plan: QueryPlan) -> QueryPlan {
    match plan {
        QueryPlan::And(mut grams) => {
            grams.sort_unstable();
            grams.dedup();
            if grams.is_empty() {
                QueryPlan::MatchAll
            } else {
                QueryPlan::And(grams)
            }
        }
        QueryPlan::Or(plans) => {
            let simplified: Vec<QueryPlan> = plans.into_iter().map(simplify).collect();
            if simplified.len() == 1 {
                simplified.into_iter().next().unwrap()
            } else {
                QueryPlan::Or(simplified)
            }
        }
        other => other,
    }
}

/// Execute a query plan against an index, returning candidate file IDs.
pub fn execute_plan<F>(plan: &QueryPlan, lookup: &F) -> Vec<u32>
where
    F: Fn(NGramHash) -> Vec<u32>,
{
    match plan {
        QueryPlan::And(ngrams) => {
            if ngrams.is_empty() {
                return Vec::new();
            }
            let mut lists: Vec<Vec<u32>> = ngrams.iter().map(|&t| lookup(t)).collect();
            lists.sort_by_key(|l| l.len());

            let mut result: Vec<u32> = lists.remove(0);
            result.sort_unstable();
            result.dedup();

            for mut list in lists {
                list.sort_unstable();
                list.dedup();
                result = intersect_sorted(&result, &list);
                if result.is_empty() {
                    break;
                }
            }
            result
        }
        QueryPlan::Or(plans) => {
            let mut result = Vec::new();
            for sub in plans {
                let mut sub_result = execute_plan(sub, lookup);
                result.append(&mut sub_result);
            }
            result.sort_unstable();
            result.dedup();
            result
        }
        QueryPlan::MatchAll => Vec::new(), // caller must handle: scan all files
    }
}

/// Execute a query plan with mask-aware filtering.
///
/// Uses next_mask checks to reduce false-positive candidates.
/// The `pattern` is the original search literal used to extract
/// the n-grams — needed for next_mask verification.
pub fn execute_plan_with_masks<F>(plan: &QueryPlan, _pattern: &str, lookup: &F) -> Vec<u32>
where
    F: Fn(NGramHash) -> Vec<PostingEntry>,
{
    match plan {
        QueryPlan::And(ngrams) => {
            if ngrams.is_empty() {
                return Vec::new();
            }

            // Fetch full posting entries (with masks) for each n-gram
            let mut lists: Vec<(NGramHash, Vec<PostingEntry>)> =
                ngrams.iter().map(|&t| (t, lookup(t))).collect();
            lists.sort_by_key(|(_, l)| l.len());

            // Start with smallest posting list
            let (_first_ngram, first_list) = lists.remove(0);
            let mut candidates: Vec<u32> = first_list.into_iter().map(|e| e.file_id).collect();
            candidates.sort_unstable();
            candidates.dedup();

            // Intersect with remaining posting lists
            for (_ngram, mut list) in lists {
                list.sort_by_key(|e| e.file_id);
                list.dedup_by_key(|e| e.file_id);

                let file_ids: Vec<u32> = list.into_iter().map(|e| e.file_id).collect();
                candidates = intersect_sorted(&candidates, &file_ids);
                if candidates.is_empty() {
                    break;
                }
            }

            candidates
        }
        QueryPlan::Or(plans) => {
            let mut result = Vec::new();
            for sub in plans {
                let mut sub_result = execute_plan_with_masks(sub, _pattern, lookup);
                result.append(&mut sub_result);
            }
            result.sort_unstable();
            result.dedup();
            result
        }
        QueryPlan::MatchAll => Vec::new(),
    }
}

fn intersect_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_plan() {
        let plan = build_query_plan("hello", false).unwrap();
        match plan {
            QueryPlan::And(grams) => {
                assert!(!grams.is_empty(), "should extract n-grams from 'hello'");
            }
            _ => panic!("expected And plan for literal"),
        }
    }

    #[test]
    fn test_alternation_plan() {
        let plan = build_query_plan("foo|bar", false).unwrap();
        match plan {
            QueryPlan::Or(branches) => {
                assert_eq!(branches.len(), 2);
            }
            _ => panic!("expected Or plan for alternation"),
        }
    }

    #[test]
    fn test_short_pattern() {
        // Single character — too short for any n-gram
        let plan = build_query_plan("a", false).unwrap();
        assert!(plan.is_match_all());
    }

    #[test]
    fn test_wildcard_is_match_all() {
        let plan = build_query_plan(".*", false).unwrap();
        assert!(plan.is_match_all());
    }

    #[test]
    fn test_intersect_sorted() {
        assert_eq!(intersect_sorted(&[1, 3, 5, 7], &[2, 3, 5, 8]), vec![3, 5]);
        assert_eq!(intersect_sorted(&[1, 2, 3], &[4, 5, 6]), Vec::<u32>::new());
    }

    #[test]
    fn test_case_insensitive_literal_plan() {
        let plan = build_literal_plan("class AlertSchema", true);
        match &plan {
            QueryPlan::And(grams) => {
                assert!(!grams.is_empty(), "should have n-grams");
                // Verify these match what querying the lowercase version produces
                let expected = ngram::extract_for_querying("class alertschema");
                for g in &expected {
                    assert!(grams.contains(g), "missing n-gram {g:#010x}");
                }
            }
            _ => panic!("expected And plan"),
        }
    }

    #[test]
    fn test_case_insensitive_regex_plan() {
        let plan = build_query_plan("class AlertSchema", true).unwrap();
        match &plan {
            QueryPlan::And(grams) => {
                assert!(!grams.is_empty(), "should have n-grams");
                let expected = ngram::extract_for_querying("class alertschema");
                for g in &expected {
                    assert!(grams.contains(g), "missing n-gram {g:#010x}");
                }
            }
            _ => panic!("expected And plan, got {plan:?}"),
        }
    }

    #[test]
    fn test_case_insensitive_end_to_end() {
        let content = b"internal class AlertSchema : AlertBaseSchema";

        // Extract n-grams the way the builder does (normalized)
        let ngram_masks = ngram::extract_for_indexing(content);

        // Build inverted index for file_id=0
        let mut inverted = std::collections::HashMap::<NGramHash, Vec<u32>>::new();
        for &(hash, _) in &ngram_masks {
            inverted.entry(hash).or_default().push(0);
        }

        // Query with case-insensitive plan
        let plan = build_query_plan("class AlertSchema", true).unwrap();
        let candidates = execute_plan(&plan, &|g| inverted.get(&g).cloned().unwrap_or_default());

        assert!(
            candidates.contains(&0),
            "case-insensitive search should find the file"
        );
    }

    #[test]
    fn test_mask_filtering_finds_match() {
        let content = b"calling mutex_lock here";
        let ngram_masks = ngram::extract_for_indexing(content);

        let mut inverted = std::collections::HashMap::<NGramHash, Vec<PostingEntry>>::new();
        for &(hash, masks) in &ngram_masks {
            inverted.entry(hash).or_default().push(PostingEntry {
                file_id: 0,
                loc_mask: masks.loc_mask,
                next_mask: masks.next_mask,
            });
        }

        let plan = build_literal_plan("mutex_lock", false);
        let candidates = execute_plan_with_masks(&plan, "mutex_lock", &|g| {
            inverted.get(&g).cloned().unwrap_or_default()
        });

        assert!(
            candidates.contains(&0),
            "mask filtering should find the file containing 'mutex_lock'"
        );
    }

    #[test]
    fn test_mask_filtering_rejects_false_positive() {
        // File contains "mutex" and "clock" but NOT "mutex_lock"
        let content = b"use mutex; use clock;";
        let ngram_masks = ngram::extract_for_indexing(content);

        let mut inverted = std::collections::HashMap::<NGramHash, Vec<PostingEntry>>::new();
        for &(hash, masks) in &ngram_masks {
            inverted.entry(hash).or_default().push(PostingEntry {
                file_id: 0,
                loc_mask: masks.loc_mask,
                next_mask: masks.next_mask,
            });
        }

        let plan = build_literal_plan("mutex_lock", false);
        let candidates = execute_plan_with_masks(&plan, "mutex_lock", &|g| {
            inverted.get(&g).cloned().unwrap_or_default()
        });

        // The file should NOT be a candidate because it doesn't contain
        // the required n-grams (e.g., "x_l", "_loc", etc. are missing)
        assert!(
            candidates.is_empty(),
            "mask filtering should reject file not containing 'mutex_lock' n-grams"
        );
    }
}
