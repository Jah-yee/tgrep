/// Index builder: walks a repo, extracts sparse n-grams, writes the on-disk index.
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::Result;
use crate::meta::{self, IndexMeta};
use crate::ngram::{self, NGramHash, NGramMasks};
use crate::ondisk::{self, LookupEntry, PostingEntry};
use crate::walker;

const INDEX_DIR_NAME: &str = ".tgrep";

/// Build a trigram index for all text files under `root`.
pub fn build_index(
    root: &Path,
    index_dir: Option<&Path>,
    include_hidden: bool,
    exclude_dirs: &[String],
) -> Result<()> {
    let root = std::fs::canonicalize(root)?;
    let index_dir = match index_dir {
        Some(d) => d.to_path_buf(),
        None => root.join(INDEX_DIR_NAME),
    };
    std::fs::create_dir_all(&index_dir)?;

    eprintln!("Walking {}...", root.display());
    let walk = walker::walk_dir(
        &root,
        &walker::WalkOptions {
            include_hidden,
            exclude_dirs: exclude_dirs.to_vec(),
            ..Default::default()
        },
    );
    eprintln!(
        "Found {} text files ({} binary skipped, {} errors)",
        walk.files.len(),
        walk.skipped_binary,
        walk.skipped_error
    );

    // Read all files and extract sparse n-grams with masks in parallel.
    // Binary content check is done here (not in walker) to avoid an extra
    // 8KB read per file — we're already reading the full file anyway.
    eprintln!("Extracting n-grams...");
    let binary_skipped = std::sync::atomic::AtomicUsize::new(0);
    let file_data: Vec<(String, Vec<(NGramHash, NGramMasks)>)> = walk
        .files
        .par_iter()
        .filter_map(|path| {
            let data = std::fs::read(path).ok()?;
            if ngram::is_binary(&data) {
                binary_skipped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return None;
            }
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            // Extract sparse n-grams from normalized content
            let ngram_masks = ngram::extract_for_indexing(&data);
            Some((rel, ngram_masks))
        })
        .collect();
    let extra_binary = binary_skipped.into_inner();
    if extra_binary > 0 {
        eprintln!(
            "Skipped {} additional binary files (detected by content)",
            extra_binary
        );
    }

    // Assign file IDs and build inverted index with masks
    let mut file_id_map: Vec<(u32, String)> = Vec::with_capacity(file_data.len());
    // ngram_hash → Vec<PostingEntry>
    let mut inverted: HashMap<NGramHash, Vec<PostingEntry>> = HashMap::new();

    for (id, (path, ngram_masks)) in file_data.iter().enumerate() {
        let file_id = id as u32;
        file_id_map.push((file_id, path.clone()));

        for &(ngram_hash, masks) in ngram_masks {
            inverted.entry(ngram_hash).or_default().push(PostingEntry {
                file_id,
                loc_mask: masks.loc_mask,
                next_mask: masks.next_mask,
            });
        }
    }

    write_index_v2(&index_dir, &root, &file_id_map, &inverted)?;

    // Write per-file stamps for ALL walked files (including those later
    // rejected as binary-by-content) so the stale check on next startup
    // won't re-process unchanged files that aren't in the index.
    let all_walked: Vec<String> = walk
        .files
        .iter()
        .filter_map(|p| p.strip_prefix(&root).ok())
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .collect();
    let stamps = meta::collect_filestamps(&root, &all_walked);
    meta::write_filestamps(&stamps, &index_dir)?;

    eprintln!("Index built successfully at {}", index_dir.display());
    Ok(())
}

/// Return the default index directory for a given repo root.
pub fn default_index_dir(root: &Path) -> std::path::PathBuf {
    root.join(INDEX_DIR_NAME)
}

/// Write the on-disk index from a pre-computed snapshot (paths + inverted index).
/// This allows the caller to snapshot under a brief lock, then write without holding it.
///
/// Note: snapshots from the server still use the legacy format (no masks).
/// This writes v1-compatible posting entries (file_id only) unless masks are available.
pub fn write_index_from_snapshot(
    root: &Path,
    index_dir: &Path,
    paths: &[String],
    inverted: &HashMap<u32, Vec<u32>>,
    complete: bool,
) -> Result<()> {
    std::fs::create_dir_all(index_dir)?;

    let mut sorted_ngrams: Vec<u32> = inverted.keys().copied().collect();
    sorted_ngrams.sort_unstable();

    // Write index.bin — posting entries with all-ones masks (no filtering).
    let mut postings_file =
        std::io::BufWriter::new(std::fs::File::create(index_dir.join("index.bin"))?);
    let mut lookup_entries: Vec<LookupEntry> = Vec::with_capacity(sorted_ngrams.len());
    let mut offset: u64 = 0;

    for &ngram in &sorted_ngrams {
        let posting_list = inverted.get(&ngram).unwrap();
        let length = posting_list.len() as u32;

        lookup_entries.push(LookupEntry {
            trigram: ngram,
            offset,
            length,
        });

        for &fid in posting_list {
            let entry = PostingEntry {
                file_id: fid,
                loc_mask: u8::MAX, // all bits set = no filtering
                next_mask: u8::MAX,
            };
            postings_file.write_all(&entry.encode())?;
        }
        offset += length as u64 * ondisk::POSTING_ENTRY_SIZE as u64;
    }
    postings_file.flush()?;

    // Write lookup.bin
    let mut lookup_file =
        std::io::BufWriter::new(std::fs::File::create(index_dir.join("lookup.bin"))?);
    for entry in &lookup_entries {
        lookup_file.write_all(&entry.encode())?;
    }
    lookup_file.flush()?;

    // Write files.bin
    let mut files_file =
        std::io::BufWriter::new(std::fs::File::create(index_dir.join("files.bin"))?);
    for (id, path) in paths.iter().enumerate() {
        files_file.write_all(&ondisk::encode_file_entry(id as u32, path))?;
    }
    files_file.flush()?;

    // Write meta.json
    let root = std::fs::canonicalize(root)?;
    let mut meta_obj = IndexMeta::new(
        &root.to_string_lossy(),
        paths.len() as u64,
        sorted_ngrams.len() as u64,
    );
    meta_obj.complete = complete;
    meta_obj.save(index_dir)?;

    Ok(())
}

/// Internal: write v2 index with masks.
fn write_index_v2(
    index_dir: &Path,
    root: &Path,
    file_id_map: &[(u32, String)],
    inverted: &HashMap<NGramHash, Vec<PostingEntry>>,
) -> Result<()> {
    let mut sorted_ngrams: Vec<NGramHash> = inverted.keys().copied().collect();
    sorted_ngrams.sort_unstable();

    eprintln!(
        "Writing index ({} n-grams, {} files)...",
        sorted_ngrams.len(),
        file_id_map.len()
    );

    let mut postings_file =
        std::io::BufWriter::new(std::fs::File::create(index_dir.join("index.bin"))?);
    let mut lookup_entries: Vec<LookupEntry> = Vec::with_capacity(sorted_ngrams.len());
    let mut offset: u64 = 0;

    for &ngram in &sorted_ngrams {
        let posting_list = inverted.get(&ngram).unwrap();
        let length = posting_list.len() as u32;

        lookup_entries.push(LookupEntry {
            trigram: ngram,
            offset,
            length,
        });

        for entry in posting_list {
            postings_file.write_all(&entry.encode())?;
        }
        offset += length as u64 * ondisk::POSTING_ENTRY_SIZE as u64;
    }
    postings_file.flush()?;

    let mut lookup_file =
        std::io::BufWriter::new(std::fs::File::create(index_dir.join("lookup.bin"))?);
    for entry in &lookup_entries {
        lookup_file.write_all(&entry.encode())?;
    }
    lookup_file.flush()?;

    let mut files_file =
        std::io::BufWriter::new(std::fs::File::create(index_dir.join("files.bin"))?);
    for (id, path) in file_id_map {
        files_file.write_all(&ondisk::encode_file_entry(*id, path))?;
    }
    files_file.flush()?;

    let meta = IndexMeta::new(
        &root.to_string_lossy(),
        file_id_map.len() as u64,
        sorted_ngrams.len() as u64,
    );
    meta.save(index_dir)?;

    Ok(())
}
