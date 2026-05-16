//! # rerank-blend
//!
//! Blend N reranker score streams into one final ranking.
//!
//! Two strategies:
//!
//! - [`blend_weighted`] — min-max-normalize each stream, multiply by
//!   its weight, sum, return sorted descending.
//! - [`blend_rrf`] — Reciprocal Rank Fusion (Cormack et al. 2009):
//!   each stream contributes `1 / (k + rank)`, default `k = 60`.
//!   Score-free; works across heterogeneous score scales.
//!
//! ## Example
//!
//! ```
//! use rerank_blend::{blend_weighted, blend_rrf, RrfOpts};
//!
//! let dense = vec![("a", 0.91), ("b", 0.88), ("c", 0.55)];
//! let bm25  = vec![("a", 12.3), ("c", 8.1), ("b", 3.4)];
//!
//! let blended = blend_weighted(&[(&dense, 0.7), (&bm25, 0.3)]);
//! assert_eq!(blended[0].0, "a");
//!
//! let rrf = blend_rrf(&[&dense, &bm25], RrfOpts::default());
//! assert_eq!(rrf[0].0, "a");
//! ```

#![deny(missing_docs)]

use std::collections::HashMap;
use std::hash::Hash;

/// Options for Reciprocal Rank Fusion.
#[derive(Debug, Clone, Copy)]
pub struct RrfOpts {
    /// The `k` smoothing constant. Cormack's paper recommends 60.
    pub k: f32,
}

impl Default for RrfOpts {
    fn default() -> Self {
        Self { k: 60.0 }
    }
}

/// Weighted blend with per-stream min-max normalization. Streams whose
/// max == min contribute zero to ensure the normalization is well-defined.
///
/// Input: a slice of `(stream, weight)` pairs. Each stream is a slice
/// of `(id, score)` pre-sorted in any order.
///
/// Output: sorted descending by the blended score.
pub fn blend_weighted<'a, K: Eq + Hash + Clone + 'a>(
    streams: &[(&'a [(K, f32)], f32)],
) -> Vec<(K, f32)> {
    let mut totals: HashMap<K, f32> = HashMap::new();
    for (stream, weight) in streams {
        if stream.is_empty() {
            continue;
        }
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        for (_, s) in stream.iter() {
            if *s < min {
                min = *s;
            }
            if *s > max {
                max = *s;
            }
        }
        let range = max - min;
        if range == 0.0 {
            continue;
        }
        for (k, s) in stream.iter() {
            let norm = (s - min) / range;
            *totals.entry(k.clone()).or_insert(0.0) += norm * weight;
        }
    }
    let mut out: Vec<(K, f32)> = totals.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Reciprocal Rank Fusion. Streams must be pre-sorted by rank (best first).
pub fn blend_rrf<'a, K: Eq + Hash + Clone + 'a>(
    streams: &[&'a [(K, f32)]],
    opts: RrfOpts,
) -> Vec<(K, f32)> {
    let mut totals: HashMap<K, f32> = HashMap::new();
    for stream in streams {
        for (rank, (k, _)) in stream.iter().enumerate() {
            let contribution = 1.0 / (opts.k + rank as f32 + 1.0);
            *totals.entry(k.clone()).or_insert(0.0) += contribution;
        }
    }
    let mut out: Vec<(K, f32)> = totals.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}
