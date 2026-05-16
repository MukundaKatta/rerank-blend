# rerank-blend

[![crates.io](https://img.shields.io/crates/v/rerank-blend.svg)](https://crates.io/crates/rerank-blend)

Blend N RAG reranker score streams with weighted normalization or
Reciprocal Rank Fusion.

```rust
use rerank_blend::{blend_weighted, blend_rrf, RrfOpts};

let dense = vec![("a", 0.91), ("b", 0.88), ("c", 0.55)];
let bm25  = vec![("a", 12.3), ("c", 8.1),  ("b", 3.4)];

let weighted = blend_weighted(&[(&dense[..], 0.7), (&bm25[..], 0.3)]);
let rrf      = blend_rrf(&[&dense[..], &bm25[..]], RrfOpts::default());
```

`blend_weighted` min-max normalizes each stream so scale differences
don't dominate. `blend_rrf` is score-free (Cormack et al. 2009 RRF;
default `k=60`). Zero deps. MIT or Apache-2.0.
