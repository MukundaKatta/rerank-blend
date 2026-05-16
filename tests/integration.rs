use rerank_blend::{blend_rrf, blend_weighted, RrfOpts};

#[test]
fn weighted_blends_consistent_winner() {
    let dense = vec![("a", 0.91), ("b", 0.88), ("c", 0.55)];
    let bm25  = vec![("a", 12.3), ("c", 8.1),  ("b", 3.4)];
    let out = blend_weighted(&[(&dense[..], 0.7), (&bm25[..], 0.3)]);
    assert_eq!(out[0].0, "a");
}

#[test]
fn weighted_normalization_avoids_scale_bias() {
    // dense scores are 0..1; bm25 scores are 0..100. Without
    // normalization bm25 would dominate.
    let dense = vec![("a", 0.95), ("b", 0.50)];
    let bm25 = vec![("b", 95.0), ("a", 5.0)];
    // Equal weights -> a and b should tie on normalized blend.
    let out = blend_weighted(&[(&dense[..], 0.5), (&bm25[..], 0.5)]);
    let a = out.iter().find(|(k, _)| *k == "a").unwrap().1;
    let b = out.iter().find(|(k, _)| *k == "b").unwrap().1;
    assert!((a - b).abs() < 1e-5, "a={a} b={b}");
}

#[test]
fn weighted_handles_partial_overlap() {
    let s1 = vec![("a", 1.0), ("b", 0.5)];
    let s2 = vec![("b", 1.0), ("c", 0.5)];
    let out = blend_weighted(&[(&s1[..], 0.5), (&s2[..], 0.5)]);
    let keys: Vec<&str> = out.iter().map(|(k, _)| *k).collect();
    assert!(keys.contains(&"a") && keys.contains(&"b") && keys.contains(&"c"));
}

#[test]
fn rrf_default_picks_consistent_top() {
    let s1 = vec![("a", 0.0), ("b", 0.0), ("c", 0.0)];
    let s2 = vec![("a", 0.0), ("c", 0.0), ("b", 0.0)];
    let out = blend_rrf(&[&s1[..], &s2[..]], RrfOpts::default());
    assert_eq!(out[0].0, "a"); // top of both streams
}

#[test]
fn rrf_score_decreases_with_rank() {
    let s = vec![("a", 0.0), ("b", 0.0), ("c", 0.0)];
    let out = blend_rrf(&[&s[..]], RrfOpts::default());
    assert!(out[0].1 > out[1].1);
    assert!(out[1].1 > out[2].1);
}

#[test]
fn empty_streams_handled() {
    let s: Vec<(&str, f32)> = vec![];
    let out = blend_weighted::<&str>(&[(&s[..], 1.0)]);
    assert!(out.is_empty());
}
