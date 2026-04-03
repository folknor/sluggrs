//! Unit tests for the sluggrs glyph pipeline.
//!
//! Run with: cargo test --test glyph_pipeline_test

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping};
use sluggrs::band::{CurveLocation, build_bands};
use sluggrs::glyph_cache::{GlyphEntry, GlyphKey, GlyphMap, NON_VECTOR_GLYPH};
use sluggrs::outline::extract_outline;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn hash_of(key: &GlyphKey) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Build a GlyphKey with synthetic values (avoids needing cosmic_text layout).
fn make_key(font_id_val: u32, glyph_id: u16, weight: u16, flags_bits: u32) -> GlyphKey {
    // cosmic_text::fontdb::ID is just a newtype wrapping u32
    let font_id = cosmic_text::fontdb::ID::dummy();
    // We can't construct arbitrary IDs, so we'll use dummy + differentiate via other fields.
    // For tests that need distinct font_ids we use the layout-based helper below.
    let _ = font_id_val; // used only in layout-based tests
    GlyphKey {
        font_id,
        glyph_id,
        font_weight: weight,
        cache_key_flags: cosmic_text::CacheKeyFlags::from_bits_truncate(flags_bits),
    }
}

fn make_dummy_entry(band_offset: u32) -> GlyphEntry {
    GlyphEntry::new(
        band_offset,
        3,
        4,
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 0.0, 100.0, 100.0],
        1000.0,
    )
}

// ---------------------------------------------------------------------------
// 1. GlyphKey equality
// ---------------------------------------------------------------------------

#[test]
fn glyph_key_equality_same_values() {
    let a = make_key(0, 42, 400, 0);
    let b = make_key(0, 42, 400, 0);
    assert_eq!(a, b, "Identical GlyphKeys should be equal");
}

#[test]
fn glyph_key_inequality_different_glyph_id() {
    let a = make_key(0, 42, 400, 0);
    let b = make_key(0, 99, 400, 0);
    assert_ne!(
        a, b,
        "GlyphKeys with different glyph_id should not be equal"
    );
}

#[test]
fn glyph_key_inequality_different_weight() {
    let a = make_key(0, 42, 400, 0);
    let b = make_key(0, 42, 700, 0);
    assert_ne!(
        a, b,
        "GlyphKeys with different font_weight should not be equal"
    );
}

#[test]
fn glyph_key_inequality_different_flags() {
    let a = make_key(0, 42, 400, 0);
    let b = make_key(0, 42, 400, 1);
    assert_ne!(
        a, b,
        "GlyphKeys with different cache_key_flags should not be equal"
    );
}

// ---------------------------------------------------------------------------
// 2. GlyphKey hashing — no collisions for a small, distinct set
// ---------------------------------------------------------------------------

#[test]
fn glyph_key_hashing_no_collisions() {
    let keys = [
        make_key(0, 1, 400, 0),
        make_key(0, 2, 400, 0),
        make_key(0, 1, 700, 0),
        make_key(0, 1, 400, 1),
        make_key(0, 100, 300, 0),
    ];

    let mut seen = HashMap::new();
    for (i, key) in keys.iter().enumerate() {
        let h = hash_of(key);
        if let Some(&prev_idx) = seen.get(&h) {
            assert_ne!(
                keys[prev_idx], *key,
                "Hash collision between distinct GlyphKeys at indices {prev_idx} and {i}"
            );
        }
        seen.insert(h, i);
    }
    assert_eq!(
        seen.len(),
        keys.len(),
        "All distinct GlyphKeys should produce distinct hashes"
    );
}

#[test]
fn glyph_key_consistent_hashing() {
    let key = make_key(0, 42, 400, 0);
    let h1 = hash_of(&key);
    let h2 = hash_of(&key);
    assert_eq!(h1, h2, "Same GlyphKey must produce consistent hash");
}

#[test]
fn glyph_key_equal_implies_same_hash() {
    let a = make_key(0, 42, 400, 0);
    let b = make_key(0, 42, 400, 0);
    assert_eq!(a, b);
    assert_eq!(
        hash_of(&a),
        hash_of(&b),
        "Equal GlyphKeys must have equal hashes"
    );
}

// ---------------------------------------------------------------------------
// 3. GlyphEntry non-vector sentinel
// ---------------------------------------------------------------------------

#[test]
fn non_vector_glyph_sentinel_is_non_vector() {
    assert!(
        NON_VECTOR_GLYPH.is_non_vector(),
        "NON_VECTOR_GLYPH sentinel should report is_non_vector() == true"
    );
}

#[test]
fn normal_glyph_entry_is_not_non_vector() {
    let entry = make_dummy_entry(0);
    assert!(
        !entry.is_non_vector(),
        "A normal GlyphEntry (band_offset != u32::MAX) should not be non-vector"
    );
}

#[test]
fn edge_case_max_minus_one_not_sentinel() {
    let entry = make_dummy_entry(u32::MAX - 1);
    assert!(
        !entry.is_non_vector(),
        "band_offset = u32::MAX - 1 should not be treated as non-vector"
    );
}

// ---------------------------------------------------------------------------
// 4. GlyphMap basic operations
// ---------------------------------------------------------------------------

#[test]
fn glyph_map_insert_and_get() {
    let mut map = GlyphMap::new();
    let key = make_key(0, 42, 400, 0);
    let entry = make_dummy_entry(100);

    map.insert_and_mark_used(key, entry);

    assert!(map.contains_key(&key));
    let got = map
        .get_and_mark_used(&key)
        .expect("Should find inserted key");
    assert_eq!(got.band_offset, 100);
}

#[test]
fn glyph_map_len() {
    let mut map = GlyphMap::new();
    assert_eq!(map.len(), 0);

    map.insert_and_mark_used(make_key(0, 1, 400, 0), make_dummy_entry(0));
    assert_eq!(map.len(), 1);

    map.insert_and_mark_used(make_key(0, 2, 400, 0), make_dummy_entry(1));
    assert_eq!(map.len(), 2);

    // Re-insert same key should overwrite, not increase length
    map.insert_and_mark_used(make_key(0, 1, 400, 0), make_dummy_entry(99));
    assert_eq!(map.len(), 2);
}

#[test]
fn glyph_map_clear() {
    let mut map = GlyphMap::new();
    map.insert_and_mark_used(make_key(0, 1, 400, 0), make_dummy_entry(0));
    map.insert_and_mark_used(make_key(0, 2, 400, 0), make_dummy_entry(1));
    assert_eq!(map.len(), 2);

    map.clear();
    assert_eq!(map.len(), 0);
    assert!(!map.contains_key(&make_key(0, 1, 400, 0)));
}

#[test]
fn glyph_map_get_missing_key_returns_none() {
    let mut map = GlyphMap::new();
    assert!(map.get_and_mark_used(&make_key(0, 999, 400, 0)).is_none());
}

// ---------------------------------------------------------------------------
// 4b. GlyphMap usage tracking
// ---------------------------------------------------------------------------

#[test]
fn glyph_map_usage_tracking_basic() {
    let mut map = GlyphMap::new();
    let k1 = make_key(0, 1, 400, 0);
    let k2 = make_key(0, 2, 400, 0);
    map.insert_and_mark_used(k1, make_dummy_entry(0));
    map.insert_and_mark_used(k2, make_dummy_entry(1));

    // Start a new frame so usage counts reset
    map.next_frame();
    assert_eq!(map.in_use_count(), 0);

    map.get_and_mark_used(&k1);
    assert_eq!(map.in_use_count(), 1);

    map.get_and_mark_used(&k2);
    assert_eq!(map.in_use_count(), 2);

    // Marking same key again doesn't increase count
    map.get_and_mark_used(&k1);
    assert_eq!(map.in_use_count(), 2);
}

#[test]
fn glyph_map_next_frame_resets_in_use() {
    let mut map = GlyphMap::new();
    let k1 = make_key(0, 1, 400, 0);
    map.insert_and_mark_used(k1, make_dummy_entry(0));
    assert_eq!(map.in_use_count(), 1);

    map.next_frame();
    assert_eq!(map.in_use_count(), 0);
    // Glyph is still cached
    assert!(map.contains_key(&k1));
}

#[test]
fn glyph_map_clear_resets_both_map_and_usage() {
    let mut map = GlyphMap::new();
    let k1 = make_key(0, 1, 400, 0);
    map.insert_and_mark_used(k1, make_dummy_entry(0));

    map.clear();
    assert_eq!(map.len(), 0);
    assert_eq!(map.in_use_count(), 0);
}

#[test]
fn non_vector_glyph_can_be_marked_in_use() {
    let mut map = GlyphMap::new();
    let k1 = make_key(0, 1, 400, 0);
    map.insert_and_mark_used(k1, NON_VECTOR_GLYPH);

    // Start a new frame so we can test mark-used via get_and_mark_used
    map.next_frame();
    map.get_and_mark_used(&k1);
    assert_eq!(map.in_use_count(), 1);

    // Non-vector and vector glyphs both count toward in_use
    let k2 = make_key(0, 2, 400, 0);
    map.insert_and_mark_used(k2, make_dummy_entry(42));
    assert_eq!(map.in_use_count(), 2);
}

// ---------------------------------------------------------------------------
// 5. Outline extraction round-trip (via cosmic_text)
// ---------------------------------------------------------------------------

#[test]
fn outline_extraction_round_trip() {
    let mut font_system = FontSystem::new();

    let metrics = Metrics::new(24.0, 30.0);
    let mut buffer = Buffer::new(&mut font_system, metrics);
    buffer.set_text(
        &mut font_system,
        "A",
        &Attrs::new(),
        Shaping::Advanced,
        None,
    );
    buffer.shape_until_scroll(&mut font_system, false);

    let run = buffer
        .layout_runs()
        .next()
        .expect("Should have a layout run");
    let glyph = &run.glyphs[0];

    let font = font_system
        .get_font(glyph.font_id, glyph.font_weight)
        .expect("Font should be available");
    let font_data = font.data();

    let outline = extract_outline(font_data, 0, glyph.glyph_id, &[])
        .expect("Glyph 'A' should have an outline");

    assert!(
        !outline.curves.is_empty(),
        "Outline for 'A' should have curves"
    );
    assert!(
        outline.bounds[2] > outline.bounds[0],
        "Bounds width should be positive (max_x > min_x)"
    );
    assert!(
        outline.bounds[3] > outline.bounds[1],
        "Bounds height should be positive (max_y > min_y)"
    );

    // Outline bounds should be valid
    assert!(
        outline.bounds[2] > outline.bounds[0],
        "Outline bounds width should be positive"
    );
    assert!(
        outline.bounds[3] > outline.bounds[1],
        "Outline bounds height should be positive"
    );
}

// ---------------------------------------------------------------------------
// 6. Band data sanity
// ---------------------------------------------------------------------------

#[test]
fn band_data_sanity() {
    let mut font_system = FontSystem::new();

    let metrics = Metrics::new(24.0, 30.0);
    let mut buffer = Buffer::new(&mut font_system, metrics);
    buffer.set_text(
        &mut font_system,
        "B",
        &Attrs::new(),
        Shaping::Advanced,
        None,
    );
    buffer.shape_until_scroll(&mut font_system, false);

    let run = buffer
        .layout_runs()
        .next()
        .expect("Should have a layout run");
    let glyph = &run.glyphs[0];

    let font = font_system
        .get_font(glyph.font_id, glyph.font_weight)
        .expect("Font should be available");
    let font_data = font.data();

    let outline = extract_outline(font_data, 0, glyph.glyph_id, &[])
        .expect("Glyph 'B' should have an outline");
    let gpu_outline = &outline;

    // Create curve locations (one per curve, sequentially laid out)
    let curve_locations: Vec<CurveLocation> = (0..gpu_outline.curves.len())
        .map(|i| CurveLocation {
            offset: (i * 3) as u32,
        })
        .collect();

    let band_count_x = 4u32;
    let band_count_y = 4u32;

    let band_data = build_bands(
        gpu_outline,
        &curve_locations,
        band_count_x,
        band_count_y,
        Vec::new(),
        &mut sluggrs::band::BandScratch::default(),
    );

    // Band counts should match requested
    assert_eq!(band_data.band_count_x, band_count_x);
    assert_eq!(band_data.band_count_y, band_count_y);

    // Entries should be non-empty (at minimum we have headers)
    assert!(
        !band_data.entries.is_empty(),
        "Band entries should not be empty"
    );

    // band_transform should have non-zero scale values
    let [scale_x, scale_y, _offset_x, _offset_y] = band_data.band_transform;
    assert!(
        scale_x.abs() > 1e-10,
        "band_transform scale_x should be non-zero, got {scale_x}"
    );
    assert!(
        scale_y.abs() > 1e-10,
        "band_transform scale_y should be non-zero, got {scale_y}"
    );

    // At least some curves should appear in the band structure.
    // The total entries should be more than just the headers
    // (headers = (band_count_x + band_count_y) * 4 u32s each).
    let header_u32s = (band_count_x + band_count_y) as usize * 4;
    assert!(
        band_data.entries.len() > header_u32s,
        "Band data should contain curve references beyond headers. \
         entries.len()={}, header_u32s={}",
        band_data.entries.len(),
        header_u32s
    );
}

#[test]
fn band_data_single_band() {
    // Edge case: build with 1x1 bands — all curves go into the single band.
    let mut font_system = FontSystem::new();

    let metrics = Metrics::new(24.0, 30.0);
    let mut buffer = Buffer::new(&mut font_system, metrics);
    buffer.set_text(
        &mut font_system,
        "O",
        &Attrs::new(),
        Shaping::Advanced,
        None,
    );
    buffer.shape_until_scroll(&mut font_system, false);

    let run = buffer
        .layout_runs()
        .next()
        .expect("Should have a layout run");
    let glyph = &run.glyphs[0];

    let font = font_system
        .get_font(glyph.font_id, glyph.font_weight)
        .expect("Font should be available");
    let font_data = font.data();

    let outline = extract_outline(font_data, 0, glyph.glyph_id, &[])
        .expect("Glyph 'O' should have an outline");
    let gpu_outline = &outline;

    let curve_locations: Vec<CurveLocation> = (0..gpu_outline.curves.len())
        .map(|i| CurveLocation {
            offset: (i * 3) as u32,
        })
        .collect();

    let band_data = build_bands(gpu_outline, &curve_locations, 1, 1, Vec::new(), &mut sluggrs::band::BandScratch::default());

    assert_eq!(band_data.band_count_x, 1);
    assert_eq!(band_data.band_count_y, 1);

    // With 1 band in each direction, every non-axis-aligned curve appears
    // in both bands, with dual sorted lists (desc + asc) per band.
    // Headers: 2 bands * 4 u32s = 8
    // Curve refs: non_horiz * 2 lists + non_vert * 2 lists, * 4 u32s each
    let non_horiz = gpu_outline.curves.iter().filter(|c| {
        let min_y = c.p1[1].min(c.p2[1]).min(c.p3[1]);
        let max_y = c.p1[1].max(c.p2[1]).max(c.p3[1]);
        min_y != max_y
    }).count();
    let non_vert = gpu_outline.curves.iter().filter(|c| {
        let min_x = c.p1[0].min(c.p2[0]).min(c.p3[0]);
        let max_x = c.p1[0].max(c.p2[0]).max(c.p3[0]);
        min_x != max_x
    }).count();
    let expected_curve_refs = (non_horiz * 2 + non_vert * 2) * 4;
    let expected_total = 8 + expected_curve_refs;
    assert_eq!(
        band_data.entries.len(),
        expected_total,
        "With 1x1 bands all curves should appear in both the h-band and v-band"
    );
}
