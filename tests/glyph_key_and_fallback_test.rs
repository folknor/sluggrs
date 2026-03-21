//! Tests for GlyphKey font-ID discrimination and non-vector glyph fallback.
//!
//! Run with: cargo test --test glyph_key_and_fallback_test -- --nocapture

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};

use cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping};
use sluggrs::glyph_cache::{GlyphEntry, GlyphKey, GlyphMap, NON_VECTOR_GLYPH};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn hash_of(key: &GlyphKey) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Shape a single character with the given font family and return the first LayoutGlyph's GlyphKey.
fn shape_and_key(font_system: &mut FontSystem, ch: char, family: Family<'_>) -> GlyphKey {
    let metrics = Metrics::new(24.0, 30.0);
    let mut buffer = Buffer::new(font_system, metrics);
    let text = String::from(ch);
    buffer.set_text(
        font_system,
        &text,
        &Attrs::new().family(family),
        Shaping::Advanced,
        None,
    );
    buffer.shape_until_scroll(font_system, false);

    let run = buffer
        .layout_runs()
        .next()
        .expect("Should produce at least one layout run");
    let glyph = &run.glyphs[0];
    GlyphKey::from_layout_glyph(glyph)
}

// ---------------------------------------------------------------------------
// Test 1: GlyphKey distinguishes actual font IDs from layout
// ---------------------------------------------------------------------------

#[test]
fn glyph_key_distinguishes_font_ids_from_layout() {
    let mut font_system = FontSystem::new();

    // Shape the same character using two different generic font families.
    // SansSerif and Monospace should resolve to different system fonts,
    // producing different font_id values in the resulting LayoutGlyphs.
    let key_sans = shape_and_key(&mut font_system, 'A', Family::SansSerif);
    let key_mono = shape_and_key(&mut font_system, 'A', Family::Monospace);

    // The two keys should differ because they come from different fonts.
    assert_ne!(
        key_sans, key_mono,
        "GlyphKeys from SansSerif vs Monospace should differ. \
         SansSerif font_id={:?}, Monospace font_id={:?}",
        key_sans.font_id, key_mono.font_id
    );

    // Verify they hash differently (important for HashMap/HashSet usage).
    let mut set = HashSet::new();
    set.insert(hash_of(&key_sans));
    set.insert(hash_of(&key_mono));
    assert_eq!(
        set.len(),
        2,
        "GlyphKeys from different fonts should produce different hashes"
    );
}

#[test]
fn glyph_key_same_font_produces_equal_keys() {
    let mut font_system = FontSystem::new();

    // Shape the same character with the same family twice — keys should be identical.
    let key_a = shape_and_key(&mut font_system, 'A', Family::SansSerif);
    let key_b = shape_and_key(&mut font_system, 'A', Family::SansSerif);

    assert_eq!(
        key_a, key_b,
        "Shaping the same character with the same font family should produce equal GlyphKeys"
    );
    assert_eq!(
        hash_of(&key_a),
        hash_of(&key_b),
        "Equal GlyphKeys must produce equal hashes"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Non-vector glyph fallback behavior
// ---------------------------------------------------------------------------

#[test]
fn non_vector_glyph_tracked_in_glyph_map() {
    let mut map = GlyphMap::new();

    // Create a synthetic key representing a glyph that has no vector outline
    // (e.g., an emoji or bitmap-only glyph).
    let key = GlyphKey {
        font_id: cosmic_text::fontdb::ID::dummy(),
        glyph_id: 9999,
        font_weight: 400,
        cache_key_flags: cosmic_text::CacheKeyFlags::empty(),
    };

    // Mark it as non-vector (this is the pattern used in the rendering pipeline
    // when outline extraction fails or returns no curves).
    map.insert(key, NON_VECTOR_GLYPH);

    // Verify the atlas records it and reports it as non-vector.
    let entry = map
        .get(&key)
        .expect("GlyphMap should contain the non-vector entry after insert");
    assert!(
        entry.is_non_vector(),
        "Entry inserted with NON_VECTOR_GLYPH sentinel should report is_non_vector() == true"
    );

    // Verify contains_key returns true so we skip re-extraction on subsequent frames.
    assert!(
        map.contains_key(&key),
        "contains_key should return true for a tracked non-vector glyph, \
         preventing redundant outline extraction attempts"
    );
}

#[test]
fn non_vector_entry_does_not_shadow_real_entries() {
    let mut map = GlyphMap::new();

    let non_vector_key = GlyphKey {
        font_id: cosmic_text::fontdb::ID::dummy(),
        glyph_id: 1,
        font_weight: 400,
        cache_key_flags: cosmic_text::CacheKeyFlags::empty(),
    };

    let real_key = GlyphKey {
        font_id: cosmic_text::fontdb::ID::dummy(),
        glyph_id: 2,
        font_weight: 400,
        cache_key_flags: cosmic_text::CacheKeyFlags::empty(),
    };

    let real_entry = GlyphEntry {
        band_offset: 42,
        band_max_x: 3,
        band_max_y: 4,
        band_transform: [1.0, 2.0, 3.0, 4.0],
        bounds: [0.0, 0.0, 100.0, 100.0],
    };

    map.insert(non_vector_key, NON_VECTOR_GLYPH);
    map.insert(real_key, real_entry);

    // Non-vector entry is still non-vector.
    assert!(map.get(&non_vector_key).expect("should have non_vector entry").is_non_vector());

    // Real entry is NOT non-vector.
    let got = map.get(&real_key).expect("should have real entry");
    assert!(
        !got.is_non_vector(),
        "A real glyph entry should not be flagged as non-vector"
    );
    assert_eq!(got.band_offset, 42);
}
