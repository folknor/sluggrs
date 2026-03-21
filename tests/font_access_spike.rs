//! Phase 0 spike: prove the path from cosmic_text layout → font bytes → skrifa outline.
//!
//! Run with: cargo test --test font_access_spike -- --nocapture

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping};
use sluggrs::outline::extract_outline;
use sluggrs::prepare::prepare_outline;

#[test]
fn extract_outline_from_cosmic_text_layout() {
    let mut font_system = FontSystem::new();

    // Shape some text to get LayoutGlyphs
    let metrics = Metrics::new(24.0, 30.0);
    let mut buffer = Buffer::new(&mut font_system, metrics);
    buffer.set_text(&mut font_system, "Hello, Slug!", &Attrs::new(), Shaping::Advanced, None);
    buffer.shape_until_scroll(&mut font_system, false);

    let mut glyphs_extracted = 0;
    let mut glyphs_total = 0;

    for run in buffer.layout_runs() {
        for glyph in run.glyphs {
            glyphs_total += 1;

            // Get font data through cosmic_text
            let font = font_system
                .get_font(glyph.font_id, glyph.font_weight)
                .expect("Font should be available");

            let font_data = font.data();
            assert!(!font_data.is_empty(), "Font data should not be empty");

            // Verify skrifa can parse the font bytes
            let skrifa_font = skrifa::FontRef::new(font_data)
                .expect("skrifa should parse the font bytes");

            // Verify we can get units_per_em
            let units_per_em = {
                use skrifa::raw::TableProvider;
                skrifa_font.head().expect("head table").units_per_em()
            };
            assert!(units_per_em > 0, "units_per_em should be positive");

            // Extract outline
            match extract_outline(font_data, 0, glyph.glyph_id, &[]) {
                Some(outline) => {
                    assert!(!outline.curves.is_empty(), "Outline should have curves");
                    assert!(
                        outline.bounds[2] > outline.bounds[0],
                        "Bounds width should be positive"
                    );

                    // Prepare for GPU
                    let gpu_outline = prepare_outline(&outline);
                    assert_eq!(
                        gpu_outline.curves.len(),
                        outline.curves.len(),
                        "GPU outline should have same curve count"
                    );

                    glyphs_extracted += 1;

                    println!(
                        "  glyph_id={}, font_weight={}, flags={:?}, curves={}, bounds=[{:.0},{:.0},{:.0},{:.0}]",
                        glyph.glyph_id,
                        glyph.font_weight.0,
                        glyph.cache_key_flags,
                        outline.curves.len(),
                        outline.bounds[0], outline.bounds[1],
                        outline.bounds[2], outline.bounds[3],
                    );
                }
                None => {
                    // Space or non-drawing glyph — acceptable
                    println!(
                        "  glyph_id={} — no outline (space or non-drawing)",
                        glyph.glyph_id
                    );
                }
            }
        }
    }

    println!(
        "\nResult: {glyphs_extracted}/{glyphs_total} glyphs extracted successfully"
    );
    assert!(glyphs_extracted > 0, "Should extract at least some glyphs");
    // "Hello, Slug!" has 10 visible glyphs (excluding spaces), all should have outlines
    assert!(
        glyphs_extracted >= 10,
        "Expected at least 10 outlines, got {glyphs_extracted}"
    );
}

#[test]
fn glyph_key_fields_available() {
    // Verify all GlyphKey fields are accessible from LayoutGlyph
    let mut font_system = FontSystem::new();
    let metrics = Metrics::new(24.0, 30.0);
    let mut buffer = Buffer::new(&mut font_system, metrics);
    buffer.set_text(&mut font_system, "A", &Attrs::new(), Shaping::Advanced, None);
    buffer.shape_until_scroll(&mut font_system, false);

    let run = buffer.layout_runs().next().expect("Should have a layout run");
    let glyph = &run.glyphs[0];

    // These are the fields we need for GlyphKey
    let _font_id: cosmic_text::fontdb::ID = glyph.font_id;
    let _glyph_id: u16 = glyph.glyph_id;
    let _font_weight: cosmic_text::fontdb::Weight = glyph.font_weight;
    let _cache_key_flags: cosmic_text::CacheKeyFlags = glyph.cache_key_flags;

    println!("GlyphKey fields available:");
    println!("  font_id: {_font_id:?}");
    println!("  glyph_id: {_glyph_id}");
    println!("  font_weight: {}", _font_weight.0);
    println!("  cache_key_flags: {_cache_key_flags:?}");
}
