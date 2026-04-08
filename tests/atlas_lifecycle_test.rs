//! Integration tests for TextAtlas trim() semantics and texture growth invariants.
//!
//! These tests require a wgpu Device and Queue (GPU or software renderer).
//! They are marked #[ignore] because CI environments may lack GPU/software
//! rendering support. Run manually with:
//!
//!     cargo test --test atlas_lifecycle_test -- --ignored --nocapture

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping};
use sluggrs::{
    Cache, ColorMode, Resolution, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
    Viewport,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_test_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: true,
    }))
    .expect("Failed to find adapter - this test requires a GPU or software renderer");

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("Failed to create device")
}

/// Set up the full rendering pipeline objects needed to exercise TextAtlas
/// through the public TextRenderer::prepare API.
struct TestHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    atlas: TextAtlas,
    renderer: TextRenderer,
    viewport: Viewport,
    font_system: FontSystem,
    swash_cache: SwashCache,
}

impl TestHarness {
    fn new() -> Self {
        let (device, queue) = create_test_device();
        let cache = Cache::new(&device);
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let mut atlas =
            TextAtlas::with_color_mode(&device, &queue, &cache, format, ColorMode::Accurate);
        let renderer =
            TextRenderer::new(&mut atlas, &device, wgpu::MultisampleState::default(), None);
        let mut viewport = Viewport::new(&device, &cache);
        viewport.update(
            &queue,
            Resolution {
                width: 800,
                height: 600,
            },
        );
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();

        Self {
            device,
            queue,
            atlas,
            renderer,
            viewport,
            font_system,
            swash_cache,
        }
    }

    /// Prepare a text area containing the given string. Returns Ok(()) on success.
    /// This drives glyph extraction, outline preparation, and atlas upload
    /// through the public TextRenderer::prepare path.
    fn prepare_text(&mut self, text: &str) -> Result<(), sluggrs::PrepareError> {
        let metrics = Metrics::new(24.0, 30.0);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        buffer.set_text(
            &mut self.font_system,
            text,
            &Attrs::new(),
            Shaping::Advanced,
            None,
        );
        buffer.shape_until_scroll(&mut self.font_system, false);

        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let text_area = TextArea {
            buffer: &buffer,
            left: 0.0,
            top: 0.0,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: 0,
                right: 800,
                bottom: 600,
            },
            default_color: cosmic_text::Color::rgb(0, 0, 0),
        };

        self.renderer.prepare(
            &self.device,
            &self.queue,
            &encoder,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            [text_area],
            &mut self.swash_cache,
        )
    }
}

// ---------------------------------------------------------------------------
// Test 1: trim() retains cached glyphs
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn trim_retains_cached_glyphs() {
    let mut h = TestHarness::new();

    // Upload glyphs for "Hello" into the atlas
    h.prepare_text("Hello")
        .expect("First prepare should succeed");

    // Trim the atlas - since trim() is documented to retain cached data,
    // subsequent prepare of the same text should still succeed without
    // needing to re-extract outlines.
    h.atlas.trim();

    // Prepare the same text again. If trim() had cleared the glyph cache,
    // this would still work (re-extraction), but we verify it does not panic
    // and succeeds cleanly.
    h.prepare_text("Hello")
        .expect("Prepare after trim() should succeed");

    // Prepare different text that shares some glyphs with the first batch
    // ("l" and "o" overlap with "Hello"). This exercises the cache-hit path
    // for previously uploaded glyphs that survived trim().
    h.prepare_text("lo world")
        .expect("Prepare with overlapping glyphs after trim() should succeed");
}

// ---------------------------------------------------------------------------
// Test 2: Texture growth preserves offsets (observable via stable rendering)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn texture_growth_preserves_offsets() {
    let mut h = TestHarness::new();

    // Upload a batch of glyphs that will be cached
    h.prepare_text("ABCDEFGHIJ")
        .expect("Initial prepare should succeed");

    // Now upload a large number of distinct glyphs to force texture growth.
    // The curve texture starts at width 4096 x height 1 (4096 texels).
    // Each glyph uses ~2 texels per curve and typical glyphs have 10-40 curves,
    // so ~20-80 texels per glyph. With ~100-200 distinct glyphs we should
    // overflow the initial 4096-texel row.
    //
    // We use a wide variety of Unicode characters to maximize distinct glyph IDs.
    // The system font should cover basic Latin, extended Latin, and common symbols.
    let growth_text = concat!(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz",
        "0123456789",
        "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        // Extended Latin characters (if the system font supports them)
        "\u{00C0}\u{00C1}\u{00C2}\u{00C3}\u{00C4}\u{00C5}\u{00C6}\u{00C7}",
        "\u{00C8}\u{00C9}\u{00CA}\u{00CB}\u{00CC}\u{00CD}\u{00CE}\u{00CF}",
        "\u{00D0}\u{00D1}\u{00D2}\u{00D3}\u{00D4}\u{00D5}\u{00D6}\u{00D8}",
        "\u{00D9}\u{00DA}\u{00DB}\u{00DC}\u{00DD}\u{00DE}\u{00DF}",
        "\u{00E0}\u{00E1}\u{00E2}\u{00E3}\u{00E4}\u{00E5}\u{00E6}\u{00E7}",
        "\u{00E8}\u{00E9}\u{00EA}\u{00EB}\u{00EC}\u{00ED}\u{00EE}\u{00EF}",
        "\u{00F0}\u{00F1}\u{00F2}\u{00F3}\u{00F4}\u{00F5}\u{00F6}\u{00F8}",
        "\u{00F9}\u{00FA}\u{00FB}\u{00FC}\u{00FD}\u{00FE}\u{00FF}",
    );

    h.prepare_text(growth_text)
        .expect("Growth prepare should succeed");

    // Now re-prepare the original text. If texture growth had corrupted the
    // earlier glyph entries (e.g. stale band_offset pointing into a destroyed
    // texture), this would produce incorrect GlyphInstances or panic.
    // The atlas caches entries by GlyphKey, so previously uploaded glyphs
    // should still reference valid offsets after growth (because grow_*_texture
    // re-uploads the CPU-side data copies into the new, larger texture).
    h.prepare_text("ABCDEFGHIJ")
        .expect("Re-prepare after growth should succeed");
}

// ---------------------------------------------------------------------------
// Test 3: Multiple trim cycles don't corrupt state
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn multiple_trim_cycles_stable() {
    let mut h = TestHarness::new();

    // Simulate multiple frame cycles where trim() is called each frame
    for i in 0..5 {
        let text = match i % 3 {
            0 => "Frame zero text",
            1 => "Different frame one",
            _ => "Yet another frame",
        };

        h.prepare_text(text)
            .unwrap_or_else(|_| panic!("Prepare at cycle {i} should succeed"));
        h.atlas.trim();
    }

    // Final prepare after several trim cycles should still work
    h.prepare_text("Final text after many trims")
        .expect("Final prepare should succeed");
}

// ---------------------------------------------------------------------------
// Test 4: trim() on empty atlas is safe
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn trim_empty_atlas_is_safe() {
    let mut h = TestHarness::new();

    // Trim before any glyphs have been uploaded
    h.atlas.trim();

    // Atlas should still be usable after trimming empty state
    h.prepare_text("Works after empty trim")
        .expect("Prepare after trimming empty atlas should succeed");
}

// ---------------------------------------------------------------------------
// Test 5: Growth + trim interleaved
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn growth_then_trim_then_more_glyphs() {
    let mut h = TestHarness::new();

    // Force texture growth with many distinct glyphs
    let many_chars: String = ('A'..='z').collect();
    h.prepare_text(&many_chars)
        .expect("Initial large prepare should succeed");

    // Trim
    h.atlas.trim();

    // Add more glyphs (these may land in the grown texture)
    h.prepare_text("0123456789!@#$%")
        .expect("Prepare after growth+trim should succeed");

    // Trim again
    h.atlas.trim();

    // Verify everything still works
    h.prepare_text("Final check")
        .expect("Final prepare should succeed");
}

// ---------------------------------------------------------------------------
// Test 6: trim() does not reset when textures haven't grown
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn trim_does_not_reset_without_texture_growth() {
    let mut h = TestHarness::new();

    // Upload a small number of glyphs — not enough to trigger texture growth
    h.prepare_text("Hi").expect("Prepare should succeed");

    let glyph_count_before = h.atlas.glyph_count();
    assert!(glyph_count_before > 0);

    // Trim with no glyphs marked as in-use (we didn't call prepare again
    // after the last trim). Even though in_use < cached / 2, the textures
    // haven't grown beyond initial size, so no reset should happen.
    h.atlas.trim();

    // Glyphs should still be cached
    assert_eq!(
        h.atlas.glyph_count(),
        glyph_count_before,
        "trim() should not evict when textures haven't grown"
    );
}

// ---------------------------------------------------------------------------
// Test 7: trim() resets when textures grew and working set shifted
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn trim_resets_when_textures_grew_and_working_set_shifted() {
    let mut h = TestHarness::new();

    // Upload enough distinct glyphs to force texture growth.
    let many_chars = concat!(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz",
        "0123456789",
        "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        "\u{00C0}\u{00C1}\u{00C2}\u{00C3}\u{00C4}\u{00C5}\u{00C6}\u{00C7}",
        "\u{00C8}\u{00C9}\u{00CA}\u{00CB}\u{00CC}\u{00CD}\u{00CE}\u{00CF}",
        "\u{00D0}\u{00D1}\u{00D2}\u{00D3}\u{00D4}\u{00D5}\u{00D6}\u{00D8}",
        "\u{00D9}\u{00DA}\u{00DB}\u{00DC}\u{00DD}\u{00DE}\u{00DF}",
        "\u{00E0}\u{00E1}\u{00E2}\u{00E3}\u{00E4}\u{00E5}\u{00E6}\u{00E7}",
        "\u{00E8}\u{00E9}\u{00EA}\u{00EB}\u{00EC}\u{00ED}\u{00EE}\u{00EF}",
    );
    h.prepare_text(many_chars)
        .expect("Growth prepare should succeed");

    let cached_after_growth = h.atlas.glyph_count();
    assert!(cached_after_growth > 50, "Should have cached many glyphs");

    // Now prepare only a small subset — the working set has shifted.
    // This marks only a few glyphs as in-use.
    h.prepare_text("AB").expect("Small prepare should succeed");

    // Trim: textures grew + in_use < cached / 2 → should reset
    h.atlas.trim();

    // After reset, the glyph cache should be empty (all evicted).
    assert_eq!(
        h.atlas.glyph_count(),
        0,
        "trim() should reset when textures grew and working set shifted"
    );

    // Buffer should be back at initial size
    assert_eq!(
        h.atlas.buffer_elements_used(),
        0,
        "buffer cursor should be reset"
    );

    // The atlas should still be usable — next prepare re-extracts
    h.prepare_text("AB")
        .expect("Prepare after reset should succeed");
    assert!(
        h.atlas.glyph_count() > 0,
        "Glyphs should be re-uploaded after reset"
    );
}

// ---------------------------------------------------------------------------
// Test 8: trim() does not reset when working set is stable
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn trim_does_not_reset_when_working_set_stable() {
    let mut h = TestHarness::new();

    // Upload enough to trigger texture growth
    let many_chars = concat!(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz",
        "0123456789",
        "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        "\u{00C0}\u{00C1}\u{00C2}\u{00C3}\u{00C4}\u{00C5}\u{00C6}\u{00C7}",
        "\u{00C8}\u{00C9}\u{00CA}\u{00CB}\u{00CC}\u{00CD}\u{00CE}\u{00CF}",
        "\u{00D0}\u{00D1}\u{00D2}\u{00D3}\u{00D4}\u{00D5}\u{00D6}\u{00D8}",
        "\u{00D9}\u{00DA}\u{00DB}\u{00DC}\u{00DD}\u{00DE}\u{00DF}",
        "\u{00E0}\u{00E1}\u{00E2}\u{00E3}\u{00E4}\u{00E5}\u{00E6}\u{00E7}",
        "\u{00E8}\u{00E9}\u{00EA}\u{00EB}\u{00EC}\u{00ED}\u{00EE}\u{00EF}",
    );
    h.prepare_text(many_chars)
        .expect("Growth prepare should succeed");

    let cached_after_growth = h.atlas.glyph_count();

    // Prepare the SAME text again — all glyphs are in-use
    h.prepare_text(many_chars)
        .expect("Repeat prepare should succeed");

    // Trim: textures grew, but in_use >= cached / 2 → should NOT reset
    h.atlas.trim();

    assert_eq!(
        h.atlas.glyph_count(),
        cached_after_growth,
        "trim() should not evict when working set is stable (all glyphs in use)"
    );
}
