//! Integration tests for TextRenderer::prepare() behavior.
//!
//! These tests require a wgpu Device and Queue (GPU or software renderer).
//! They are marked #[ignore] because CI environments may lack GPU/software
//! rendering support. Run manually with:
//!
//!     cargo test --test prepare_behavior_test -- --ignored --nocapture

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping};
use sluggrs::{
    Cache, Color, ColorMode, Resolution, SwashCache, TextArea, TextAtlas, TextBorder, TextBounds,
    TextRenderer, Viewport,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_test_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: true,
    }))
    .expect("Failed to find adapter - this test requires a GPU or software renderer");

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("Failed to create device")
}

/// Full rendering pipeline objects needed to exercise prepare() through the
/// public API.
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

    /// Create a shaped Buffer containing the given text.
    fn make_buffer(&mut self, text: &str) -> Buffer {
        let metrics = Metrics::new(24.0, 30.0);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        buffer.set_text(text, &Attrs::new(), Shaping::Advanced, None);
        buffer.shape_until_scroll(&mut self.font_system, false);
        buffer
    }

    /// Run prepare() with a TextArea built from the given buffer and bounds.
    fn prepare_with_bounds(
        &mut self,
        buffer: &Buffer,
        bounds: TextBounds,
    ) -> Result<(), sluggrs::PrepareError> {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let text_area = TextArea {
            buffer,
            left: 0.0,
            top: 0.0,
            scale: 1.0,
            bounds,
            default_color: Color::rgb(255, 255, 255),
            border: None,
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

    /// Run prepare() with default (full-viewport) bounds.
    fn prepare_text(&mut self, text: &str) -> Result<(), sluggrs::PrepareError> {
        let buffer = self.make_buffer(text);
        self.prepare_with_bounds(
            &buffer,
            TextBounds {
                left: 0,
                top: 0,
                right: 800,
                bottom: 600,
            },
        )
    }

    /// Run prepare_with_depth() with the given metadata_to_depth closure.
    fn prepare_text_with_depth(
        &mut self,
        text: &str,
        metadata_to_depth: impl FnMut(usize) -> f32,
    ) -> Result<(), sluggrs::PrepareError> {
        let buffer = self.make_buffer(text);
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
            default_color: Color::rgb(255, 255, 255),
            border: None,
        };

        self.renderer.prepare_with_depth(
            &self.device,
            &self.queue,
            &encoder,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            [text_area],
            &mut self.swash_cache,
            metadata_to_depth,
        )
    }
}

// ---------------------------------------------------------------------------
// Test 1: Repeated prepare does not grow cache (cache-hit path works)
// ---------------------------------------------------------------------------

/// Preparing the same text twice in sequence should succeed both times.
/// The second call exercises the cache-hit path where glyphs are already
/// in the atlas and do not need re-extraction or re-upload.
#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn repeated_prepare_hits_cache() {
    let mut h = TestHarness::new();

    // First prepare: cold cache, glyphs are extracted and uploaded.
    h.prepare_text("Hello")
        .expect("First prepare should succeed");

    // Second prepare of the same text: all glyphs should be cache hits.
    // prepare() calls instances.clear() internally, so this is a fresh
    // instance list built from cached atlas entries.
    h.prepare_text("Hello")
        .expect("Second prepare (cache-hit path) should succeed");

    // Third prepare with partially overlapping text to exercise a mix of
    // cache hits ("l", "o") and cold misses ("w", "r", "d", " ").
    h.prepare_text("lo world")
        .expect("Third prepare (partial cache overlap) should succeed");

    println!("repeated_prepare_hits_cache: all three prepare calls succeeded");
}

// ---------------------------------------------------------------------------
// Test 2: Depth contract documentation (prepare_with_depth)
// ---------------------------------------------------------------------------

/// prepare_with_depth accepts a metadata_to_depth closure that maps glyph
/// metadata to a depth value. Currently the depth value is computed but
/// stored in a local `_depth` variable and NOT plumbed through to the
/// GlyphInstance or the shader.
///
/// TODO: depth is not yet plumbed through to rendering. When depth support
/// is added, this test should be extended to verify that different metadata
/// values produce GlyphInstances at different depth layers.
#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn prepare_with_depth_does_not_panic() {
    let mut h = TestHarness::new();

    // Supply a metadata_to_depth that returns different values for different
    // metadata. Since cosmic_text glyph metadata defaults to 0, this will
    // be called with 0 for every glyph in practice, but the closure itself
    // should not cause any issues.
    let result = h.prepare_text_with_depth("Depth test", |metadata| match metadata {
        0 => 0.0,
        1 => 0.5,
        _ => 1.0,
    });

    result.expect("prepare_with_depth should succeed regardless of depth values");

    // Also verify that an identity depth function works (the default path
    // that prepare() uses internally via zero_depth).
    let result2 = h.prepare_text_with_depth("Another depth test", |_| 0.0);
    result2.expect("prepare_with_depth with zero depth should succeed");

    println!("prepare_with_depth_does_not_panic: both calls succeeded (depth is currently unused)");
}

// ---------------------------------------------------------------------------
// Test 3: Clipping semantics - glyph outside bounds still prepared
// ---------------------------------------------------------------------------

/// sluggrs relies on scissor rect clipping at the GPU level, not per-glyph
/// cropping on the CPU. Glyphs that fall partially or fully outside the
/// TextBounds are skipped during instance generation (the bounding-box
/// check in prepare_with_depth), but this should never cause a panic or
/// error - it simply means fewer instances are emitted.
///
/// This test verifies that prepare() returns Ok even when the TextBounds
/// are tight enough to exclude some or all glyphs.
#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn clipping_bounds_do_not_cause_errors() {
    let mut h = TestHarness::new();

    let buffer = h.make_buffer("Hello, World! This is a long line of text.");

    // Case 1: Very tight bounds that only cover the first few pixels.
    // Some glyphs may partially overlap, others will be fully outside.
    let tight_bounds = TextBounds {
        left: 0,
        top: 0,
        right: 30,
        bottom: 30,
    };
    h.prepare_with_bounds(&buffer, tight_bounds)
        .expect("Prepare with tight bounds should succeed");

    // Case 2: Bounds that exclude the text entirely (text starts at y=0
    // but bounds start far below).
    // Note: sluggrs skips glyphs outside bounds via a bounding-box check,
    // so this should produce zero instances but still return Ok.
    let disjoint_bounds = TextBounds {
        left: 0,
        top: 500,
        right: 800,
        bottom: 600,
    };
    h.prepare_with_bounds(&buffer, disjoint_bounds)
        .expect("Prepare with disjoint bounds should succeed");

    // Case 3: Zero-area bounds.
    let zero_bounds = TextBounds {
        left: 100,
        top: 100,
        right: 100,
        bottom: 100,
    };
    h.prepare_with_bounds(&buffer, zero_bounds)
        .expect("Prepare with zero-area bounds should succeed");

    println!(
        "clipping_bounds_do_not_cause_errors: all bound configurations succeeded \
         (sluggrs relies on scissor rect clipping, not per-glyph cropping)"
    );
}

// ---------------------------------------------------------------------------
// Retained-cache placement regressions (scroll-aware culling rework)
// ---------------------------------------------------------------------------

/// Prepare one frame containing two TextAreas that share a single buffer at
/// different horizontal placements. Returns the emitted vector instances.
fn prepare_shared_buffer_frame(
    h: &mut TestHarness,
    buffer: &cosmic_text::Buffer,
    left_a: f32,
    left_b: f32,
) -> Vec<sluggrs::GlyphInstance> {
    let encoder = h
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let bounds = TextBounds {
        left: 0,
        top: 0,
        right: 800,
        bottom: 600,
    };
    let area = |left: f32| TextArea {
        buffer,
        left,
        top: 0.0,
        scale: 1.0,
        bounds,
        default_color: Color::rgb(255, 255, 255),
        border: None,
    };
    h.renderer
        .prepare(
            &h.device,
            &h.queue,
            &encoder,
            &mut h.font_system,
            &mut h.atlas,
            &h.viewport,
            [area(left_a), area(left_b)],
            &mut h.swash_cache,
        )
        .expect("prepare should succeed");
    h.renderer.prepared_instances().to_vec()
}

/// Two TextAreas sharing one buffer (iced deduplicates identical text) must
/// keep distinct placements across retained-cache frames. Regression test for
/// mid-emission cache writes leaking one area's re-culled placement into a
/// later plan for the same buffer pointer.
#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn shared_buffer_areas_keep_distinct_placements() {
    let mut h = TestHarness::new();
    let mut buffer = h.make_buffer("AB");
    buffer.set_redraw(false);

    for frame in 0..3 {
        let instances = prepare_shared_buffer_frame(&mut h, &buffer, 0.0, 100.0);
        assert!(!instances.is_empty(), "frame {frame}: no instances emitted");
        assert_eq!(
            instances.len() % 2,
            0,
            "frame {frame}: expected two equal-sized area emissions"
        );
        let half = instances.len() / 2;
        for i in 0..half {
            let delta = instances[half + i].screen_rect[0] - instances[i].screen_rect[0];
            assert!(
                (delta - 100.0).abs() < 0.01,
                "frame {frame}: instance {i} placement delta {delta}, expected 100"
            );
        }
    }
}

/// The original demo2 symptom: content culled under one scroll offset must
/// reappear when the scroll offset changes, instead of the retained cache
/// replaying the stale culled list.
#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn scrolled_content_reappears_after_cache_hit() {
    let mut h = TestHarness::new();
    let metrics = Metrics::new(24.0, 30.0);
    let mut buffer = Buffer::new(&mut h.font_system, metrics);
    buffer.set_text("first\nsecond", &Attrs::new(), Shaping::Advanced, None);
    buffer.shape_until_scroll(&mut h.font_system, false);
    buffer.set_redraw(false);

    // Clip to the first line only: the second line's run is culled and the
    // cached area is incomplete.
    let bounds = TextBounds {
        left: 0,
        top: 0,
        right: 800,
        bottom: 25,
    };
    h.prepare_with_bounds(&buffer, bounds)
        .expect("first prepare should succeed");
    let line2_before = h
        .renderer
        .prepared_instances()
        .iter()
        .filter(|inst| inst.screen_rect[1] > 25.0)
        .count();
    assert_eq!(line2_before, 0, "second line should start culled");

    // Scroll the second line into the clip window and prepare again with
    // identical areas: the placement mismatch must rebuild, not replay.
    h.viewport.set_scroll_offset(&h.queue, [0.0, -30.0]);
    h.prepare_with_bounds(&buffer, bounds)
        .expect("second prepare should succeed");
    let line2_after = h
        .renderer
        .prepared_instances()
        .iter()
        .filter(|inst| inst.screen_rect[1] > 25.0)
        .count();
    assert!(
        line2_after > 0,
        "second line must reappear after scrolling it into view"
    );
}

fn assert_instances_equal(actual: &[sluggrs::GlyphInstance], expected: &[sluggrs::GlyphInstance]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.screen_rect, expected.screen_rect);
        assert_eq!(actual.color, expected.color);
        assert_eq!(actual.glyph_offset, expected.glyph_offset);
        assert_eq!(actual.cmd_texel_count, expected.cmd_texel_count);
        assert_eq!(actual.depth, expected.depth);
        assert_eq!(actual.ppem, expected.ppem);
    }
}

fn prepare_areas<'a>(
    h: &mut TestHarness,
    areas: impl IntoIterator<Item = TextArea<'a>>,
) -> Vec<sluggrs::GlyphInstance> {
    let encoder = h
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    h.renderer
        .prepare(
            &h.device,
            &h.queue,
            &encoder,
            &mut h.font_system,
            &mut h.atlas,
            &h.viewport,
            areas,
            &mut h.swash_cache,
        )
        .expect("prepare should succeed");
    h.renderer.prepared_instances().to_vec()
}

fn full_bounds() -> TextBounds {
    TextBounds {
        left: 0,
        top: 0,
        right: 800,
        bottom: 600,
    }
}

// No deterministic non-vector fixture is available in this repository, so
// shared-buffer raster coverage remains intentionally skipped.

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn shared_buffer_distinct_placements_become_direct_hits() {
    let mut h = TestHarness::new();
    let mut buffer = h.make_buffer("AB");
    buffer.set_redraw(false);
    let area = |left| TextArea {
        buffer: &buffer,
        left,
        top: 0.0,
        scale: 1.0,
        bounds: full_bounds(),
        default_color: Color::rgb(255, 255, 255),
        border: None,
    };

    let first = prepare_areas(&mut h, [area(0.0), area(100.0)]);
    assert_eq!(
        h.renderer.last_prepare_stats(),
        sluggrs::text_renderer::PrepareStats {
            direct_hits: 0,
            reculls: 0,
            misses: 2,
        }
    );
    let second = prepare_areas(&mut h, [area(0.0), area(100.0)]);
    assert_eq!(
        h.renderer.last_prepare_stats(),
        sluggrs::text_renderer::PrepareStats {
            direct_hits: 2,
            reculls: 0,
            misses: 0,
        }
    );
    let third = prepare_areas(&mut h, [area(0.0), area(100.0)]);
    assert_eq!(
        h.renderer.last_prepare_stats(),
        sluggrs::text_renderer::PrepareStats {
            direct_hits: 2,
            reculls: 0,
            misses: 0,
        }
    );
    assert_instances_equal(&second, &first);
    assert_instances_equal(&third, &first);
}

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn shared_buffer_order_swap_converges_after_recull() {
    let mut h = TestHarness::new();
    let mut buffer = h.make_buffer("AB");
    buffer.set_redraw(false);
    let area = |left| TextArea {
        buffer: &buffer,
        left,
        top: 0.0,
        scale: 1.0,
        bounds: full_bounds(),
        default_color: Color::rgb(255, 255, 255),
        border: None,
    };

    let cold = prepare_areas(&mut h, [area(0.0), area(100.0)]);
    assert!(!cold.is_empty());
    assert_eq!(cold.len() % 2, 0, "two equal-sized area emissions expected");
    let half = cold.len() / 2;

    // Swapping the input order must swap the emitted groups: the first half
    // of the swapped frame is the @100 area (cold first half shifted +100,
    // via re-cull, so the x compare needs a rounding tolerance), the second
    // half is the @0 area.
    let swapped = prepare_areas(&mut h, [area(100.0), area(0.0)]);
    assert_eq!(swapped.len(), cold.len());
    assert_instances_shifted(&swapped[..half], &cold[..half], 100.0);
    assert_instances_shifted(&swapped[half..], &cold[half..], -100.0);

    prepare_areas(&mut h, [area(100.0), area(0.0)]);
    assert_eq!(
        h.renderer.last_prepare_stats(),
        sluggrs::text_renderer::PrepareStats {
            direct_hits: 2,
            reculls: 0,
            misses: 0,
        }
    );
}

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn shared_buffer_shrink_grow_discards_stale_occurrences() {
    let mut h = TestHarness::new();
    let mut buffer = h.make_buffer("AB");
    buffer.set_redraw(false);
    let area = |left| TextArea {
        buffer: &buffer,
        left,
        top: 0.0,
        scale: 1.0,
        bounds: full_bounds(),
        default_color: Color::rgb(255, 255, 255),
        border: None,
    };

    prepare_areas(&mut h, [area(0.0), area(100.0)]);
    prepare_areas(&mut h, [area(0.0)]);
    prepare_areas(&mut h, [area(0.0), area(100.0)]);
    assert_eq!(
        h.renderer.last_prepare_stats(),
        sluggrs::text_renderer::PrepareStats {
            direct_hits: 1,
            reculls: 0,
            misses: 1,
        }
    );
}

/// Like `assert_instances_equal`, but expects `actual` to be `expected`
/// shifted by `dx` along x. The shift can come through the re-cull path,
/// whose addition order differs from a fresh build, so x compares with a
/// rounding tolerance; everything else must be exact.
fn assert_instances_shifted(
    actual: &[sluggrs::GlyphInstance],
    expected: &[sluggrs::GlyphInstance],
    dx: f32,
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual.screen_rect[0] - expected.screen_rect[0] - dx).abs() < 0.01);
        assert_eq!(actual.screen_rect[1], expected.screen_rect[1]);
        assert_eq!(actual.screen_rect[2], expected.screen_rect[2]);
        assert_eq!(actual.screen_rect[3], expected.screen_rect[3]);
        assert_eq!(actual.color, expected.color);
        assert_eq!(actual.glyph_offset, expected.glyph_offset);
        assert_eq!(actual.cmd_texel_count, expected.cmd_texel_count);
        assert_eq!(actual.depth, expected.depth);
        assert_eq!(actual.ppem, expected.ppem);
    }
}

/// Returns the warm-frame instances so callers can add variant-specific
/// assertions (e.g. that per-area colors actually differ).
fn assert_validity_variants_become_direct_hits(
    mut make_areas: impl FnMut(&cosmic_text::Buffer) -> [TextArea<'_>; 2],
) -> Vec<sluggrs::GlyphInstance> {
    let mut h = TestHarness::new();
    let mut buffer = h.make_buffer("AB");
    buffer.set_redraw(false);
    let first = prepare_areas(&mut h, make_areas(&buffer));
    let second = prepare_areas(&mut h, make_areas(&buffer));
    assert_eq!(
        h.renderer.last_prepare_stats(),
        sluggrs::text_renderer::PrepareStats {
            direct_hits: 2,
            reculls: 0,
            misses: 0,
        }
    );
    assert_instances_equal(&second, &first);
    second
}

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn shared_buffer_default_color_variants_become_direct_hits() {
    let warm = assert_validity_variants_become_direct_hits(|buffer| {
        [
            TextArea {
                buffer,
                left: 0.0,
                top: 0.0,
                scale: 1.0,
                bounds: full_bounds(),
                default_color: Color::rgb(255, 0, 0),
                border: None,
            },
            TextArea {
                buffer,
                left: 100.0,
                top: 0.0,
                scale: 1.0,
                bounds: full_bounds(),
                default_color: Color::rgb(0, 255, 0),
                border: None,
            },
        ]
    });

    // The two areas must keep their own colors: red first, green second.
    assert!(!warm.is_empty());
    assert_eq!(warm.len() % 2, 0, "two equal-sized area emissions expected");
    let half = warm.len() / 2;
    for instance in &warm[..half] {
        assert_eq!(instance.color, [1.0, 0.0, 0.0, 1.0]);
    }
    for instance in &warm[half..] {
        assert_eq!(instance.color, [0.0, 1.0, 0.0, 1.0]);
    }
}

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn shared_buffer_scale_variants_become_direct_hits() {
    assert_validity_variants_become_direct_hits(|buffer| {
        [
            TextArea {
                buffer,
                left: 0.0,
                top: 0.0,
                scale: 1.0,
                bounds: full_bounds(),
                default_color: Color::rgb(255, 255, 255),
                border: None,
            },
            TextArea {
                buffer,
                left: 100.0,
                top: 0.0,
                scale: 1.5,
                bounds: full_bounds(),
                default_color: Color::rgb(255, 255, 255),
                border: None,
            },
        ]
    });
}

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn shared_buffer_bounds_variants_become_direct_hits() {
    assert_validity_variants_become_direct_hits(|buffer| {
        [
            TextArea {
                buffer,
                left: 0.0,
                top: 0.0,
                scale: 1.0,
                bounds: full_bounds(),
                default_color: Color::rgb(255, 255, 255),
                border: None,
            },
            TextArea {
                buffer,
                left: 100.0,
                top: 0.0,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: 700,
                    bottom: 600,
                },
                default_color: Color::rgb(255, 255, 255),
                border: None,
            },
        ]
    });
}

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn border_color_change_is_a_direct_hit_and_preserves_instances() {
    let mut h = TestHarness::new();
    let mut buffer = h.make_buffer("Outlined");
    buffer.set_redraw(false);
    let area = |color| TextArea {
        buffer: &buffer,
        left: 20.0,
        top: 20.0,
        scale: 1.0,
        bounds: full_bounds(),
        default_color: Color::rgb(255, 255, 255),
        border: Some(TextBorder { color, width: 2.0 }),
    };
    let first = prepare_areas(&mut h, [area(Color::rgb(255, 0, 0))]);
    let second = prepare_areas(&mut h, [area(Color::rgb(0, 0, 255))]);
    assert_instances_equal(&second, &first);
    assert_eq!(h.renderer.last_prepare_stats().direct_hits, 1);
}

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn border_width_change_reculls() {
    let mut h = TestHarness::new();
    let mut buffer = h.make_buffer("Outlined");
    buffer.set_redraw(false);
    let area = |width| TextArea {
        buffer: &buffer,
        left: 20.0,
        top: 20.0,
        scale: 1.0,
        bounds: full_bounds(),
        default_color: Color::rgb(255, 255, 255),
        border: Some(TextBorder {
            color: Color::rgb(0, 0, 0),
            width,
        }),
    };
    prepare_areas(&mut h, [area(1.0)]);
    prepare_areas(&mut h, [area(8.0)]);
    assert_eq!(h.renderer.last_prepare_stats().reculls, 1);
}

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn removing_border_restores_never_bordered_instances() {
    let mut h = TestHarness::new();
    let mut buffer = h.make_buffer("Outlined");
    buffer.set_redraw(false);
    let area = |border| TextArea {
        buffer: &buffer,
        left: 20.0,
        top: 20.0,
        scale: 1.0,
        bounds: full_bounds(),
        default_color: Color::rgb(255, 255, 255),
        border,
    };
    let baseline = prepare_areas(&mut h, [area(None)]);
    prepare_areas(
        &mut h,
        [area(Some(TextBorder {
            color: Color::rgb(0, 0, 0),
            width: 3.0,
        }))],
    );
    let removed = prepare_areas(&mut h, [area(None)]);
    assert_instances_equal(&removed, &baseline);
}
