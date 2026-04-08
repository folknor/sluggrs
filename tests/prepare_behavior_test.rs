//! Integration tests for TextRenderer::prepare() behavior.
//!
//! These tests require a wgpu Device and Queue (GPU or software renderer).
//! They are marked #[ignore] because CI environments may lack GPU/software
//! rendering support. Run manually with:
//!
//!     cargo test --test prepare_behavior_test -- --ignored --nocapture

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping};
use sluggrs::{
    Cache, Color, ColorMode, Resolution, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
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
        buffer.set_text(
            &mut self.font_system,
            text,
            &Attrs::new(),
            Shaping::Advanced,
            None,
        );
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
/// error — it simply means fewer instances are emitted.
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
