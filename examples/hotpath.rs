#![allow(clippy::unwrap_used)]
//! Hotpath profiling binary for brokkr integration.
//!
//! Exercises both cache-miss (cold) and cache-hit (warm) rendering paths
//! through the public TextRenderer::prepare API. The hotpath guard captures
//! per-function timing/allocation data and writes it on exit.
//!
//! Run via brokkr:  brokkr sluggrs hotpath [--alloc]
//! Run standalone:  cargo run --release --example hotpath --features hotpath

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping};
use sluggrs::{
    Cache, ColorMode, Resolution, SwashCache, TextArea, TextBounds, TextAtlas, TextRenderer,
    Viewport,
};

fn main() {
    #[cfg(feature = "hotpath")]
    let _guard = hotpath::HotpathGuardBuilder::new("sluggrs::hotpath")
        .percentiles(&[50, 95, 99])
        .with_functions_limit(0)
        .build();

    let (device, queue) = create_device();

    // -- Cold path: first prepare with empty cache --
    let mut harness = RenderHarness::new(&device, &queue);

    // A paragraph of mixed Latin text to exercise many distinct glyphs
    let cold_text = concat!(
        "The quick brown fox jumps over the lazy dog. ",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz ",
        "0123456789 !@#$%^&*()_+-=[]{}|;':\",./<>? ",
        "Pack my box with five dozen liquor jugs. ",
        "How vexingly quick daft zebras jump!",
    );

    harness
        .prepare_text(cold_text)
        .expect("Cold prepare should succeed");

    // -- Warm path: re-prepare with fully populated cache --
    // Run multiple iterations to capture cache-hit performance
    for _ in 0..20 {
        harness
            .prepare_text(cold_text)
            .expect("Warm prepare should succeed");
    }

    // -- Mixed path: partially overlapping text --
    let mixed_text = "Sphinx of black quartz, judge my vow. 0123456789";
    for _ in 0..10 {
        harness
            .prepare_text(mixed_text)
            .expect("Mixed prepare should succeed");
    }
}

fn create_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapter found");

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("Failed to create device")
}

struct RenderHarness {
    renderer: TextRenderer,
    atlas: TextAtlas,
    viewport: Viewport,
    font_system: FontSystem,
    swash_cache: SwashCache,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl RenderHarness {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let cache = Cache::new(device);
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let mut atlas =
            TextAtlas::with_color_mode(device, queue, &cache, format, ColorMode::Accurate);
        let renderer = TextRenderer::new(
            &mut atlas,
            device,
            wgpu::MultisampleState::default(),
            None,
        );
        let mut viewport = Viewport::new(device, &cache);
        viewport.update(
            queue,
            Resolution {
                width: 1920,
                height: 1080,
            },
        );

        // Clone device/queue handles for prepare calls
        // (wgpu Device and Queue are internally Arc'd)
        Self {
            renderer,
            atlas,
            viewport,
            font_system: FontSystem::new(),
            swash_cache: SwashCache::new(),
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    fn prepare_text(&mut self, text: &str) -> Result<(), sluggrs::PrepareError> {
        let metrics = Metrics::new(16.0, 20.0);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        buffer.set_text(
            &mut self.font_system,
            text,
            &Attrs::new(),
            Shaping::Advanced,
            None,
        );
        buffer.shape_until_scroll(&mut self.font_system, false);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let text_area = TextArea {
            buffer: &buffer,
            left: 10.0,
            top: 10.0,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: 0,
                right: 1920,
                bottom: 1080,
            },
            default_color: cosmic_text::Color::rgb(255, 255, 255),
        };

        self.renderer.prepare(
            &self.device,
            &self.queue,
            &mut encoder,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            [text_area],
            &mut self.swash_cache,
        )
    }
}
