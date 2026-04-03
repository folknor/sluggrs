#![allow(clippy::unwrap_used)]
//! Hotpath profiling binary for brokkr integration.
//!
//! Exercises both cache-miss (cold) and cache-hit (warm) rendering paths
//! through the public TextRenderer::prepare API. The hotpath guard captures
//! per-function timing/allocation data and writes it on exit.
//!
//! Run via brokkr:  brokkr sluggrs hotpath [--alloc]
//! Run standalone:  cargo run --release --example hotpath --features hotpath

use std::time::Instant;

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping};
use sluggrs::{
    Cache, ColorMode, Resolution, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
    Viewport,
};

fn main() {
    let _guard = hotpath::HotpathGuardBuilder::new("sluggrs::hotpath")
        .percentiles(&[50, 95, 99])
        .with_functions_limit(0)
        .build();

    let (device, queue) = create_device();

    let mut harness = RenderHarness::new(&device, &queue);

    // A paragraph of mixed Latin text to exercise many distinct glyphs
    let cold_text = concat!(
        "The quick brown fox jumps over the lazy dog. ",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz ",
        "0123456789 !@#$%^&*()_+-=[]{}|;':\",./<>? ",
        "Pack my box with five dozen liquor jugs. ",
        "How vexingly quick daft zebras jump!",
    );

    // -- Cold path: first prepare with empty cache --
    let cold_start = Instant::now();
    harness
        .prepare_text(cold_text)
        .expect("Cold prepare should succeed");
    let cold_us = cold_start.elapsed().as_micros();

    let distinct_glyphs = harness.atlas.glyph_count();

    // -- Warm path: re-prepare with fully populated cache --
    let warm_iterations = 20u32;
    let warm_start = Instant::now();
    for _ in 0..warm_iterations {
        harness
            .prepare_text(cold_text)
            .expect("Warm prepare should succeed");
    }
    let warm_avg_us = warm_start.elapsed().as_micros() / warm_iterations as u128;

    // -- Mixed path: partially overlapping text --
    let mixed_text = "Sphinx of black quartz, judge my vow. 0123456789";
    let mixed_iterations = 10u32;
    let mixed_start = Instant::now();
    for _ in 0..mixed_iterations {
        harness
            .prepare_text(mixed_text)
            .expect("Mixed prepare should succeed");
    }
    let mixed_avg_us = mixed_start.elapsed().as_micros() / mixed_iterations as u128;

    // -- GPU render: measure actual fragment shader time --
    // Warmup frames to flush the profiler pipeline
    for _ in 0..5 {
        harness.render_gpu();
    }
    // Collect 10 measurements, take the median
    let mut gpu_times: Vec<f64> = Vec::new();
    for _ in 0..10 {
        if let Some(ms) = harness.render_gpu() {
            gpu_times.push(ms);
        }
    }
    gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let final_glyphs = harness.atlas.glyph_count();
    let buffer_elements = harness.atlas.buffer_elements_used();

    // -- Emit KV pairs to stderr for brokkr capture --
    eprintln!("distinct_glyphs={distinct_glyphs}");
    eprintln!("final_glyphs={final_glyphs}");
    eprintln!("buffer_elements={buffer_elements}");
    eprintln!("cold_prepare_us={cold_us}");
    eprintln!("warm_prepare_avg_us={warm_avg_us}");
    eprintln!("mixed_prepare_avg_us={mixed_avg_us}");
    eprintln!("warm_iterations={warm_iterations}");
    eprintln!("mixed_iterations={mixed_iterations}");

    // Buffer memory: 16 bytes per vec4<i32> element
    let buffer_bytes = buffer_elements as u64 * 16;
    eprintln!("buffer_bytes={buffer_bytes}");

    if let Some(median) = gpu_times.get(gpu_times.len() / 2) {
        eprintln!("gpu_text_render_us={}", (*median * 1000.0) as u64);
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

    let mut features = wgpu::Features::empty();
    if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if adapter
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    {
        features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
    }

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sluggrs hotpath"),
        required_features: features,
        ..Default::default()
    }))
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
    _render_target: wgpu::Texture,
    render_view: wgpu::TextureView,
    gpu_profiler: Option<wgpu_profiler::GpuProfiler>,
}

impl RenderHarness {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let cache = Cache::new(device);
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let mut atlas =
            TextAtlas::with_color_mode(device, queue, &cache, format, ColorMode::Accurate);
        let renderer =
            TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);
        let mut viewport = Viewport::new(device, &cache);
        viewport.update(
            queue,
            Resolution {
                width: 1920,
                height: 1080,
            },
        );

        let render_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen render target"),
            size: wgpu::Extent3d {
                width: 1920,
                height: 1080,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let render_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

        let gpu_profiler = if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            Some(
                wgpu_profiler::GpuProfiler::new(
                    device,
                    wgpu_profiler::GpuProfilerSettings {
                        enable_timer_queries: true,
                        enable_debug_groups: false,
                        max_num_pending_frames: 8,
                    },
                )
                .expect("Failed to create GPU profiler"),
            )
        } else {
            None
        };

        Self {
            renderer,
            atlas,
            viewport,
            font_system: FontSystem::new(),
            swash_cache: SwashCache::new(),
            device: device.clone(),
            queue: queue.clone(),
            _render_target: render_target,
            render_view,
            gpu_profiler,
        }
    }

    /// Run an actual GPU render pass and measure fragment shader time.
    fn render_gpu(&mut self) -> Option<f64> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gpu profiling pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            let query = self
                .gpu_profiler
                .as_ref()
                .map(|p| p.begin_query("text_render", &mut pass));

            self.renderer
                .render(&self.atlas, &self.viewport, &mut pass)
                .unwrap();

            if let (Some(profiler), Some(query)) = (&self.gpu_profiler, query) {
                profiler.end_query(&mut pass, query);
            }
        }

        if let Some(profiler) = &mut self.gpu_profiler {
            profiler.resolve_queries(&mut encoder);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        if let Some(profiler) = &mut self.gpu_profiler {
            let _ = profiler.end_frame();
            if let Some(results) =
                profiler.process_finished_frame(self.queue.get_timestamp_period())
            {
                for r in &results {
                    if let Some(time) = &r.time {
                        return Some((time.end - time.start) * 1000.0);
                    }
                }
            }
        }
        None
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
