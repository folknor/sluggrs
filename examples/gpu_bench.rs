#![allow(clippy::unwrap_used)]
//! GPU-side shader benchmark using wgpu timestamp queries.
//!
//! Measures actual fragment shader execution time for text rendering,
//! independent of CPU-side preparation cost. Renders to an offscreen
//! texture (no window needed).
//!
//! Run: cargo run --release --example gpu-bench

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping};
use sluggrs::{
    Cache, ColorMode, Resolution, SwashCache, TextArea, TextBounds, TextAtlas, TextRenderer,
    Viewport,
};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const WARMUP_FRAMES: u32 = 5;
const BENCH_FRAMES: u32 = 50;

fn main() {
    env_logger::init();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapter found");

    let adapter_name = adapter.get_info().name;
    eprintln!("adapter={adapter_name}");

    let has_timestamps = adapter
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY);

    if !has_timestamps {
        eprintln!("ERROR: adapter does not support TIMESTAMP_QUERY — cannot measure GPU time");
        std::process::exit(1);
    }

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("gpu-bench"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        },
    ))
    .expect("Failed to create device");

    let timestamp_period = queue.get_timestamp_period();

    // Offscreen render target
    let format = wgpu::TextureFormat::Bgra8UnormSrgb;
    let target_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Timestamp infrastructure: query set, resolve buffer, readback buffer
    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts resolve"),
        size: 16, // 2 × u64
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts readback"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Set up sluggrs pipeline
    let cache = Cache::new(&device);
    let mut atlas = TextAtlas::with_color_mode(&device, &queue, &cache, format, ColorMode::Accurate);
    let mut renderer = TextRenderer::new(
        &mut atlas, &device, wgpu::MultisampleState::default(), None,
    );
    let mut viewport = Viewport::new(&device, &cache);
    viewport.update(&queue, Resolution { width: WIDTH, height: HEIGHT });

    let mut font_system = FontSystem::new();
    let mut swash_cache = SwashCache::new();

    // Prepare text
    let text = concat!(
        "The quick brown fox jumps over the lazy dog. ",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz ",
        "0123456789 !@#$%^&*()_+-=[]{}|;':\",./<>? ",
        "Pack my box with five dozen liquor jugs. ",
        "How vexingly quick daft zebras jump! ",
        "Sphinx of black quartz, judge my vow.",
    );

    let metrics = Metrics::new(16.0, 20.0);
    let mut buffer = Buffer::new(&mut font_system, metrics);
    buffer.set_text(&mut font_system, text, &Attrs::new(), Shaping::Advanced, None);
    buffer.shape_until_scroll(&mut font_system, false);

    // Prepare (populates atlas + instances)
    {
        let text_area = TextArea {
            buffer: &buffer,
            left: 10.0,
            top: 10.0,
            scale: 1.0,
            bounds: TextBounds { left: 0, top: 0, right: WIDTH as i32, bottom: HEIGHT as i32 },
            default_color: cosmic_text::Color::rgb(255, 255, 255),
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer.prepare(
            &device, &queue, &mut encoder, &mut font_system,
            &mut atlas, &viewport, [text_area], &mut swash_cache,
        ).expect("prepare failed");
    }

    eprintln!("glyphs={}", atlas.glyph_count());
    eprintln!("warmup_frames={WARMUP_FRAMES}");
    eprintln!("bench_frames={BENCH_FRAMES}");
    eprintln!("resolution={WIDTH}x{HEIGHT}");

    // Warmup
    for _ in 0..WARMUP_FRAMES {
        render_frame(&device, &queue, &renderer, &atlas, &viewport, &target_view, None, None, None);
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    // Benchmark
    let mut gpu_times_us = Vec::with_capacity(BENCH_FRAMES as usize);

    for _ in 0..BENCH_FRAMES {
        render_frame(
            &device, &queue, &renderer, &atlas, &viewport, &target_view,
            Some(&query_set), Some(&resolve_buf), Some(&readback_buf),
        );

        // Wait for GPU to finish and read back timestamps
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        let us = read_timestamp_us(&device, &readback_buf, timestamp_period);
        if let Some(us) = us {
            gpu_times_us.push(us);
        }
    }

    if gpu_times_us.is_empty() {
        eprintln!("No valid GPU timestamps collected");
        return;
    }

    gpu_times_us.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let len = gpu_times_us.len();
    let min = gpu_times_us[0];
    let max = gpu_times_us[len - 1];
    let median = gpu_times_us[len / 2];
    let p95 = gpu_times_us[((len as f64 * 0.95) as usize).min(len - 1)];
    let avg: f64 = gpu_times_us.iter().sum::<f64>() / len as f64;

    eprintln!();
    eprintln!("=== GPU render pass timing ({len} frames) ===");
    eprintln!("min={min:.1}µs  median={median:.1}µs  avg={avg:.1}µs  p95={p95:.1}µs  max={max:.1}µs");

    // KV output
    eprintln!("gpu_min_us={min:.0}");
    eprintln!("gpu_median_us={median:.0}");
    eprintln!("gpu_avg_us={avg:.0}");
    eprintln!("gpu_p95_us={p95:.0}");
    eprintln!("gpu_max_us={max:.0}");
}

fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &TextRenderer,
    atlas: &TextAtlas,
    viewport: &Viewport,
    target_view: &wgpu::TextureView,
    query_set: Option<&wgpu::QuerySet>,
    resolve_buf: Option<&wgpu::Buffer>,
    readback_buf: Option<&wgpu::Buffer>,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bench encoder"),
    });

    let timestamp_writes = query_set.map(|qs| wgpu::RenderPassTimestampWrites {
        query_set: qs,
        beginning_of_pass_write_index: Some(0),
        end_of_pass_write_index: Some(1),
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bench pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        renderer.render(atlas, viewport, &mut pass).expect("render failed");
    }

    // Resolve timestamps → resolve buffer → readback buffer
    if let (Some(qs), Some(rb), Some(rbb)) = (query_set, resolve_buf, readback_buf) {
        encoder.resolve_query_set(qs, 0..2, rb, 0);
        encoder.copy_buffer_to_buffer(rb, 0, rbb, 0, 16);
    }

    queue.submit(std::iter::once(encoder.finish()));
}

fn read_timestamp_us(
    device: &wgpu::Device,
    readback_buf: &wgpu::Buffer,
    timestamp_period: f32,
) -> Option<f64> {
    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    rx.recv().unwrap().ok()?;

    let data = slice.get_mapped_range();
    let timestamps: &[u64] = bytemuck::cast_slice(&data);
    let begin = timestamps[0];
    let end = timestamps[1];
    drop(data);
    readback_buf.unmap();

    if end <= begin {
        return None;
    }

    let ticks = (end - begin) as f64;
    let ns = ticks * timestamp_period as f64;
    Some(ns / 1000.0) // convert to microseconds
}
