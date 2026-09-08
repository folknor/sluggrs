//! Shared-buffer retained-cache benchmark for brokkr integration.

use std::time::Instant;

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping};
use sluggrs::{
    Cache, Color, ColorMode, Resolution, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
    Viewport,
};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const OCCURRENCES: usize = 32;
const WARM_FRAMES: u32 = 50;

// hotpath 0.17+: the consumer must declare the counting allocator itself,
// or --alloc builds silently report 0 bytes (see hotpath.rs).
#[cfg(feature = "hotpath-alloc")]
#[global_allocator]
static ALLOC: hotpath::CountingAllocator = hotpath::CountingAllocator::new();

fn main() {
    let _guard = hotpath::HotpathGuardBuilder::new("sluggrs::shared_buffer_bench")
        .percentiles(&[50.0, 95.0, 99.0])
        .functions_limit(0)
        .build();

    let (device, queue) = create_device();
    let cache = Cache::new(&device);
    let mut atlas = TextAtlas::with_color_mode(
        &device,
        &queue,
        &cache,
        wgpu::TextureFormat::Bgra8UnormSrgb,
        ColorMode::Accurate,
    );
    let mut renderer =
        TextRenderer::new(&mut atlas, &device, wgpu::MultisampleState::default(), None);
    let mut viewport = Viewport::new(&device, &cache);
    viewport.update(
        &queue,
        Resolution {
            width: WIDTH,
            height: HEIGHT,
        },
    );
    let mut font_system = FontSystem::new();
    let mut swash_cache = SwashCache::new();

    let mut buffer = Buffer::new(&mut font_system, Metrics::new(24.0, 30.0));
    buffer.set_text(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        &Attrs::new(),
        Shaping::Advanced,
        None,
    );
    buffer.shape_until_scroll(&mut font_system, false);

    let bounds = TextBounds {
        left: 0,
        top: 0,
        right: WIDTH as i32,
        bottom: HEIGHT as i32,
    };
    let areas = make_areas(&buffer, bounds);
    let measured_start = Instant::now();

    let cold_start = Instant::now();
    prepare(
        &mut renderer,
        &device,
        &queue,
        &mut font_system,
        &mut atlas,
        &viewport,
        &areas,
        &mut swash_cache,
    );
    let cold_prepare_us = cold_start.elapsed().as_micros();

    buffer.set_redraw(false);
    let areas = make_areas(&buffer, bounds);
    let mut total_direct = 0usize;
    let warm_start = Instant::now();
    for _ in 0..WARM_FRAMES {
        prepare(
            &mut renderer,
            &device,
            &queue,
            &mut font_system,
            &mut atlas,
            &viewport,
            &areas,
            &mut swash_cache,
        );
        total_direct += renderer.last_prepare_stats().direct_hits;
    }
    let warm_prepare_avg_us = warm_start.elapsed().as_micros() / u128::from(WARM_FRAMES);
    // Validate outside the timed loop: every warm frame must be all-direct.
    assert_eq!(total_direct, OCCURRENCES * WARM_FRAMES as usize);
    let stats = renderer.last_prepare_stats();

    eprintln!(
        "elapsed_ms={:.3}",
        measured_start.elapsed().as_secs_f64() * 1000.0
    );
    eprintln!("cold_prepare_us={cold_prepare_us}");
    eprintln!("warm_prepare_avg_us={warm_prepare_avg_us}");
    eprintln!("direct_hits={}", stats.direct_hits);
    eprintln!("reculls={}", stats.reculls);
    eprintln!("misses={}", stats.misses);
}

fn make_areas(buffer: &Buffer, bounds: TextBounds) -> [TextArea<'_>; OCCURRENCES] {
    std::array::from_fn(|index| TextArea {
        buffer,
        left: 20.0 + (index % 8) as f32 * 230.0,
        top: 20.0 + (index / 8) as f32 * 250.0,
        scale: 1.0,
        bounds,
        default_color: Color::rgb(255, 255, 255),
        border: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    renderer: &mut TextRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    font_system: &mut FontSystem,
    atlas: &mut TextAtlas,
    viewport: &Viewport,
    areas: &[TextArea<'_>],
    swash_cache: &mut SwashCache,
) {
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    renderer
        .prepare(
            device,
            queue,
            &encoder,
            font_system,
            atlas,
            viewport,
            areas.iter().copied(),
            swash_cache,
        )
        .expect("prepare should succeed");
}

fn create_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapter found");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("Failed to create device")
}
