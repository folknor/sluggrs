#![allow(clippy::unwrap_used)]

use std::time::Instant;

use cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping, SwashCache};
use sluggrs::{
    Cache, ColorMode, Resolution, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

fn main() {
    let _guard = hotpath::HotpathGuardBuilder::new("sluggrs::atlas_repopulate")
        .functions_limit(0)
        .build();
    let (device, queue) = device();
    let cache = Cache::new(&device);
    let format = wgpu::TextureFormat::Bgra8UnormSrgb;
    let mut atlas =
        TextAtlas::with_initial_buffer_capacity(&device, &cache, format, ColorMode::Accurate, 256);
    let mut renderer =
        TextRenderer::new(&mut atlas, &device, wgpu::MultisampleState::default(), None);
    let mut viewport = Viewport::new(&device, &cache);
    viewport.update(
        &queue,
        Resolution {
            width: 1920,
            height: 1080,
        },
    );
    // Bundled font only: host font discovery would make the distinct-glyph
    // count (and thus the compaction trigger) host-dependent.
    let mut db = cosmic_text::fontdb::Database::new();
    db.load_font_data(include_bytes!("fonts/InterVariable.ttf").to_vec());
    let mut font_system = FontSystem::new_with_locale_and_db("en-US".to_string(), db);
    let mut swash_cache = SwashCache::new();
    let mut set_a = buffer(&mut font_system, &distinct_text());
    let mut set_b = buffer(&mut font_system, "0123456789");
    let elapsed = Instant::now();
    let cold = Instant::now();
    prepare(
        &mut renderer,
        &mut atlas,
        &device,
        &queue,
        &viewport,
        &mut font_system,
        &mut swash_cache,
        &set_a,
    )
    .expect("prepare A");
    let cold_prepare_us = cold.elapsed().as_micros();
    set_a.set_redraw(false);
    atlas.trim();
    prepare(
        &mut renderer,
        &mut atlas,
        &device,
        &queue,
        &viewport,
        &mut font_system,
        &mut swash_cache,
        &set_b,
    )
    .expect("prepare B");
    set_b.set_redraw(false);
    let generation = atlas.generation();
    atlas.trim();
    assert!(atlas.generation() != generation, "small atlas must compact");
    let repopulate = Instant::now();
    prepare(
        &mut renderer,
        &mut atlas,
        &device,
        &queue,
        &viewport,
        &mut font_system,
        &mut swash_cache,
        &set_a,
    )
    .expect("repopulate A");
    let repopulate_us = repopulate.elapsed().as_micros();
    let stats = atlas.blob_cache_stats();
    assert!(stats.hits > 0, "A should restore from the blob cache");
    eprintln!("elapsed_ms={:.3}", elapsed.elapsed().as_secs_f64() * 1000.0);
    eprintln!("cold_prepare_us={cold_prepare_us}");
    eprintln!("repopulate_us={repopulate_us}");
    eprintln!("blob_cache_hits={}", stats.hits);
    eprintln!("repopulated_bytes={}", stats.repopulated_bytes);
    eprintln!("blob_cache_bytes={}", stats.bytes);
    eprintln!("blob_cache_evictions={}", stats.budget_evictions);
}

/// Hundreds of distinct codepoints from ranges InterVariable covers
/// (Latin-1/Ext-A, Greek, Cyrillic). Distinct-glyph count is what drives
/// both atlas growth and the compaction trigger (in_use < cached / 4);
/// repeating a fixed sample would add zero distinct glyphs.
fn distinct_text() -> String {
    let mut text = String::from("Latin sample ");
    let ranges = [(0xC0u32, 0x17F), (0x391, 0x3C9), (0x410, 0x44F)];
    let mut i = 0u32;
    for (start, end) in ranges {
        for cp in start..=end {
            if let Some(c) = char::from_u32(cp) {
                text.push(c);
                i += 1;
                if i % 16 == 15 {
                    text.push(' ');
                }
            }
        }
    }
    text
}

fn buffer(font_system: &mut FontSystem, text: &str) -> Buffer {
    let mut buffer = Buffer::new(font_system, Metrics::new(18.0, 24.0));
    buffer.set_size(Some(1900.0), None);
    buffer.set_text(
        text,
        &Attrs::new().family(Family::SansSerif),
        Shaping::Advanced,
        None,
    );
    buffer.shape_until_scroll(font_system, false);
    buffer
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    renderer: &mut TextRenderer,
    atlas: &mut TextAtlas,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    viewport: &Viewport,
    font_system: &mut FontSystem,
    swash_cache: &mut SwashCache,
    buffer: &Buffer,
) -> Result<(), sluggrs::PrepareError> {
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    renderer.prepare(
        device,
        queue,
        font_system,
        atlas,
        viewport,
        [TextArea {
            buffer,
            left: 0.0,
            top: 0.0,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: 0,
                right: 1920,
                bottom: 1080,
            },
            default_color: cosmic_text::Color::rgb(255, 255, 255),
        }]
    )
}

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("adapter");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).expect("device")
}
