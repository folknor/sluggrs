#![allow(clippy::unwrap_used)]
//! Visual snapshot target for brokkr integration.
//!
//! Contract (brokkr src/sluggrs/cmd.rs):
//! - invoked with cwd = project root, release profile, default features
//! - argv: --id <id> --output <abs> --width <n> --height <n>,
//!   then --font <abs> repeated, then --optional-font <abs> repeated
//! - writes a PNG at --output
//! - prints exactly one stdout line starting with `{`:
//!   {"adapter":"...","backend":"..."}
//! - exits 0 on success; nonzero records an ERROR row with the first
//!   stderr line surfaced
//!
//! Fonts come only from argv: the FontSystem is built from an empty fontdb
//! plus the given files, never from host font discovery, so snapshots are
//! reproducible across machines.

use std::fs;
use std::io::BufWriter;

use cosmic_text::{Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, Weight};
use sluggrs::{
    Cache, ColorMode, Resolution, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
    Viewport,
};

const MARGIN: f32 = 24.0;
const GAP: f32 = 12.0;

struct Args {
    id: String,
    output: String,
    width: u32,
    height: u32,
    fonts: Vec<String>,
    optional_fonts: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut id = None;
    let mut output = None;
    let mut width = None;
    let mut height = None;
    let mut fonts = Vec::new();
    let mut optional_fonts = Vec::new();

    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || {
            argv.next()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--id" => id = Some(value()?),
            "--output" => output = Some(value()?),
            "--width" => width = Some(value()?.parse::<u32>().map_err(|e| e.to_string())?),
            "--height" => height = Some(value()?.parse::<u32>().map_err(|e| e.to_string())?),
            "--font" => fonts.push(value()?),
            "--optional-font" => optional_fonts.push(value()?),
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        id: id.ok_or("missing --id")?,
        output: output.ok_or("missing --output")?,
        width: width.ok_or("missing --width")?,
        height: height.ok_or("missing --height")?,
        fonts,
        optional_fonts,
    })
}

// -- Scene definitions ------------------------------------------------------

struct Block {
    family: Family<'static>,
    weight: Weight,
    size: f32,
    color: Color,
    text: &'static str,
}

impl Block {
    fn new(family: Family<'static>, size: f32, text: &'static str) -> Self {
        Self {
            family,
            weight: Weight::NORMAL,
            size,
            color: Color::rgb(255, 255, 255),
            text,
        }
    }

    fn weight(mut self, weight: Weight) -> Self {
        self.weight = weight;
        self
    }

    fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

const PANGRAM: &str = "Sphinx of black quartz, judge my vow. 0123456789";

fn scene(id: &str) -> Option<Vec<Block>> {
    let inter = Family::Name("Inter Variable");
    let roboto = Family::Name("Roboto");
    match id {
        // Size sweep across the shader's regimes: MSAA blend below 16ppem,
        // plain single-sample midrange, darkening cutoff at 48ppem, large.
        "latin-sizes" => Some(vec![
            Block::new(inter, 8.0, PANGRAM),
            Block::new(roboto, 10.0, PANGRAM),
            Block::new(inter, 12.0, PANGRAM),
            Block::new(roboto, 14.0, PANGRAM),
            Block::new(inter, 18.0, PANGRAM),
            Block::new(roboto, 24.0, PANGRAM).weight(Weight::BOLD),
            Block::new(inter, 36.0, PANGRAM),
            Block::new(inter, 48.0, "Judge my vow, quick sphinx! 48px"),
            Block::new(roboto, 72.0, "Vexed zebras jump 72px"),
        ]),
        // Thin stems at small sizes, at several brightness levels: stem
        // darkening's gamma exponent depends on foreground brightness.
        "thin-stems" => Some(vec![
            Block::new(roboto, 10.0, PANGRAM).weight(Weight::THIN),
            Block::new(roboto, 12.0, PANGRAM).weight(Weight::THIN),
            Block::new(roboto, 14.0, PANGRAM).weight(Weight::THIN),
            Block::new(roboto, 20.0, PANGRAM).weight(Weight::THIN),
            Block::new(roboto, 12.0, PANGRAM)
                .weight(Weight::THIN)
                .color(Color::rgb(128, 128, 128)),
            Block::new(roboto, 12.0, PANGRAM)
                .weight(Weight::THIN)
                .color(Color::rgb(220, 140, 40)),
            Block::new(roboto, 12.0, PANGRAM).color(Color::rgb(128, 128, 128)),
            Block::new(roboto, 12.0, PANGRAM).color(Color::rgb(220, 140, 40)),
        ]),
        // COLRv0 (Twemoji) and COLRv1 (Noto) color emoji at three sizes.
        "emoji-colr" => Some(vec![
            Block::new(inter, 14.0, "COLRv0 Twemoji at 16 / 32 / 64:"),
            Block::new(Family::Name("Twemoji COLRv0"), 16.0, "😀🥶🎉🍕🚀❤️🔥👍"),
            Block::new(Family::Name("Twemoji COLRv0"), 32.0, "😀🥶🎉🍕🚀❤️🔥👍"),
            Block::new(Family::Name("Twemoji COLRv0"), 64.0, "😀🥶🎉🍕🚀❤️🔥👍"),
            Block::new(inter, 14.0, "COLRv1 Noto at 16 / 32 / 64:"),
            Block::new(Family::Name("Noto Color Emoji"), 16.0, "😀🥶🎉🍕🚀❤️🔥👍"),
            Block::new(Family::Name("Noto Color Emoji"), 32.0, "😀🥶🎉🍕🚀❤️🔥👍"),
            Block::new(Family::Name("Noto Color Emoji"), 64.0, "😀🥶🎉🍕🚀❤️🔥👍"),
        ]),
        // Unusual outlines (runes) and a ligature-heavy monospace face.
        "rune-mono" => Some(vec![
            Block::new(
                Family::Name("EBH Runes"),
                24.0,
                "ᚠᚢᚦᚨᚱᚲᚷᚹ ᚺᚾᛁᛃᛇᛈᛉᛊ ᛏᛒᛖᛗᛚᛜᛞᛟ",
            ),
            Block::new(Family::Name("EBH Runes"), 48.0, "ᚠᚢᚦᚨᚱᚲ ᛏᛒᛖᛗ"),
            Block::new(
                Family::Name("CaskaydiaCove Nerd Font"),
                12.0,
                "fn solve(a: f32) -> f32 { a != 0.0 && a >= 1.0 || a <= -1.0 } // => <= != ->",
            ),
            Block::new(
                Family::Name("CaskaydiaCove Nerd Font"),
                16.0,
                "fn solve(a: f32) -> f32 { a != 0.0 && a >= 1.0 || a <= -1.0 } // => <= != ->",
            ),
            Block::new(
                Family::Name("CaskaydiaCove Nerd Font"),
                24.0,
                "let band = curves[i] >> 2; // 0x2E74 & 0x0101",
            ),
        ]),
        _ => None,
    }
}

// -- Main -------------------------------------------------------------------

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("argument error: {e}");
            std::process::exit(1);
        }
    };

    let Some(blocks) = scene(&args.id) else {
        eprintln!("unknown snapshot id: {}", args.id);
        std::process::exit(1);
    };

    // Font system from exactly the argv-provided files.
    let mut db = cosmic_text::fontdb::Database::new();
    for path in &args.fonts {
        let Ok(data) = fs::read(path) else {
            eprintln!("required font not readable: {path}");
            std::process::exit(1);
        };
        db.load_font_data(data);
    }
    for path in &args.optional_fonts {
        match fs::read(path) {
            Ok(data) => db.load_font_data(data),
            Err(_) => eprintln!("optional font skipped: {path}"),
        }
    }
    let mut font_system = FontSystem::new_with_locale_and_db("en-US".to_string(), db);

    // Headless device, offscreen sRGB target with readback.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapter found");
    let info = adapter.get_info();

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sluggrs snapshot"),
        ..Default::default()
    }))
    .expect("Failed to create device");

    let format = wgpu::TextureFormat::Bgra8UnormSrgb;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("snapshot target"),
        size: wgpu::Extent3d {
            width: args.width,
            height: args.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let cache = Cache::new(&device);
    let mut atlas =
        TextAtlas::with_color_mode(&device, &queue, &cache, format, ColorMode::Accurate);
    let mut renderer =
        TextRenderer::new(&mut atlas, &device, wgpu::MultisampleState::default(), None);
    let mut viewport = Viewport::new(&device, &cache);
    viewport.update(
        &queue,
        Resolution {
            width: args.width,
            height: args.height,
        },
    );

    // Lay the blocks out top to bottom, wrapping at the target width.
    let usable_width = args.width as f32 - MARGIN * 2.0;
    let mut buffers: Vec<(Buffer, Color, f32)> = Vec::new();
    let mut cursor_y = MARGIN;
    for block in &blocks {
        let line_height = (block.size * 1.3).ceil();
        let metrics = Metrics::new(block.size, line_height);
        let mut buffer = Buffer::new(&mut font_system, metrics);
        buffer.set_size(Some(usable_width), None);
        let attrs = Attrs::new().family(block.family).weight(block.weight);
        buffer.set_text(block.text, &attrs, Shaping::Advanced, None);
        buffer.shape_until_scroll(&mut font_system, false);

        let mut block_height = line_height;
        for run in buffer.layout_runs() {
            block_height = block_height.max(run.line_top + line_height);
        }

        buffers.push((buffer, block.color, cursor_y));
        cursor_y += block_height + GAP;
    }

    let bounds = TextBounds {
        left: 0,
        top: 0,
        right: args.width as i32,
        bottom: args.height as i32,
    };
    let areas: Vec<TextArea> = buffers
        .iter()
        .map(|(buffer, color, top)| TextArea {
            buffer,
            left: MARGIN,
            top: *top,
            scale: 1.0,
            bounds,
            default_color: *color,
            border: None,
        })
        .collect();

    let mut swash_cache = SwashCache::new();
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    renderer
        .prepare(
            &device,
            &queue,
            &encoder,
            &mut font_system,
            &mut atlas,
            &viewport,
            areas,
            &mut swash_cache,
        )
        .expect("prepare failed");

    // Render and copy out.
    let unpadded_bytes_per_row = args.width * 4;
    let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("snapshot readback"),
        size: u64::from(padded_bytes_per_row) * u64::from(args.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("snapshot pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });
        renderer
            .render(&atlas, &viewport, &mut pass)
            .expect("render failed");
    }
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(args.height),
            },
        },
        wgpu::Extent3d {
            width: args.width,
            height: args.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv().unwrap().expect("readback map failed");

    // BGRA (padded rows) to RGBA. Alpha forced opaque: the comparison is
    // about color, and leftover blend alpha varies across drivers.
    let mapped = readback.slice(..).get_mapped_range();
    let mut rgba = Vec::with_capacity((args.width * args.height * 4) as usize);
    for row in 0..args.height {
        let start = (row * padded_bytes_per_row) as usize;
        let row_data = &mapped[start..start + unpadded_bytes_per_row as usize];
        let (pixels, _) = row_data.as_chunks::<4>();
        for px in pixels {
            rgba.extend_from_slice(&[px[2], px[1], px[0], 255]);
        }
    }
    drop(mapped);
    readback.unmap();

    let file = fs::File::create(&args.output).expect("cannot create output file");
    let mut encoder = png::Encoder::new(BufWriter::new(file), args.width, args.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header failed");
    writer.write_image_data(&rgba).expect("png write failed");
    writer.finish().expect("png finish failed");

    // Exactly one stdout line, starting with `{`.
    println!(
        "{{\"adapter\":\"{}\",\"backend\":\"{:?}\"}}",
        info.name, info.backend
    );
}
