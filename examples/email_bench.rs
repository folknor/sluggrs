#![allow(clippy::unwrap_used)]
//! Email-client-scale benchmark for brokkr integration.
//!
//! Simulates an email client viewport: multiple messages with mixed fonts,
//! sizes (8-20px), accented text, emoji, code snippets, and quoted replies.
//! Exercises cold/warm/scroll prepare paths and GPU rendering at realistic
//! glyph counts (~15k instances, ~300 distinct glyphs).
//!
//! Run via brokkr:  brokkr sluggrs hotpath  (once wired)
//! Run standalone:  cargo run --release --example email-bench --features hotpath

use std::time::Instant;

use cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping, Weight};
use sluggrs::{
    Cache, ColorMode, Resolution, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
    Viewport,
};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

// -- Email content ----------------------------------------------------------

struct EmailMessage {
    header: &'static str,
    metadata: &'static str,
    body: &'static str,
    quote: &'static str,
    code: &'static str,
}

const EMAILS: &[EmailMessage] = &[
    EmailMessage {
        header: "Re: Performance review of the Slug GPU text renderer",
        metadata: "From: María García <maria@example.com>  To: team@example.com\n\
                   Date: 2026-04-02 14:32 UTC  CC: François Dubois, Jürgen Müller",
        body: "Hi everyone,\n\n\
               I've finished the initial performance review of the Slug GPU text rendering \
               pipeline. The cold prepare path is dominated by per-glyph allocation in \
               build_bands() — 14 fresh Vec allocations per cache miss. At 92 distinct glyphs, \
               that's 1,288 malloc/free cycles on the critical path.\n\n\
               The warm path looks reasonable at ~50-100µs, but our benchmark includes \
               cosmic_text shaping overhead which inflates the number. We need a shaping-free \
               benchmark before we can properly attribute warm-path cost.\n\n\
               Key findings:\n\
               • BandScratch reuse would save ~100µs cold (10-15%)\n\
               • prepare_outline() clone is pure waste — 5-10% of cold prepare\n\
               • Batching queue.write_buffer from 92 calls to 1 saves ~20-90µs\n\
               • GPU time is already good: 11µs headless, 71µs windowed\n\n\
               Let me know if you want to discuss priorities. I think items 1-3 are the \
               highest ROI and we should tackle them before the architecture changes.\n\n\
               Regards,\nMaría",
        quote: "> Jürgen wrote:\n\
                > The band count heuristic needs revisiting. Currently using min(curves, 16)\n\
                > which matches harfbuzz, but adaptive scaling based on ppem could reduce\n\
                > shader work significantly for complex glyphs at small sizes.\n\
                > \n\
                > Also, the 5× MSAA path below 16ppem is the dominant GPU cost for\n\
                > production email UIs with 12-14px body text.",
        code: "fn build_bands(outline: &GlyphOutline, band_count: u32) -> BandResult {\n    \
               let mut scratch = BandScratch::default();\n    \
               scratch.clear_and_resize(outline.curves.len(), band_count);\n    \
               // Phase 1: classify curves into bands\n    \
               for (i, curve) in outline.curves.iter().enumerate() {\n        \
               let (min_y, max_y) = curve.y_range();\n        \
               scratch.assign_to_bands(i, min_y, max_y);\n    \
               }\n\
               }",
    },
    EmailMessage {
        header: "Shipping estimate for Q3 iced integration",
        metadata: "From: Søren Andersen <soren@example.com>  To: team@example.com\n\
                   Date: 2026-04-02 11:05 UTC",
        body: "Team,\n\n\
               Quick update on the iced integration timeline. The sluggrs branch on folknor/iced \
               has text.rs swapped from cryoglyph and passes basic rendering tests. Remaining \
               work before we can submit upstream:\n\n\
               1. Emoji/non-vector glyph fallback — currently silently dropped, needs explicit \
                  classification API for two-pass routing (sluggrs → cryoglyph)\n\
               2. trim() invalidation bug — prepare→trim→render sequence can draw stale data\n\
               3. ColorMode actually needs to work — sRGB vs linear framebuffer handling\n\
               4. GlyphInstance stride alignment — 96 bytes with 8 bytes padding, verify on \
                  all backends (Metal, Vulkan, DX12, WebGPU)\n\n\
               The font system integration is clean — cosmic_text 0.18 gives us everything we \
               need. FontRef caching per (font_id, face_index) would help cold starts with \
               large font collections.\n\n\
               For reference, the Noto font family alone has 2,000+ glyphs per weight × \
               multiple weights × CJK variants. An email with Japanese + Latin mixed content \
               could easily hit 500+ distinct glyphs on first render.\n\n\
               Estimated completion: 3-4 weeks for items 1-3, then a week of integration \
               testing before PR.\n\n\
               — Søren",
        quote: "> María wrote:\n\
                > The prepare_outline() clone removal is trivial — GpuOutline is already a\n\
                > type alias. Just pass &GlyphOutline through to upload_glyph and only\n\
                > clone for the ~1% of glyphs that need italic shear.",
        code: "// GlyphKey: 12-byte cache key, currently using SipHash\n\
               pub struct GlyphKey {\n    \
               pub font_id: fontdb::ID,   // u32\n    \
               pub glyph_id: u16,\n    \
               pub font_weight: u16,\n    \
               pub cache_key_flags: CacheKeyFlags, // u32\n\
               }",
    },
    EmailMessage {
        header: "📧 Meeting notes: GPU shader optimization discussion",
        metadata: "From: Aïsha Traoré <aisha@example.com>  To: team@example.com\n\
                   Date: 2026-04-01 16:20 UTC  CC: Kōji Tanaka",
        body: "Notes from today's shader review session 📎\n\n\
               Attendees: Aïsha, Kōji, François, María\n\n\
               ✅ Agreed: pack i16 pairs into i32 to halve storage buffer bandwidth\n\
               ✅ Agreed: precompute a,b from unshifted coords (matches harfbuzz pattern)\n\
               ❌ Rejected: compute shader rewrite — incompatible with render-pass integration\n\
               ⚠️ Deferred: analytical AA to replace 5× MSAA — needs more research\n\n\
               Kōji presented RenderDoc captures showing the fragment shader spends 60% of \
               time in curve evaluation loops. The storage buffer fetch pattern is efficient \
               on RTX 3080 (L2 hit rate >90%) but may be bandwidth-bound on integrated GPUs.\n\n\
               Action items:\n\
               🔧 François: prototype i16 packing in simple_shader.wgsl\n\
               💡 Kōji: capture Nsight traces for branch divergence analysis\n\
               📊 María: run email-scale benchmark on Intel UHD 630\n\
               🏗️ Søren: design blob header format for em_rect/band_transform\n\n\
               Next meeting: Thursday 15:00 UTC\n\n\
               — Aïsha",
        quote: "> Kōji wrote:\n\
                > The CalcRootCode in our shader uses select() where the Slug reference\n\
                > uses asuint(y) >> 31. On NVIDIA this reduces to a single LOP3 instruction.\n\
                > We should verify naga's SPIR-V output — it might already optimize this.",
        code: "// Current solver — both paths computed unconditionally\n\
               let ra = 1.0 / a.y;\n\
               let rb = 0.5 / b.y;\n\
               let d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));\n\
               var t1 = (b.y - d) * ra;\n\
               var t2 = (b.y + d) * ra;\n\
               if a.y == 0.0 {\n    \
               let lin = p12.y * rb;\n    \
               t1 = lin; t2 = lin;\n\
               }",
    },
    EmailMessage {
        header: "Re: Font rendering comparison: Noto Sans vs Inter vs Roboto",
        metadata: "From: François Dubois <francois@example.com>  To: team@example.com\n\
                   Date: 2026-04-01 09:47 UTC",
        body: "Bonjour,\n\n\
               I ran the visual comparison across the three font families at various sizes. \
               Results below (all measured on RTX 3080 at 1920×1080, Bgra8UnormSrgb format):\n\n\
               Noto Sans Regular:\n\
               • 10px: stem darkening visible, MSAA 4× active, γ=1.8\n\
               • 14px: clean rendering, no MSAA, dilation 0.5px\n\
               • 20px: crisp edges, could disable dilation entirely\n\
               • 48px+: stem darkening disabled (no-op above threshold)\n\n\
               Inter Variable (wght 100-900):\n\
               • Weight 400 at 14px: 34 curves/glyph average, 8 bands\n\
               • Weight 700 at 14px: 38 curves/glyph average, 8 bands\n\
               • Weight 100 at 10px: thin stems need careful γ tuning\n\n\
               Roboto Regular:\n\
               • CFF outlines — exercises cu2qu cubic→quadratic conversion\n\
               • Higher curve count after conversion (~15% more than TTF)\n\
               • f64 precision in cu2qu is overkill at 0.5 font-unit tolerance\n\n\
               All three render correctly with our half-pixel dilation model. The Jacobian-\
               based dilation from harfbuzz would only matter for rotated/scaled text, \
               which email clients don't typically need.\n\n\
               Cordialement,\nFrançois",
        quote: "> Aïsha wrote:\n\
                > For the Intel integrated GPU test, use WGPU_BACKEND=vulkan and capture\n\
                > with RenderDoc. The timestamp queries should work on Intel UHD 630+\n\
                > but verify TIMESTAMP_QUERY_INSIDE_PASSES support first.",
        code: "// Per-span metrics for mixed-size email rendering\n\
               let header_attrs = Attrs::new()\n    \
               .family(Family::SansSerif)\n    \
               .weight(Weight::BOLD)\n    \
               .metrics(Metrics::new(20.0, 26.0));\n\
               let body_attrs = Attrs::new()\n    \
               .family(Family::SansSerif)\n    \
               .metrics(Metrics::new(14.0, 20.0));",
    },
    EmailMessage {
        header: "Re: Texture growth stall on CJK-heavy content",
        metadata: "From: Kōji Tanaka <koji@example.com>  To: team@example.com\n\
                   Date: 2026-03-31 22:15 UTC",
        body: "こんにちは team,\n\n\
               I reproduced the texture growth stall with a mixed Japanese-Latin email. When \
               loading 200+ new CJK glyphs across growth boundaries, grow_buffer() copies \
               the entire atlas contents multiple times — O(n) full re-uploads.\n\n\
               Test case: email with Japanese subject + body + quoted English reply:\n\
               • 日本語のテキストレンダリングテスト (Japanese rendering test)\n\
               • フォントシステムの統合 (Font system integration)\n\
               • グリフキャッシュの最適化 (Glyph cache optimization)\n\n\
               Each CJK glyph has 40-80 curves (vs 20-40 for Latin), so band building \
               is proportionally more expensive. The BandScratch optimization becomes \
               even more important for CJK-heavy workloads.\n\n\
               Numbers from my test (Core i9-13900K + RTX 4090):\n\
               • 200 CJK glyphs cold: 4.2ms prepare (vs 753µs for 92 Latin)\n\
               • Buffer grew 3 times during prepare: 8192 → 16384 → 32768 → 65536\n\
               • Each growth copied all accumulated data\n\n\
               Proposed fix: predict final buffer size from total curve count before \
               uploading any glyphs. We know the curve count after extract_outline(), \
               so we can estimate blob size and pre-allocate.\n\n\
               よろしく,\nKōji",
        quote: "> Søren wrote:\n\
                > The atlas initial capacity of 8192 elements is fine for Latin-only\n\
                > workloads but way too small for CJK. Should we add a capacity hint\n\
                > parameter, or just start larger by default?",
        code: "// Growth pattern: doubles until large enough\n\
               fn grow_buffer(&mut self, needed: u32) {\n    \
               let mut new_cap = self.buffer_capacity;\n    \
               while new_cap < self.buffer_cursor + needed {\n        \
               new_cap = (new_cap * 2).min(max_elements);\n    \
               }\n    \
               // Full re-upload of all prior data\n    \
               queue.write_buffer(&new_buf, 0, &self.buffer_data);\n\
               }",
    },
];

// -- Main --------------------------------------------------------------------

fn main() {
    let _guard = hotpath::HotpathGuardBuilder::new("sluggrs::email_bench")
        .percentiles(&[50, 95, 99])
        .with_functions_limit(0)
        .build();

    let (device, queue) = create_device();
    let mut harness = RenderHarness::new(&device, &queue);

    // Build all email buffers (shape once, reuse across phases)
    let buffers = build_email_buffers(&mut harness.font_system);
    let text_areas = layout_text_areas(&buffers);

    // -- Cold prepare: all caches empty --
    let cold_start = Instant::now();
    harness
        .prepare_areas(&text_areas)
        .expect("Cold prepare failed");
    let cold_us = cold_start.elapsed().as_micros();

    let distinct_glyphs = harness.atlas.glyph_count();
    let total_instances: usize = text_areas
        .iter()
        .map(|a| {
            a.buffer
                .layout_runs()
                .flat_map(|run| run.glyphs.iter())
                .count()
        })
        .sum();

    // -- Warm prepare: same pre-shaped buffers, no re-shaping --
    let warm_iterations = 50u32;
    let warm_start = Instant::now();
    for _ in 0..warm_iterations {
        harness
            .prepare_areas(&text_areas)
            .expect("Warm prepare failed");
    }
    let warm_avg_us = warm_start.elapsed().as_micros() / warm_iterations as u128;

    // -- Scroll simulation: shift offsets, swap one buffer periodically --
    let scroll_iterations = 20u32;
    let extra_buffer = build_scroll_replacement_buffer(&mut harness.font_system);
    let scroll_start = Instant::now();
    for i in 0..scroll_iterations {
        let offset = (i as f32) * 3.0;
        let scroll_areas: Vec<TextArea> = text_areas
            .iter()
            .enumerate()
            .map(|(idx, area)| {
                // Every 5th frame, swap the last message buffer
                let buf = if i % 5 == 0 && idx == buffers.len() - 1 {
                    &extra_buffer
                } else {
                    area.buffer
                };
                TextArea {
                    buffer: buf,
                    left: area.left,
                    top: area.top - offset,
                    scale: area.scale,
                    bounds: area.bounds,
                    default_color: area.default_color,
                }
            })
            .collect();
        harness
            .prepare_areas(&scroll_areas)
            .expect("Scroll prepare failed");
    }
    let scroll_avg_us = scroll_start.elapsed().as_micros() / scroll_iterations as u128;

    // -- GPU render --
    for _ in 0..5 {
        harness.render_gpu();
    }
    let mut gpu_times: Vec<f64> = Vec::new();
    for _ in 0..20 {
        if let Some(ms) = harness.render_gpu() {
            gpu_times.push(ms);
        }
    }
    gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let final_glyphs = harness.atlas.glyph_count();
    let buffer_elements = harness.atlas.buffer_elements_used();

    // -- Emit KV pairs for brokkr --
    eprintln!("distinct_glyphs={distinct_glyphs}");
    eprintln!("final_glyphs={final_glyphs}");
    eprintln!("total_glyph_instances={total_instances}");
    eprintln!("text_areas={}", text_areas.len());
    eprintln!("buffer_elements={buffer_elements}");
    eprintln!("buffer_bytes={}", buffer_elements as u64 * 16);
    eprintln!("cold_prepare_us={cold_us}");
    eprintln!("warm_prepare_avg_us={warm_avg_us}");
    eprintln!("warm_iterations={warm_iterations}");
    eprintln!("scroll_prepare_avg_us={scroll_avg_us}");
    eprintln!("scroll_iterations={scroll_iterations}");

    if let Some(median) = gpu_times.get(gpu_times.len() / 2) {
        eprintln!("gpu_text_render_us={}", (*median * 1000.0) as u64);
    }
}

// -- Email buffer construction -----------------------------------------------

fn header_attrs() -> Attrs<'static> {
    Attrs::new()
        .family(Family::SansSerif)
        .weight(Weight::BOLD)
        .metrics(Metrics::new(20.0, 26.0))
}

fn meta_attrs() -> Attrs<'static> {
    Attrs::new()
        .family(Family::SansSerif)
        .color(cosmic_text::Color::rgb(140, 140, 140))
        .metrics(Metrics::new(10.0, 14.0))
}

fn body_attrs() -> Attrs<'static> {
    Attrs::new()
        .family(Family::SansSerif)
        .metrics(Metrics::new(14.0, 20.0))
}

fn quote_attrs() -> Attrs<'static> {
    Attrs::new()
        .family(Family::SansSerif)
        .color(cosmic_text::Color::rgb(100, 130, 180))
        .metrics(Metrics::new(12.0, 16.0))
}

fn code_attrs() -> Attrs<'static> {
    Attrs::new()
        .family(Family::Monospace)
        .color(cosmic_text::Color::rgb(180, 200, 160))
        .metrics(Metrics::new(13.0, 18.0))
}

fn build_email_buffers(font_system: &mut FontSystem) -> Vec<Buffer> {
    EMAILS
        .iter()
        .map(|email| {
            let mut buffer = Buffer::new(font_system, Metrics::new(14.0, 20.0));
            buffer.set_size(font_system, Some(WIDTH as f32 - 40.0), None);

            let spans: Vec<(&str, Attrs)> = vec![
                (email.header, header_attrs()),
                ("\n", body_attrs()),
                (email.metadata, meta_attrs()),
                ("\n\n", body_attrs()),
                (email.body, body_attrs()),
                ("\n\n", body_attrs()),
                (email.quote, quote_attrs()),
                ("\n\n", body_attrs()),
                (email.code, code_attrs()),
            ];

            buffer.set_rich_text(
                font_system,
                spans,
                &body_attrs(),
                Shaping::Advanced,
                None,
            );
            buffer.shape_until_scroll(font_system, false);
            buffer
        })
        .collect()
}

fn build_scroll_replacement_buffer(font_system: &mut FontSystem) -> Buffer {
    let mut buffer = Buffer::new(font_system, Metrics::new(14.0, 20.0));
    buffer.set_size(font_system, Some(WIDTH as f32 - 40.0), None);
    buffer.set_rich_text(
        font_system,
        [(
            "This is a replacement message that appears during scroll simulation. \
             It introduces a few new glyphs to exercise partial cache misses. \
             Characters like ø, å, æ, þ, ð help test Nordic coverage. \
             Numbers: 2026-04-03T14:32:00Z — ISO 8601 timestamps are common in email.\n\n\
             Ärger mit Übersetzungen ist häufig in mehrsprachigen E-Mail-Clients. \
             Die Lösung liegt in der korrekten Zeichenkodierung und Font-Auswahl.",
            body_attrs(),
        )],
        &body_attrs(),
        Shaping::Advanced,
        None,
    );
    buffer.shape_until_scroll(font_system, false);
    buffer
}

fn layout_text_areas(buffers: &[Buffer]) -> Vec<TextArea<'_>> {
    let mut top = 20.0f32;
    let gap = 30.0f32;

    buffers
        .iter()
        .map(|buffer| {
            let line_count = buffer.layout_runs().count();
            // Estimate height from line count × average line height
            let height = (line_count as f32) * 20.0;

            let area = TextArea {
                buffer,
                left: 20.0,
                top,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: WIDTH as i32,
                    bottom: HEIGHT as i32,
                },
                default_color: cosmic_text::Color::rgb(230, 230, 230),
            };
            top += height + gap;
            area
        })
        .collect()
}

// -- GPU infrastructure (mirrors hotpath.rs) ---------------------------------

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
        label: Some("sluggrs email-bench"),
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
                width: WIDTH,
                height: HEIGHT,
            },
        );

        let render_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen render target"),
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

    fn prepare_areas(&mut self, areas: &[TextArea]) -> Result<(), sluggrs::PrepareError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.renderer.prepare(
            &self.device,
            &self.queue,
            &mut encoder,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            areas.iter().cloned(),
            &mut self.swash_cache,
        )
    }

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
}
