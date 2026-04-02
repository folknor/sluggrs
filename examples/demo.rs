use sluggrs::band::{self, CurveLocation, build_bands};
use sluggrs::outline::{char_to_glyph_id, extract_outline};
use sluggrs::prepare::{self, GpuOutline};

const CURVE_TEXTURE_WIDTH: u32 = 4096;

use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop, window::Window,
};

// Embedded fonts (always available)
const INTER_VARIABLE: &[u8] = include_bytes!("fonts/InterVariable.ttf");
const ROBOTO_REGULAR: &[u8] = include_bytes!("fonts/Roboto-Regular.ttf");
const ROBOTO_THIN: &[u8] = include_bytes!("fonts/Roboto-Thin.ttf");
const ROBOTO_BOLD: &[u8] = include_bytes!("fonts/Roboto-Bold.ttf");
const CASKAYDIA: &[u8] = include_bytes!("fonts/CaskaydiaCoveNerdFont-Regular.ttf");
const RUNES: &[u8] = include_bytes!("fonts/EBH Runes.otf");

/// Per-instance vertex data for a glyph (matches GlyphInstance in shader).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GlyphInstance {
    screen_rect: [f32; 4],    // x, y, width, height
    em_rect: [f32; 4],        // min_x, min_y, max_x, max_y
    band_transform: [f32; 4], // scale_x, scale_y, offset_x, offset_y
    glyph_data: [u32; 4],     // glyph_loc.x, glyph_loc.y, band_max.x, band_max.y
    color: [f32; 4],          // RGBA
    depth: f32,               // z-depth for widget layering
    ppem: f32,                // pixels per em
    _pad: [f32; 2],           // alignment padding
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    screen_size: [f32; 2],
    scroll_offset: [f32; 2],
    flags: u32,     // bit 0: enable MSAA+stem darkening
    _pad: u32,
}

/// Prepared glyph data ready for GPU upload.
struct PreparedGlyph {
    gpu_outline: GpuOutline,
    band_data: band::BandData,
}

/// Quantize f32 em-space coordinate to i16 at 4 units/em.
fn q(v: f32) -> i16 {
    (v * 4.0).round() as i16
}

/// Build curve texture data from glyph outlines (int16 quantized).
fn build_curve_texture(glyphs: &[PreparedGlyph]) -> Vec<[i16; 4]> {
    let mut texels: Vec<[i16; 4]> = Vec::new();

    for glyph in glyphs {
        for curve in &glyph.gpu_outline.curves {
            texels.push([q(curve.p1[0]), q(curve.p1[1]), q(curve.p2[0]), q(curve.p2[1])]);
            texels.push([q(curve.p3[0]), q(curve.p3[1]), 0, 0]);
        }
    }

    if texels.is_empty() {
        texels.push([0; 4]);
    }

    texels
}

/// Build band texture data from prepared glyphs.
fn build_band_texture(glyphs: &[PreparedGlyph]) -> Vec<[i16; 4]> {
    let mut texels: Vec<[i16; 4]> = Vec::new();

    for glyph in glyphs {
        for chunk in glyph.band_data.entries.chunks(4) {
            let mut texel = [0i16; 4];
            for (i, &val) in chunk.iter().enumerate() {
                texel[i] = val;
            }
            texels.push(texel);
        }
    }

    if texels.is_empty() {
        texels.push([0i16; 4]);
    }

    texels
}

/// Prepare glyphs for a string of text with a given font.
#[allow(clippy::too_many_arguments)]
fn prepare_text(
    font_data: &[u8],
    text: &str,
    font_size: f32,
    start_x: f32,
    start_y: f32,
    color: [f32; 4],
    weight: Option<f32>,
    base_curve_offset: u32,
    base_band_offset: u32,
) -> (Vec<PreparedGlyph>, Vec<GlyphInstance>) {
    let font = match skrifa::FontRef::new(font_data) {
        Ok(f) => f,
        Err(_) => {
            // Try as font collection (index 0)
            match skrifa::FontRef::from_index(font_data, 0) {
                Ok(f) => f,
                Err(e) => {
                    log::error!("Failed to parse font: {e:?}");
                    return (Vec::new(), Vec::new());
                }
            }
        }
    };

    let units_per_em = {
        use skrifa::raw::TableProvider;
        font.head().expect("no head table").units_per_em() as f32
    };
    let scale = font_size / units_per_em;

    let mut prepared: Vec<PreparedGlyph> = Vec::new();
    let mut instances: Vec<GlyphInstance> = Vec::new();
    let mut cursor_x = start_x;
    let mut curve_offset: u32 = base_curve_offset;
    let mut band_offset: u32 = base_band_offset;

    let hmtx = {
        use skrifa::raw::TableProvider;
        font.hmtx().expect("no hmtx table")
    };

    for ch in text.chars() {
        let glyph_id = match char_to_glyph_id(font_data, 0, ch) {
            Some(id) => id,
            None => continue,
        };

        let advance = hmtx
            .h_metrics()
            .get(glyph_id as usize)
            .map(|m| m.advance.get() as f32)
            .unwrap_or_else(|| {
                hmtx.h_metrics()
                    .last()
                    .map(|m| m.advance.get() as f32)
                    .unwrap_or(units_per_em * 0.5)
            });

        let wght_tag = skrifa::Tag::new(b"wght");
        let location: Vec<skrifa::setting::VariationSetting> = weight
            .map(|w| vec![skrifa::setting::VariationSetting::new(wght_tag, w)])
            .unwrap_or_default();
        let outline = match extract_outline(font_data, 0, glyph_id, &location) {
            Some(o) => o,
            None => {
                cursor_x += advance * scale;
                continue;
            }
        };

        let gpu_outline = prepare::prepare_outline(&outline);
        let num_curves = gpu_outline.curves.len();

        let curve_locations: Vec<CurveLocation> = (0..num_curves)
            .map(|i| CurveLocation {
                offset: curve_offset + (i as u32) * 2,
            })
            .collect();

        let band_count = if num_curves < 10 {
            4
        } else if num_curves < 30 {
            8
        } else {
            12
        };
        let band_data = build_bands(
            &gpu_outline,
            &curve_locations,
            band_count,
            band_count,
            Vec::new(),
        );

        let [min_x, min_y, max_x, max_y] = gpu_outline.bounds;

        let screen_x = cursor_x + min_x * scale;
        let screen_y = start_y - max_y * scale;
        let screen_w = (max_x - min_x) * scale;
        let screen_h = (max_y - min_y) * scale;

        let glyph_band_texel_count = (band_data.entries.len() / 4) as u32;

        instances.push(GlyphInstance {
            screen_rect: [screen_x, screen_y, screen_w, screen_h],
            em_rect: [min_x, min_y, max_x, max_y],
            band_transform: band_data.band_transform,
            glyph_data: [
                band_offset % sluggrs::BAND_TEXTURE_WIDTH,
                band_offset / sluggrs::BAND_TEXTURE_WIDTH,
                (band_count - 1) as u32,
                (band_count - 1) as u32,
            ],
            color,
            depth: 0.0,
            ppem: font_size,
            _pad: [0.0; 2],
        });

        prepared.push(PreparedGlyph {
            gpu_outline,
            band_data,
        });

        curve_offset += (num_curves as u32) * 2;
        band_offset += glyph_band_texel_count;
        cursor_x += advance * scale;
    }

    (prepared, instances)
}

struct App {
    state: Option<RenderState>,
    window: Option<Arc<Window>>,
    warmup_frames: u32,
    enhance: bool,
}

struct RenderState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
    zoom: f32,
    scroll: [f32; 2],
    dragging: bool,
    last_mouse: [f32; 2],
    gpu_profiler: Option<wgpu_profiler::GpuProfiler>,
}

impl RenderState {
    fn update_params(&self, enhance: bool) {
        let params = Params {
            screen_size: [
                self.config.width as f32 / self.zoom,
                self.config.height as f32 / self.zoom,
            ],
            scroll_offset: self.scroll,
            flags: if enhance { 1 } else { 0 },
            _pad: 0,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }
}

impl App {
    fn new() -> Self {
        Self {
            state: None,
            window: None,
            warmup_frames: 5,
            enhance: true,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("sluggrs demo")
                        .with_inner_size(winit::dpi::LogicalSize::new(1200, 900)),
                )
                .expect("failed to create window"),
        );

        let state = pollster::block_on(init_render_state(Arc::clone(&window)));
        self.state = Some(state);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(state) = &mut self.state {
                    state.config.width = new_size.width.max(1);
                    state.config.height = new_size.height.max(1);
                    state.surface.configure(&state.device, &state.config);
                    state.update_params(self.enhance);

                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(state) = &mut self.state {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                    };
                    state.zoom = (state.zoom * (1.0 + scroll * 0.1)).clamp(0.1, 20.0);
                    state.update_params(self.enhance);

                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(key),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(state) = &mut self.state {
                    let step = 50.0 / state.zoom;
                    match key {
                        winit::keyboard::KeyCode::ArrowUp => state.scroll[1] += step,
                        winit::keyboard::KeyCode::ArrowDown => state.scroll[1] -= step,
                        winit::keyboard::KeyCode::ArrowLeft => state.scroll[0] += step,
                        winit::keyboard::KeyCode::ArrowRight => state.scroll[0] -= step,
                        winit::keyboard::KeyCode::Home => {
                            state.scroll = [0.0, 0.0];
                            state.zoom = 1.0;
                        }
                        winit::keyboard::KeyCode::KeyE => {
                            self.enhance = !self.enhance;
                            eprintln!(
                                "MSAA + stem darkening: {}",
                                if self.enhance { "ON" } else { "OFF" }
                            );
                        }
                        _ => return,
                    }
                    state.update_params(self.enhance);
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                if let Some(state) = &mut self.state {
                    state.dragging = btn_state == winit::event::ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(state) = &mut self.state {
                    let pos = [position.x as f32, position.y as f32];
                    if state.dragging {
                        let dx = pos[0] - state.last_mouse[0];
                        let dy = pos[1] - state.last_mouse[1];
                        state.scroll[0] += dx / state.zoom;
                        state.scroll[1] += dy / state.zoom;
                        state.update_params(self.enhance);
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                    }
                    state.last_mouse = pos;
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    render(state);
                    // Warmup frames to flush GPU profiler pipeline
                    if self.warmup_frames > 0 {
                        self.warmup_frames -= 1;
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Try to load an optional font from disk (for licensed fonts not in git).
fn try_load_font(path: &str) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

async fn init_render_state(window: Arc<Window>) -> RenderState {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let surface = instance
        .create_surface(Arc::clone(&window))
        .expect("failed to create surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("failed to find adapter");

    log::info!("Using adapter: {:?}", adapter.get_info().name);

    let has_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let has_pass_timestamps = adapter
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
    let mut features = wgpu::Features::empty();
    if has_timestamps {
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if has_pass_timestamps {
        features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
    }
    if !has_timestamps {
        eprintln!("WARNING: TIMESTAMP_QUERY not supported, GPU profiling disabled");
    } else if !has_pass_timestamps {
        eprintln!(
            "WARNING: TIMESTAMP_QUERY_INSIDE_PASSES not supported, pass-level profiling unavailable"
        );
    }

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("sluggrs demo device"),
            required_features: features,
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("failed to get device");

    let size = window.inner_size();
    let mut config = surface
        .get_default_config(&adapter, size.width.max(1), size.height.max(1))
        .expect("surface not supported");

    // Force sRGB so the shader's linear coverage output gets proper gamma correction.
    // Without this, nvidia surfaces may default to non-sRGB (Bgra8Unorm) which
    // produces washed-out, thin-looking text.
    config.format = config.format.add_srgb_suffix();
    surface.configure(&device, &config);

    let sf = window.scale_factor() as f32;
    eprintln!("Adapter: {:?}", adapter.get_info().name);
    eprintln!("Surface format: {:?}", config.format);
    eprintln!("Physical size: {}x{}", config.width, config.height);
    eprintln!("Scale factor: {sf}");

    // --- Optional licensed fonts (not in git, loaded from disk) ---
    let tisa_data = try_load_font("/home/folk/.local/share/fonts/TisaPro-Regular.otf");
    let berlingske_data =
        try_load_font("/home/folk/.local/share/fonts/BerlingskeSerif-Regular.ttf");

    let white = [1.0, 1.0, 1.0, 1.0];
    let light_gray = [0.75, 0.75, 0.75, 1.0];
    let gold = [0.95, 0.78, 0.2, 1.0];
    let cyan = [0.4, 0.85, 0.9, 1.0];
    let green = [0.5, 0.9, 0.5, 1.0];
    let pink = [0.95, 0.5, 0.65, 1.0];

    let mut all_prepared: Vec<PreparedGlyph> = Vec::new();
    let mut all_instances: Vec<GlyphInstance> = Vec::new();

    let mut curve_offset: u32 = 0;
    let mut band_offset: u32 = 0;

    // add_line takes logical sizes/positions and scales to physical pixels
    let mut add_line = |font_data: &[u8],
                        text: &str,
                        size: f32,
                        x: f32,
                        y: f32,
                        color: [f32; 4],
                        weight: Option<f32>| {
        let (prepared, instances) = prepare_text(
            font_data,
            text,
            size * sf,
            x * sf,
            y * sf,
            color,
            weight,
            curve_offset,
            band_offset,
        );
        for g in &prepared {
            curve_offset += (g.gpu_outline.curves.len() as u32) * 2;
            band_offset += (g.band_data.entries.len() / 4) as u32;
        }
        all_prepared.extend(prepared);
        all_instances.extend(instances);
    };

    let left = 40.0;
    let mut y = 30.0;

    // --- Sizes (Inter Variable, TTF) ---
    add_line(
        INTER_VARIABLE,
        "8px Inter: the quick brown fox jumps over the lazy dog — MSAA target",
        8.0,
        left,
        y,
        light_gray,
        None,
    );
    y += 16.0;

    add_line(
        INTER_VARIABLE,
        "10px Inter: the quick brown fox jumps over the lazy dog",
        10.0,
        left,
        y,
        light_gray,
        None,
    );
    y += 20.0;

    add_line(
        INTER_VARIABLE,
        "12px Inter: the quick brown fox jumps over the lazy dog",
        12.0,
        left,
        y,
        light_gray,
        None,
    );
    y += 24.0;

    add_line(
        INTER_VARIABLE,
        "16px Inter: the quick brown fox jumps over the lazy dog",
        16.0,
        left,
        y,
        white,
        None,
    );
    y += 30.0;

    add_line(
        INTER_VARIABLE,
        "24px Inter: the quick brown fox jumps over the lazy dog",
        24.0,
        left,
        y,
        white,
        None,
    );
    y += 40.0;

    add_line(
        INTER_VARIABLE,
        "48px Inter: Slug GPU text rendering",
        48.0,
        left,
        y,
        white,
        None,
    );
    y += 68.0;

    add_line(INTER_VARIABLE, "72px Inter", 72.0, left, y, gold, None);
    y += 90.0;

    // --- Inter Variable weights (wght axis) ---
    add_line(
        INTER_VARIABLE,
        "24px Inter Thin (wght=100): fine hairline strokes",
        24.0,
        left,
        y,
        light_gray,
        Some(100.0),
    );
    y += 38.0;

    add_line(
        INTER_VARIABLE,
        "24px Inter Light (wght=300): lightweight text",
        24.0,
        left,
        y,
        white,
        Some(300.0),
    );
    y += 38.0;

    add_line(
        INTER_VARIABLE,
        "24px Inter Regular (wght=400): standard weight",
        24.0,
        left,
        y,
        white,
        Some(400.0),
    );
    y += 38.0;

    add_line(
        INTER_VARIABLE,
        "24px Inter Bold (wght=700): heavy strokes",
        24.0,
        left,
        y,
        white,
        Some(700.0),
    );
    y += 38.0;

    add_line(
        INTER_VARIABLE,
        "24px Inter Black (wght=900): maximum weight",
        24.0,
        left,
        y,
        white,
        Some(900.0),
    );
    y += 44.0;

    // --- Weights (Roboto, TTF, separate files per weight) ---
    add_line(
        ROBOTO_THIN,
        "24px Roboto Thin (separate TTF)",
        24.0,
        left,
        y,
        light_gray,
        None,
    );
    y += 38.0;

    add_line(
        ROBOTO_REGULAR,
        "24px Roboto Regular (separate TTF)",
        24.0,
        left,
        y,
        white,
        None,
    );
    y += 38.0;

    add_line(
        ROBOTO_BOLD,
        "24px Roboto Bold (separate TTF): tight joins",
        24.0,
        left,
        y,
        white,
        None,
    );
    y += 44.0;

    // --- Font variety ---
    add_line(
        CASKAYDIA,
        "20px Caskaydia Cove (mono, TTF): fn main() { let x = 42; }",
        20.0,
        left,
        y,
        cyan,
        None,
    );
    y += 36.0;

    if let Some(ref tisa) = tisa_data {
        add_line(
            tisa,
            "22px Tisa Pro (serif, OTF/CFF cubic curves)",
            22.0,
            left,
            y,
            white,
            None,
        );
        y += 38.0;
    }

    if let Some(ref berlingske) = berlingske_data {
        add_line(
            berlingske,
            "22px Berlingske Serif (TTF)",
            22.0,
            left,
            y,
            white,
            None,
        );
        y += 38.0;
    }

    add_line(RUNES, "abcdefghijklm", 36.0, left, y, gold, None);
    y += 14.0;
    add_line(
        INTER_VARIABLE,
        "36px EBH Runes (OTF): decorative outlines",
        14.0,
        left,
        y,
        light_gray,
        None,
    );
    y += 40.0;

    // --- CFF/OTF cubic subdivision test ---
    let nimbus_roman =
        try_load_font("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf");
    let nimbus_sans = try_load_font("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf");
    let urw_bookman = try_load_font("/usr/share/fonts/opentype/urw-base35/URWBookman-Light.otf");
    let zapf_chancery = try_load_font("/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf");

    if let Some(ref font) = nimbus_roman {
        add_line(
            font,
            "24px Nimbus Roman (CFF): Sphinx of black quartz, judge my vow",
            24.0,
            left,
            y,
            white,
            None,
        );
        y += 38.0;
        add_line(
            font,
            "48px Nimbus Roman (CFF): QWERTY &@#",
            48.0,
            left,
            y,
            gold,
            None,
        );
        y += 68.0;
    }

    if let Some(ref font) = nimbus_sans {
        add_line(
            font,
            "24px Nimbus Sans (CFF): Pack my box with five dozen liquor jugs",
            24.0,
            left,
            y,
            white,
            None,
        );
        y += 38.0;
    }

    if let Some(ref font) = urw_bookman {
        add_line(
            font,
            "24px URW Bookman Light (CFF): Curved serifs test",
            24.0,
            left,
            y,
            cyan,
            None,
        );
        y += 38.0;
    }

    if let Some(ref font) = zapf_chancery {
        add_line(
            font,
            "30px Zapf Chancery (CFF italic): Flowing script curves",
            30.0,
            left,
            y,
            pink,
            None,
        );
        y += 48.0;
    }

    // --- Known artifact glyphs (bold-weight curve joins) ---
    add_line(
        INTER_VARIABLE,
        "36px Inter Bold artifact test: a & a & a & a",
        36.0,
        left,
        y,
        pink,
        Some(700.0),
    );
    y += 54.0;

    add_line(
        ROBOTO_BOLD,
        "36px Roboto Bold artifact test: a & a & a & a",
        36.0,
        left,
        y,
        pink,
        None,
    );
    y += 54.0;

    add_line(
        INTER_VARIABLE,
        "60px Inter Bold: & & & a a a",
        60.0,
        left,
        y,
        green,
        Some(700.0),
    );

    log::info!(
        "Prepared {} glyphs, {} instances across all lines",
        all_prepared.len(),
        all_instances.len()
    );

    // Build GPU textures
    let curve_texels = build_curve_texture(&all_prepared);
    let band_texels = build_band_texture(&all_prepared);

    // Curve texture: fixed width, wrap into rows (same as library's text_atlas.rs)
    let curve_w = CURVE_TEXTURE_WIDTH;
    let curve_count = curve_texels.len().max(1) as u32;
    let curve_h = curve_count.div_ceil(curve_w);
    let mut padded_curve_texels = curve_texels;
    padded_curve_texels.resize((curve_w * curve_h) as usize, [0i16; 4]);

    // Band texture: fixed width, wrap into rows
    let band_w = sluggrs::BAND_TEXTURE_WIDTH;
    let band_count = band_texels.len().max(1) as u32;
    let band_h = band_count.div_ceil(band_w);
    let mut padded_band_texels = band_texels;
    padded_band_texels.resize((band_w * band_h) as usize, [0i16; 4]);

    log::info!(
        "Curve texture: {curve_w}x{curve_h} ({curve_count} used), Band texture: {band_w}x{band_h} ({band_count} used)"
    );

    let curve_texture = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("curve texture"),
            size: wgpu::Extent3d {
                width: curve_w,
                height: curve_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Sint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&padded_curve_texels),
    );

    let band_texture = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("band texture"),
            size: wgpu::Extent3d {
                width: band_w,
                height: band_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Sint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&padded_band_texels),
    );

    let curve_view = curve_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let band_view = band_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // --- Shader ---
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("slug shader"),
        source: wgpu::ShaderSource::Wgsl(sluggrs::SIMPLE_SHADER_WGSL.into()),
    });

    // --- Bind group layouts ---
    let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("params bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("texture bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Sint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Sint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    // --- Buffers + bind groups ---
    let params = Params {
        screen_size: [config.width as f32, config.height as f32],
        scroll_offset: [0.0, 0.0],
        flags: 1, // MSAA+stem darkening on by default
        _pad: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("params bind group"),
        layout: &params_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buffer.as_entire_binding(),
        }],
    });

    let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("texture bind group"),
        layout: &texture_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&curve_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&band_view),
            },
        ],
    });

    let instance_count = all_instances.len() as u32;
    let instance_data = if all_instances.is_empty() {
        vec![GlyphInstance {
            screen_rect: [0.0; 4],
            em_rect: [0.0; 4],
            band_transform: [0.0; 4],
            glyph_data: [0; 4],
            color: [0.0; 4],
            depth: 0.0,
            ppem: 0.0,
            _pad: [0.0; 2],
        }]
    } else {
        all_instances
    };

    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("glyph instances"),
        contents: bytemuck::cast_slice(&instance_data),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // --- Pipeline ---
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("slug pipeline layout"),
        bind_group_layouts: &[&params_bgl, &texture_bgl],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("slug pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GlyphInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 16,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 32,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint32x4,
                        offset: 48,
                        shader_location: 3,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 64,
                        shader_location: 4,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 80,
                        shader_location: 5,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 84,
                        shader_location: 6,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 88,
                        shader_location: 7,
                    },
                ],
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let gpu_profiler = if has_timestamps {
        Some(
            wgpu_profiler::GpuProfiler::new(
                &device,
                wgpu_profiler::GpuProfilerSettings {
                    enable_timer_queries: true,
                    enable_debug_groups: false,
                    max_num_pending_frames: 3,
                },
            )
            .expect("Failed to create GPU profiler"),
        )
    } else {
        None
    };

    RenderState {
        surface,
        device,
        queue,
        config,
        pipeline,
        params_buffer,
        params_bind_group,
        texture_bind_group,
        instance_buffer,
        instance_count,
        zoom: 1.0,
        scroll: [0.0, 0.0],
        dragging: false,
        last_mouse: [0.0, 0.0],
        gpu_profiler,
    }
}

fn render(state: &mut RenderState) {
    let frame = match state.surface.get_current_texture() {
        Ok(f) => f,
        Err(e) => {
            log::error!("Failed to get surface texture: {e:?}");
            return;
        }
    };

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("slug render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.15,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            multiview_mask: None,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // GPU profiling — measures actual fragment shader execution time
        let query = state
            .gpu_profiler
            .as_ref()
            .map(|p| p.begin_query("text_render", &mut pass));

        pass.set_pipeline(&state.pipeline);
        pass.set_bind_group(0, &state.params_bind_group, &[]);
        pass.set_bind_group(1, &state.texture_bind_group, &[]);
        pass.set_vertex_buffer(0, state.instance_buffer.slice(..));
        pass.draw(0..4, 0..state.instance_count);

        if let (Some(profiler), Some(query)) = (&state.gpu_profiler, query) {
            profiler.end_query(&mut pass, query);
        }
    }

    if let Some(profiler) = &mut state.gpu_profiler {
        profiler.resolve_queries(&mut encoder);
    }

    state.queue.submit(std::iter::once(encoder.finish()));

    if let Some(profiler) = &mut state.gpu_profiler {
        let _ = profiler.end_frame();
        if let Some(results) = profiler.process_finished_frame(state.queue.get_timestamp_period()) {
            for r in &results {
                if let Some(time) = &r.time {
                    let ms = (time.end - time.start) * 1000.0;
                    eprintln!("gpu_{}_ms={ms:.3}", r.label);
                }
            }
        }
    }

    frame.present();
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App::new();
    let _ = event_loop.run_app(&mut app);
}
