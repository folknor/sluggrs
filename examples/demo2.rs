//! Interactive sluggrs demo using the library's TextRenderer/TextAtlas pipeline.
//! Arrow keys to scroll, mouse wheel to zoom, E to toggle MSAA+stem darkening.

use sluggrs::{
    Cache, ColorMode, Resolution, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

use cosmic_text::{
    Attrs, Buffer, Family, FontSystem, Metrics, Shaping, SwashCache, Weight,
};

use std::sync::Arc;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop, window::Window,
};

// Embedded fonts
const INTER_VARIABLE: &[u8] = include_bytes!("fonts/InterVariable.ttf");
const ROBOTO_REGULAR: &[u8] = include_bytes!("fonts/Roboto-Regular.ttf");
const ROBOTO_THIN: &[u8] = include_bytes!("fonts/Roboto-Thin.ttf");
const ROBOTO_BOLD: &[u8] = include_bytes!("fonts/Roboto-Bold.ttf");
const CASKAYDIA: &[u8] = include_bytes!("fonts/CaskaydiaCoveNerdFont-Regular.ttf");
const RUNES: &[u8] = include_bytes!("fonts/EBH Runes.otf");
const TWEMOJI_COLR: &[u8] = include_bytes!("fonts/TwemojiCOLRv0.ttf");
const NOTO_COLRV1: &[u8] = include_bytes!("fonts/NotoColorEmoji-Regular.ttf");

fn color(r: u8, g: u8, b: u8) -> cosmic_text::Color {
    cosmic_text::Color::rgb(r, g, b)
}

struct TextLine {
    buffer: Buffer,
    left: f32,
    top: f32,
    default_color: cosmic_text::Color,
}

struct RenderState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    text_renderer: TextRenderer,
    viewport: Viewport,
    lines: Vec<TextLine>,
    zoom: f32,
    scroll: [f32; 2],
    dragging: bool,
    last_mouse: [f32; 2],
    enhance: bool,
    gpu_profiler: Option<wgpu_profiler::GpuProfiler>,
}

struct App {
    state: Option<RenderState>,
    window: Option<Arc<Window>>,
    warmup_frames: u32,
}

impl App {
    fn new() -> Self {
        Self {
            state: None,
            window: None,
            warmup_frames: 5,
        }
    }
}

/// Load an optional font from disk; returns None if not found.
fn try_load_font(path: &str) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

/// Create a text line with a specific font family, weight, size, position, and color.
#[allow(clippy::too_many_arguments)]
fn make_line(
    font_system: &mut FontSystem,
    text: &str,
    family: Family<'_>,
    weight: Weight,
    font_size: f32,
    left: f32,
    top: f32,
    sf: f32,
    default_color: cosmic_text::Color,
) -> TextLine {
    let metrics = Metrics::new(font_size * sf, font_size * sf * 1.2);
    let mut buffer = Buffer::new(font_system, metrics);
    let attrs = Attrs::new().family(family).weight(weight);
    buffer.set_text(text, &attrs, Shaping::Advanced, None);
    buffer.shape_until_scroll(font_system, false);
    TextLine {
        buffer,
        left: left * sf,
        top: top * sf,
        default_color,
    }
}

fn build_lines(font_system: &mut FontSystem, sf: f32) -> Vec<TextLine> {
    let mut lines = Vec::new();
    let left = 40.0;
    let mut y = 30.0;

    let white = color(255, 255, 255);
    let light_gray = color(192, 192, 192);
    let gold = color(242, 199, 51);
    let cyan = color(102, 217, 230);
    let green = color(128, 230, 128);
    let pink = color(242, 128, 166);

    let inter = Family::Name("Inter Variable");
    let roboto = Family::Name("Roboto");
    let caskaydia = Family::Name("CaskaydiaCove Nerd Font");
    let runes = Family::Name("EBH Runes");
    let twemoji = Family::Name("Twemoji Mozilla");
    let noto_emoji = Family::Name("Noto Color Emoji");
    let w = |v: u16| Weight(v);

    macro_rules! line {
        ($family:expr, $weight:expr, $text:expr, $size:expr, $color:expr) => {
            lines.push(make_line(
                font_system, $text, $family, $weight, $size, left, y, sf, $color,
            ));
        };
    }

    // --- Sizes (Inter Variable) ---
    line!(inter, w(400), "8px Inter: the quick brown fox jumps over the lazy dog \u{2014} MSAA target", 8.0, light_gray);
    y += 16.0;
    line!(inter, w(400), "10px Inter: the quick brown fox jumps over the lazy dog", 10.0, light_gray);
    y += 20.0;
    line!(inter, w(400), "12px Inter: the quick brown fox jumps over the lazy dog", 12.0, light_gray);
    y += 24.0;
    line!(inter, w(400), "16px Inter: the quick brown fox jumps over the lazy dog", 16.0, white);
    y += 30.0;
    line!(inter, w(400), "24px Inter: the quick brown fox jumps over the lazy dog", 24.0, white);
    y += 40.0;
    line!(inter, w(400), "48px Inter: Slug GPU text rendering", 48.0, white);
    y += 68.0;
    line!(inter, w(400), "72px Inter", 72.0, gold);
    y += 90.0;

    // --- Inter Variable weights ---
    line!(inter, w(100), "24px Inter Thin (wght=100): fine hairline strokes", 24.0, light_gray);
    y += 38.0;
    line!(inter, w(300), "24px Inter Light (wght=300): lightweight text", 24.0, white);
    y += 38.0;
    line!(inter, w(400), "24px Inter Regular (wght=400): standard weight", 24.0, white);
    y += 38.0;
    line!(inter, w(700), "24px Inter Bold (wght=700): heavy strokes", 24.0, white);
    y += 38.0;
    line!(inter, w(900), "24px Inter Black (wght=900): maximum weight", 24.0, white);
    y += 44.0;

    // --- Roboto weights ---
    line!(roboto, Weight::THIN, "24px Roboto Thin (separate TTF)", 24.0, light_gray);
    y += 38.0;
    line!(roboto, Weight::NORMAL, "24px Roboto Regular (separate TTF)", 24.0, white);
    y += 38.0;
    line!(roboto, Weight::BOLD, "24px Roboto Bold (separate TTF): tight joins", 24.0, white);
    y += 44.0;

    // --- Font variety ---
    line!(caskaydia, w(400), "20px Caskaydia Cove (mono, TTF): fn main() { let x = 42; }", 20.0, cyan);
    y += 36.0;

    // Optional disk fonts
    if font_system.db().faces().any(|f| {
        f.families.iter().any(|(name, _)| name == "Tisa Pro")
    }) {
        line!(Family::Name("Tisa Pro"), w(400), "22px Tisa Pro (serif, OTF/CFF cubic curves)", 22.0, white);
        y += 38.0;
    }

    if font_system.db().faces().any(|f| {
        f.families.iter().any(|(name, _)| name == "Berlingske Serif")
    }) {
        line!(Family::Name("Berlingske Serif"), w(400), "22px Berlingske Serif (TTF)", 22.0, white);
        y += 38.0;
    }

    line!(runes, w(400), "abcdefghijklm", 36.0, gold);
    y += 14.0;
    line!(inter, w(400), "36px EBH Runes (OTF): decorative outlines", 14.0, light_gray);
    y += 40.0;

    // --- COLRv0 color emoji ---
    y += 16.0;
    line!(twemoji, w(400), "\u{1F600}\u{1F60D}\u{1F525}\u{2764}\u{1F680}\u{1F308}\u{1F3B5}\u{2B50}", 48.0, white);
    y += 24.0;
    line!(inter, w(400), "48px Twemoji COLRv0: color vector emoji", 14.0, light_gray);
    y += 40.0;

    // --- COLRv1 gradient emoji ---
    y += 16.0;
    line!(noto_emoji, w(400), "\u{1F600}\u{1F60D}\u{1F525}\u{2764}\u{1F680}\u{1F308}\u{1F3B5}\u{2B50}", 48.0, white);
    y += 24.0;
    line!(inter, w(400), "48px Noto COLRv1: gradient vector emoji", 14.0, light_gray);
    y += 40.0;

    // --- CFF/OTF cubic subdivision ---
    let cff_fonts: &[(&str, &str, f32, cosmic_text::Color)] = &[
        ("Nimbus Roman", "24px Nimbus Roman (CFF): Sphinx of black quartz, judge my vow", 24.0, white),
        ("Nimbus Roman", "48px Nimbus Roman (CFF): QWERTY &@#", 48.0, gold),
        ("Nimbus Sans", "24px Nimbus Sans (CFF): Pack my box with five dozen liquor jugs", 24.0, white),
        ("URW Bookman", "24px URW Bookman Light (CFF): Curved serifs test", 24.0, cyan),
        ("Z003", "30px Zapf Chancery (CFF italic): Flowing script curves", 30.0, pink),
    ];
    for (family_name, text, size, clr) in cff_fonts {
        if font_system.db().faces().any(|f| {
            f.families.iter().any(|(name, _)| name == *family_name)
        }) {
            line!(Family::Name(family_name), w(400), text, *size, *clr);
            let spacing = (*size * 1.5).max(38.0);
            y += spacing;
        }
    }

    // --- Known artifact glyphs ---
    line!(inter, w(700), "36px Inter Bold artifact test: a & a & a & a", 36.0, pink);
    y += 54.0;
    line!(roboto, Weight::BOLD, "36px Roboto Bold artifact test: a & a & a & a", 36.0, pink);
    y += 54.0;
    line!(inter, w(700), "60px Inter Bold: & & & a a a", 60.0, green);

    lines
}

async fn init_render_state(window: Arc<Window>) -> RenderState {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let surface = instance.create_surface(Arc::clone(&window)).expect("render failed");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("failed to find adapter");

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

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("sluggrs demo2 device"),
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
    config.format = config.format.add_srgb_suffix();
    surface.configure(&device, &config);

    let sf = window.scale_factor() as f32;
    eprintln!("Adapter: {:?}", adapter.get_info().name);
    eprintln!("Surface format: {:?}", config.format);
    eprintln!("Physical size: {}x{}", config.width, config.height);
    eprintln!("Scale factor: {sf}");

    // --- Font system ---
    let mut font_system = FontSystem::new();
    font_system.db_mut().load_font_data(INTER_VARIABLE.to_vec());
    font_system.db_mut().load_font_data(ROBOTO_REGULAR.to_vec());
    font_system.db_mut().load_font_data(ROBOTO_THIN.to_vec());
    font_system.db_mut().load_font_data(ROBOTO_BOLD.to_vec());
    font_system.db_mut().load_font_data(CASKAYDIA.to_vec());
    font_system.db_mut().load_font_data(RUNES.to_vec());
    font_system.db_mut().load_font_data(TWEMOJI_COLR.to_vec());
    font_system.db_mut().load_font_data(NOTO_COLRV1.to_vec());

    // Optional disk fonts
    for path in [
        "/home/folk/.local/share/fonts/TisaPro-Regular.otf",
        "/home/folk/.local/share/fonts/BerlingskeSerif-Regular.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
        "/usr/share/fonts/opentype/urw-base35/URWBookman-Light.otf",
        "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    ] {
        if let Some(data) = try_load_font(path) {
            font_system.db_mut().load_font_data(data);
        }
    }

    // --- Library pipeline ---
    let cache = Cache::new(&device);
    let mut atlas = TextAtlas::with_color_mode(
        &device, &queue, &cache, config.format, ColorMode::Accurate,
    );
    let text_renderer = TextRenderer::new(
        &mut atlas, &device, wgpu::MultisampleState::default(), None,
    );
    let mut viewport = Viewport::new(&device, &cache);
    viewport.update(&queue, Resolution {
        width: config.width,
        height: config.height,
    });

    let lines = build_lines(&mut font_system, sf);
    eprintln!("Built {} text lines", lines.len());

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
        font_system,
        swash_cache: SwashCache::new(),
        atlas,
        text_renderer,
        viewport,
        lines,
        zoom: 1.0,
        scroll: [0.0, 0.0],
        dragging: false,
        last_mouse: [0.0, 0.0],
        enhance: true,
        gpu_profiler,
    }
}

fn render(state: &mut RenderState) {
    let frame = match state.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(f)
        | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
        other => {
            log::error!("Failed to get surface texture: {other:?}");
            return;
        }
    };

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    // Update viewport for current zoom
    let vp_w = (state.config.width as f32 / state.zoom) as u32;
    let vp_h = (state.config.height as f32 / state.zoom) as u32;
    state.viewport.update(&state.queue, Resolution {
        width: vp_w.max(1),
        height: vp_h.max(1),
    });
    state.viewport.set_scroll_offset(&state.queue, state.scroll);

    // Build text areas from pre-built lines
    let text_areas: Vec<TextArea<'_>> = state.lines.iter().map(|line| TextArea {
        buffer: &line.buffer,
        left: line.left,
        top: line.top,
        scale: 1.0,
        bounds: TextBounds {
            left: 0,
            top: 0,
            right: vp_w as i32,
            bottom: vp_h as i32,
        },
        default_color: line.default_color,
    }).collect();

    let encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });

    state
        .text_renderer
        .prepare(
            &state.device,
            &state.queue,
            &encoder,
            &mut state.font_system,
            &mut state.atlas,
            &state.viewport,
            text_areas,
            &mut state.swash_cache,
        )
        .expect("prepare failed");

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

        let query = state
            .gpu_profiler
            .as_ref()
            .map(|p| p.begin_query("text_render", &mut pass));

        state
            .text_renderer
            .render(&state.atlas, &state.viewport, &mut pass)
            .expect("render failed");

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
    state.atlas.trim();
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
                        .with_title("sluggrs demo2")
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
                            state.enhance = !state.enhance;
                            state.viewport.set_msaa_hint(&state.queue, state.enhance);
                            eprintln!(
                                "MSAA + stem darkening: {}",
                                if state.enhance { "ON" } else { "OFF" }
                            );
                        }
                        _ => return,
                    }
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

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop failed");
}
