//! Decoration showcase for the `TextArea::decorations` list.
//!
//! Each line is its own text area with its own decorations, so the demo
//! covers the axes that matter: spread, offset, color, font size, weight,
//! decoration count and order, and the glyph classes that are deliberately
//! undecorated (COLRv0/COLRv1 emoji).
//!
//! A decoration is a solid morphological shape: the glyph dilated by
//! `spread` and translated by `offset`. Spread alone is an outline, offset
//! alone is a hard drop shadow, and both together is a spread shadow.
//!
//! Mouse wheel zooms (spread and offset are logical, so zoom shows the
//! physical scaling), B toggles all decorations off for an A/B against the
//! plain path, E toggles MSAA + stem darkening.

use sluggrs::{
    Cache, ColorMode, DecorationMode, Resolution, TextArea, TextAtlas, TextBounds, TextDecoration,
    TextRenderer, Viewport,
};

use cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping, SwashCache, Weight};

use std::sync::Arc;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop, window::Window,
};

const INTER_VARIABLE: &[u8] = include_bytes!("fonts/InterVariable.ttf");
const ROBOTO_BOLD: &[u8] = include_bytes!("fonts/Roboto-Bold.ttf");
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
    decorations: Vec<TextDecoration>,
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
    borders_on: bool,
    enhance: bool,
}

struct App {
    state: Option<RenderState>,
    window: Option<Arc<Window>>,
}

/// Create one bordered line: family, weight, size, position, fill, border.
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
    decorations: Vec<TextDecoration>,
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
        decorations,
    }
}

fn build_lines(font_system: &mut FontSystem, sf: f32) -> Vec<TextLine> {
    let mut lines = Vec::new();
    let left = 40.0;
    let mut y = 24.0;

    let white = color(255, 255, 255);
    let light_gray = color(192, 192, 192);
    let gold = color(242, 199, 51);
    let cyan = color(102, 217, 230);
    let pink = color(242, 128, 166);
    let black = color(0, 0, 0);
    let navy = color(24, 32, 72);
    let crimson = color(196, 48, 64);

    let inter = Family::Name("Inter Variable");
    let roboto = Family::Name("Roboto");
    let runes = Family::Name("EBH Runes");
    let twemoji = Family::Name("Twemoji COLRv0");
    let noto_emoji = Family::Name("Noto Color Emoji");
    let w = |v: u16| Weight(v);

    let border = |clr: cosmic_text::Color, width: f32| vec![TextDecoration::outline(clr, width)];
    let none = Vec::new;

    macro_rules! line {
        ($family:expr, $weight:expr, $text:expr, $size:expr, $color:expr, $border:expr) => {
            lines.push(make_line(
                font_system,
                $text,
                $family,
                $weight,
                $size,
                left,
                y,
                sf,
                $color,
                $border,
            ));
        };
    }

    macro_rules! caption {
        ($text:expr) => {
            line!(inter, w(400), $text, 13.0, light_gray, none());
            y += 22.0;
        };
    }

    // --- Width ladder: same fill, same size, growing border ---
    caption!("Width ladder (36px Inter Bold, black border, logical widths)");
    for width in [0.5_f32, 1.0, 2.0, 3.0, 5.0] {
        let text = format!("{width}px border: Slug GPU text rendering");
        line!(inter, w(700), &text, 36.0, white, border(black, width));
        y += 52.0;
    }
    y += 14.0;

    // --- Border color against the same fill ---
    caption!("Border color (48px Inter Black, 2px border)");
    line!(inter, w(900), "navy on gold", 48.0, gold, border(navy, 2.0));
    y += 66.0;
    line!(
        inter,
        w(900),
        "crimson on white",
        48.0,
        white,
        border(crimson, 2.0)
    );
    y += 66.0;
    line!(
        inter,
        w(900),
        "gold on navy fill",
        48.0,
        navy,
        border(gold, 2.0)
    );
    y += 74.0;

    // --- Small sizes: where a border most easily swallows the fill ---
    caption!("Small sizes with a 1px border (fill can be swallowed)");
    for size in [10.0_f32, 12.0, 16.0, 24.0] {
        let text = format!("{size}px Inter with a 1px black border: the quick brown fox");
        line!(inter, w(400), &text, size, white, border(black, 1.0));
        y += size * 1.5 + 6.0;
    }
    y += 12.0;

    // --- Weight interaction: thin strokes vs a heavy border ---
    caption!("Weight vs border (28px, 1.5px black border)");
    line!(
        inter,
        w(100),
        "Inter Thin (wght=100): hairlines under a border",
        28.0,
        white,
        border(black, 1.5)
    );
    y += 44.0;
    line!(
        roboto,
        Weight::BOLD,
        "Roboto Bold (separate TTF): tight joins under a border",
        28.0,
        white,
        border(black, 1.5)
    );
    y += 52.0;

    // --- CFF/OTF outlines ---
    caption!("OTF/CFF cubic outlines (36px EBH Runes, 2px border)");
    line!(
        runes,
        w(400),
        "abcdefghijklm",
        36.0,
        cyan,
        border(navy, 2.0)
    );
    y += 60.0;

    // --- Deliberately borderless glyph classes ---
    caption!("COLRv0 / COLRv1 emoji: borderless by design, border is ignored");
    line!(
        twemoji,
        w(400),
        "\u{1F600}\u{1F60D}\u{1F525}\u{2764}\u{1F680}",
        48.0,
        white,
        border(black, 3.0)
    );
    y += 62.0;
    line!(
        noto_emoji,
        w(400),
        "\u{1F600}\u{1F60D}\u{1F525}\u{2764}\u{1F680}",
        48.0,
        white,
        border(black, 3.0)
    );
    y += 66.0;

    // --- Hard drop shadows: offset, no dilation ---
    caption!("Hard drop shadow (40px Inter Bold, spread 0, offset only)");
    for (dx, dy) in [(2.0_f32, 2.0_f32), (4.0, 4.0), (-3.0, 3.0)] {
        let text = format!("offset ({dx}, {dy}): shadow with no spread");
        line!(
            inter,
            w(700),
            &text,
            40.0,
            white,
            vec![TextDecoration::shadow(black, dx, dy)]
        );
        y += 58.0;
    }
    y += 12.0;

    // --- Spread plus offset on one decoration ---
    caption!("Spread + offset together (40px, 2px spread, offset (3, 3))");
    line!(
        inter,
        w(700),
        "a dilated, displaced shadow",
        40.0,
        white,
        vec![TextDecoration {
            color: navy,
            spread: 2.0,
            offset: [3.0, 3.0],
            mode: DecorationMode::Solid,
        }]
    );
    y += 62.0;

    // --- Several decorations on one area, CSS back-to-front order ---
    caption!("Three decorations, back-to-front like CSS: first entry on top");
    line!(
        inter,
        w(700),
        "outline over cyan over crimson",
        44.0,
        white,
        vec![
            TextDecoration::outline(black, 1.5),
            TextDecoration::shadow(cyan, 4.0, 4.0),
            TextDecoration::shadow(crimson, 8.0, 8.0),
        ]
    );
    y += 70.0;

    // --- A/B reference: identical line with no decoration at all ---
    caption!("Reference: no decorations (the zero-cost normal path)");
    line!(
        inter,
        w(700),
        "36px Inter Bold, decorations: &[]",
        36.0,
        pink,
        none()
    );

    lines
}

async fn init_render_state(window: Arc<Window>) -> RenderState {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let surface = instance
        .create_surface(Arc::clone(&window))
        .expect("render failed");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("failed to find adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("sluggrs borders device"),
            required_features: wgpu::Features::empty(),
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
    eprintln!("Scale factor: {sf}");

    // Isolated database: only the embedded fonts, so a system emoji font
    // cannot win the family match and route the emoji lines through the
    // raster fallback instead of the COLR vector paths.
    let mut db = cosmic_text::fontdb::Database::new();
    db.load_font_data(INTER_VARIABLE.to_vec());
    db.load_font_data(ROBOTO_BOLD.to_vec());
    db.load_font_data(RUNES.to_vec());
    db.load_font_data(TWEMOJI_COLR.to_vec());
    db.load_font_data(NOTO_COLRV1.to_vec());
    let mut font_system = FontSystem::new_with_locale_and_db("en-US".to_string(), db);

    let cache = Cache::new(&device);
    let mut atlas =
        TextAtlas::with_color_mode(&device, &queue, &cache, config.format, ColorMode::Accurate);
    let text_renderer =
        TextRenderer::new(&mut atlas, &device, wgpu::MultisampleState::default(), None);
    let mut viewport = Viewport::new(&device, &cache);
    viewport.update(
        &queue,
        Resolution {
            width: config.width,
            height: config.height,
        },
    );

    let lines = build_lines(&mut font_system, sf);
    eprintln!("Built {} text lines", lines.len());
    eprintln!("wheel = zoom, drag = pan, arrows = scroll, B = borders, E = MSAA, Home = reset");

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
        borders_on: true,
        enhance: true,
    }
}

fn render(state: &mut RenderState) {
    let frame = match state.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
        other => {
            log::error!("Failed to get surface texture: {other:?}");
            return;
        }
    };

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let vp_w = (state.config.width as f32 / state.zoom) as u32;
    let vp_h = (state.config.height as f32 / state.zoom) as u32;
    state.viewport.update(
        &state.queue,
        Resolution {
            width: vp_w.max(1),
            height: vp_h.max(1),
        },
    );
    state.viewport.set_scroll_offset(&state.queue, state.scroll);

    let borders_on = state.borders_on;
    let text_areas: Vec<TextArea<'_>> = state
        .lines
        .iter()
        .map(|line| TextArea {
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
            decorations: if borders_on { &line.decorations } else { &[] },
        })
        .collect();

    let encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prepare encoder"),
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

        state
            .text_renderer
            .render(&state.atlas, &state.viewport, &mut pass)
            .expect("render failed");
    }

    state.queue.submit(std::iter::once(encoder.finish()));
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
                        .with_title("sluggrs borders")
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
                        winit::keyboard::KeyCode::KeyB => {
                            state.borders_on = !state.borders_on;
                            eprintln!("borders: {}", if state.borders_on { "ON" } else { "OFF" });
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
    let mut app = App {
        state: None,
        window: None,
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}
