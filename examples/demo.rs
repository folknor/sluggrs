use sluggrs::band::{self, build_bands, CurveLocation};
use sluggrs::outline::{char_to_glyph_id, extract_outline};
use sluggrs::prepare::{self, GpuOutline};

use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    window::Window,
};

const FONT_BYTES: &[u8] = include_bytes!("fonts/InterVariable.ttf");

/// Per-instance vertex data for a glyph (matches GlyphInstance in shader).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GlyphInstance {
    screen_rect: [f32; 4],     // x, y, width, height
    em_rect: [f32; 4],         // min_x, min_y, max_x, max_y
    band_transform: [f32; 4],  // scale_x, scale_y, offset_x, offset_y
    glyph_data: [u32; 4],      // glyph_loc.x, glyph_loc.y, band_max.x, band_max.y
    color: [f32; 4],           // RGBA
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    screen_size: [f32; 2],
    _pad: [f32; 2],
}

/// Prepared glyph data ready for GPU upload.
struct PreparedGlyph {
    gpu_outline: GpuOutline,
    band_data: band::BandData,
}

/// Build curve texture data from glyph outlines.
fn build_curve_texture(glyphs: &[PreparedGlyph]) -> Vec<[f32; 4]> {
    let mut texels: Vec<[f32; 4]> = Vec::new();

    for glyph in glyphs {
        for curve in &glyph.gpu_outline.curves {
            // Texel 1: p1.x, p1.y, p2.x, p2.y
            texels.push([curve.p1[0], curve.p1[1], curve.p2[0], curve.p2[1]]);
            // Texel 2: p3.x, p3.y, 0, 0
            texels.push([curve.p3[0], curve.p3[1], 0.0, 0.0]);
        }
    }

    if texels.is_empty() {
        texels.push([0.0; 4]);
    }

    texels
}

/// Build band texture data from prepared glyphs.
fn build_band_texture(glyphs: &[PreparedGlyph]) -> Vec<[u32; 4]> {
    let mut texels: Vec<[u32; 4]> = Vec::new();

    for glyph in glyphs {
        for chunk in glyph.band_data.entries.chunks(4) {
            let mut texel = [0u32; 4];
            for (i, &val) in chunk.iter().enumerate() {
                texel[i] = val;
            }
            texels.push(texel);
        }
    }

    if texels.is_empty() {
        texels.push([0u32; 4]);
    }

    texels
}

/// Prepare glyphs for a string of text.
/// `base_curve_offset` and `base_band_offset` allow chaining multiple calls.
fn prepare_text(
    text: &str,
    font_size: f32,
    start_x: f32,
    start_y: f32,
    base_curve_offset: u32,
    base_band_offset: u32,
) -> (Vec<PreparedGlyph>, Vec<GlyphInstance>) {
    let font = skrifa::FontRef::new(FONT_BYTES).expect("failed to parse font");
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

    // Simple horizontal advance lookup
    let hmtx = {
        use skrifa::raw::TableProvider;
        font.hmtx().expect("no hmtx table")
    };

    for ch in text.chars() {
        let glyph_id = match char_to_glyph_id(FONT_BYTES, ch) {
            Some(id) => id,
            None => continue,
        };

        // Get advance width
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

        let outline = match extract_outline(FONT_BYTES, glyph_id) {
            Some(o) => o,
            None => {
                // Space or non-drawing glyph — just advance
                cursor_x += advance * scale;
                continue;
            }
        };

        let gpu_outline = prepare::prepare_outline(&outline);
        let num_curves = gpu_outline.curves.len();

        // Build curve locations (each curve takes 2 texels in the curve texture)
        let curve_locations: Vec<CurveLocation> = (0..num_curves)
            .map(|i| CurveLocation {
                x: curve_offset + (i as u32) * 2,
                y: 0,
            })
            .collect();

        // Choose band counts based on glyph complexity
        let band_count =
            if num_curves < 10 {
                4
            } else if num_curves < 30 {
                8
            } else {
                12
            };
        let band_data = build_bands(&gpu_outline, &curve_locations, band_count, band_count);

        let [min_x, min_y, max_x, max_y] = gpu_outline.bounds;

        // Screen-space rectangle for this glyph
        let screen_x = cursor_x + min_x * scale;
        let screen_y = start_y - max_y * scale; // flip Y: font is Y-up
        let screen_w = (max_x - min_x) * scale;
        let screen_h = (max_y - min_y) * scale;

        let glyph_band_texel_count = (band_data.entries.len() / 4) as u32;

        instances.push(GlyphInstance {
            screen_rect: [screen_x, screen_y, screen_w, screen_h],
            em_rect: [min_x, min_y, max_x, max_y],
            band_transform: band_data.band_transform,
            glyph_data: [
                band_offset,       // glyph data x in band texture
                0,                 // glyph data y in band texture (row 0 for now)
                (band_count - 1) as u32, // band_max_x (max index, not count)
                (band_count - 1) as u32, // band_max_y (max index, not count)
            ],
            color: [1.0, 1.0, 1.0, 1.0], // white text
        });

        prepared.push(PreparedGlyph {
            gpu_outline,
            band_data,
        });

        curve_offset += (num_curves as u32) * 2; // 2 texels per curve
        band_offset += glyph_band_texel_count;
        cursor_x += advance * scale;
    }

    (prepared, instances)
}

struct App {
    state: Option<RenderState>,
    window: Option<Arc<Window>>,
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
}

impl App {
    fn new() -> Self {
        Self {
            state: None,
            window: None,
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
                        .with_title("Slug Font Rendering PoC")
                        .with_inner_size(winit::dpi::LogicalSize::new(1024, 600)),
                )
                .expect("failed to create window"),
        );

        let state = pollster::block_on(init_render_state(window.clone()));
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

                    let params = Params {
                        screen_size: [state.config.width as f32, state.config.height as f32],
                        _pad: [0.0; 2],
                    };
                    state
                        .queue
                        .write_buffer(&state.params_buffer, 0, bytemuck::bytes_of(&params));

                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &self.state {
                    render(state);
                }
            }
            _ => {}
        }
    }
}

async fn init_render_state(window: Arc<Window>) -> RenderState {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let surface = instance
        .create_surface(window.clone())
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

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("slug-glyph device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("failed to get device");

    let size = window.inner_size();
    let config = surface
        .get_default_config(&adapter, size.width.max(1), size.height.max(1))
        .expect("surface not supported");
    surface.configure(&device, &config);

    // --- Prepare glyph data ---
    let (prepared_glyphs, instances) = prepare_text("Hello, Slug!", 72.0, 50.0, 300.0, 0, 0);

    log::info!(
        "Prepared {} glyphs, {} instances",
        prepared_glyphs.len(),
        instances.len()
    );
    for (i, g) in prepared_glyphs.iter().enumerate() {
        log::info!(
            "  glyph {}: {} curves, bounds [{:.0}, {:.0}, {:.0}, {:.0}]",
            i,
            g.gpu_outline.curves.len(),
            g.gpu_outline.bounds[0],
            g.gpu_outline.bounds[1],
            g.gpu_outline.bounds[2],
            g.gpu_outline.bounds[3]
        );
    }

    // Build GPU textures
    let curve_texels = build_curve_texture(&prepared_glyphs);
    let band_texels = build_band_texture(&prepared_glyphs);

    let curve_texture_width = curve_texels.len().max(1) as u32;
    let band_texture_width = band_texels.len().max(1) as u32;

    log::info!(
        "Curve texture: {} texels, Band texture: {} texels",
        curve_texture_width,
        band_texture_width
    );

    let curve_texture = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("curve texture"),
            size: wgpu::Extent3d {
                width: curve_texture_width,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&curve_texels),
    );

    let band_texture = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("band texture"),
            size: wgpu::Extent3d {
                width: band_texture_width,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&band_texels),
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
            visibility: wgpu::ShaderStages::VERTEX,
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
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
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
        _pad: [0.0; 2],
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

    let instance_data = if instances.is_empty() {
        // Need at least one dummy instance for buffer creation
        vec![GlyphInstance {
            screen_rect: [0.0; 4],
            em_rect: [0.0; 4],
            band_transform: [0.0; 4],
            glyph_data: [0; 4],
            color: [0.0; 4],
        }]
    } else {
        instances
    };
    let instance_count = if instance_data[0].color[3] == 0.0 && instance_data.len() == 1 {
        0
    } else {
        instance_data.len() as u32
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
                ],
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
    }
}

fn render(state: &RenderState) {
    let frame = match state.surface.get_current_texture() {
        Ok(f) => f,
        Err(e) => {
            log::error!("Failed to get surface texture: {:?}", e);
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

        pass.set_pipeline(&state.pipeline);
        pass.set_bind_group(0, &state.params_bind_group, &[]);
        pass.set_bind_group(1, &state.texture_bind_group, &[]);
        pass.set_vertex_buffer(0, state.instance_buffer.slice(..));
        pass.draw(0..4, 0..state.instance_count);
    }

    state.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App::new();
    let _ = event_loop.run_app(&mut app);
}
