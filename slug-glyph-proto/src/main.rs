mod band;
mod outline;

use band::{build_bands, CurveLocation};
use outline::{char_to_glyph_id, extract_outline, GlyphOutline};

use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    window::Window,
};

const FONT_BYTES: &[u8] = include_bytes!("../fonts/InterVariable.ttf");

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
    outline: GlyphOutline,
    band_data: band::BandData,
    /// Where this glyph's curves start in the curve texture (texel x offset).
    curve_offset: u32,
    /// Where this glyph's band data starts in the band texture (texel x offset).
    band_offset: u32,
}

/// Build curve texture data from glyph outlines.
fn build_curve_texture(glyphs: &[PreparedGlyph]) -> Vec<[f32; 4]> {
    let mut texels: Vec<[f32; 4]> = Vec::new();

    for glyph in glyphs {
        for curve in &glyph.outline.curves {
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

        let num_curves = outline.curves.len();

        // Build curve locations (each curve takes 2 texels in the curve texture)
        let curve_locations: Vec<CurveLocation> = (0..num_curves)
            .map(|i| CurveLocation {
                x: curve_offset + (i as u32) * 2,
                y: 0,
            })
            .collect();

        // Choose band counts based on glyph complexity
        // Use 1 band for very simple shapes (all straight lines) to avoid
        // band coverage gaps, higher counts for complex curves
        let all_linear = outline.curves.iter().all(|c| {
            let mid_x = (c.p1[0] + c.p3[0]) * 0.5;
            let mid_y = (c.p1[1] + c.p3[1]) * 0.5;
            (c.p2[0] - mid_x).abs() < 0.01 && (c.p2[1] - mid_y).abs() < 0.01
        });
        let band_count =
            if all_linear {
                1
            } else if num_curves < 10 {
                4
            } else if num_curves < 30 {
                8
            } else {
                12
            };
        let band_data = build_bands(&outline, &curve_locations, band_count, band_count);

        // Debug: dump curve data for comma and period
        if ch == ',' || ch == '.' {
            println!("=== '{}' DEBUG ===", ch);
            println!("  {} curves, band_count={}", num_curves, band_count);
            for (i, c) in outline.curves.iter().enumerate() {
                println!(
                    "  curve {}: p1=({:.1}, {:.1}) p2=({:.1}, {:.1}) p3=({:.1}, {:.1})",
                    i, c.p1[0], c.p1[1], c.p2[0], c.p2[1], c.p3[0], c.p3[1]
                );
            }
            println!("  bounds: {:?}", outline.bounds);
            println!("  curve_offset={}, band_offset={}", curve_offset, band_offset);
            println!("  band_transform={:?}", band_data.band_transform);

            let band_texels: Vec<[u32; 4]> = band_data
                .entries
                .chunks(4)
                .map(|chunk| {
                    let mut texel = [0u32; 4];
                    for (i, &v) in chunk.iter().enumerate() {
                        texel[i] = v;
                    }
                    texel
                })
                .collect();

            println!("  band texels (relative to glyph start):");
            for (i, texel) in band_texels.iter().enumerate() {
                println!("    {:>2}: {:?}", i, texel);
            }

            let hcount = band_count as usize;
            let vcount = band_count as usize;

            println!("  horizontal headers:");
            for i in 0..hcount {
                let header = band_texels[i];
                println!("    h{}: count={} offset={}", i, header[0], header[1]);
                for ci in 0..header[0] as usize {
                    let cref = band_texels[header[1] as usize + ci];
                    println!(
                        "      ci={}: curve_tex=({}, {}), curve_idx={}",
                        ci,
                        cref[0],
                        cref[1],
                        (cref[0] - curve_offset) / 2
                    );
                }
            }

            println!("  vertical headers:");
            for i in 0..vcount {
                let header = band_texels[hcount + i];
                println!("    v{}: count={} offset={}", i, header[0], header[1]);
                for ci in 0..header[0] as usize {
                    let cref = band_texels[header[1] as usize + ci];
                    println!(
                        "      ci={}: curve_tex=({}, {}), curve_idx={}",
                        ci,
                        cref[0],
                        cref[1],
                        (cref[0] - curve_offset) / 2
                    );
                }
            }

            println!("===================");
        }

        let [min_x, min_y, max_x, max_y] = outline.bounds;

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
            outline,
            band_data,
            curve_offset,
            band_offset,
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
    let (mut prepared_glyphs, mut instances) = prepare_text("Hello, Slug!", 72.0, 50.0, 300.0, 0, 0);
    // Calculate end offsets from first batch
    let curve_end: u32 = prepared_glyphs.iter().map(|g| g.outline.curves.len() as u32 * 2).sum();
    let band_end: u32 = prepared_glyphs.iter().map(|g| (g.band_data.entries.len() / 4) as u32).sum();
    // Big comma for debugging
    let (p2, i2) = prepare_text(",.,", 300.0, 50.0, 550.0, curve_end, band_end);
    prepared_glyphs.extend(p2);
    instances.extend(i2);

    log::info!(
        "Prepared {} glyphs, {} instances",
        prepared_glyphs.len(),
        instances.len()
    );
    for (i, g) in prepared_glyphs.iter().enumerate() {
        log::info!(
            "  glyph {}: {} curves, bounds [{:.0}, {:.0}, {:.0}, {:.0}]",
            i,
            g.outline.curves.len(),
            g.outline.bounds[0],
            g.outline.bounds[1],
            g.outline.bounds[2],
            g.outline.bounds[3]
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
        source: wgpu::ShaderSource::Wgsl(include_str!("simple_shader.wgsl").into()),
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

fn cpu_calc_root_code(y1: f32, y2: f32, y3: f32) -> u32 {
    let i1 = y1.to_bits() >> 31;
    let i2 = y2.to_bits() >> 30;
    let i3 = y3.to_bits() >> 29;
    let mut shift = (i2 & 2) | (i1 & !2);
    shift = (i3 & 4) | (shift & !4);
    (0x2E74u32 >> shift) & 0x0101
}

fn cpu_solve_vert_poly(p12: [f32; 4], p3: [f32; 2]) -> [f32; 2] {
    let ax = p12[0] - p12[2] * 2.0 + p3[0];
    let ay = p12[1] - p12[3] * 2.0 + p3[1];
    let bx = p12[0] - p12[2];
    let by = p12[1] - p12[3];

    let (t1, t2);
    if ax.abs() < 1.0 / 65536.0 {
        let rb = 0.5 / bx;
        let lin = p12[0] * rb;
        t1 = lin;
        t2 = lin;
    } else {
        let ra = 1.0 / ax;
        let d = (bx * bx - ax * p12[0]).max(0.0).sqrt();
        t1 = (bx - d) * ra;
        t2 = (bx + d) * ra;
    }

    [
        (ay * t1 - by * 2.0) * t1 + p12[1],
        (ay * t2 - by * 2.0) * t2 + p12[1],
    ]
}

fn cpu_simulate_comma() {
    println!("\n=== CPU SIMULATION ===");

    // Comma curves (from debug output)
    let curves: Vec<([f32; 2], [f32; 2], [f32; 2])> = vec![
        ([128.0, -359.0], [172.0, -75.5], [216.0, 208.0]),   // curve 0
        ([216.0, 208.0], [319.0, 208.0], [422.0, 208.0]),     // curve 1
        ([422.0, 208.0], [344.0, -75.5], [266.0, -359.0]),    // curve 2
        ([266.0, -359.0], [197.0, -359.0], [128.0, -359.0]),  // curve 3
    ];

    // Test pixel in em-space (center of comma)
    let render_coord = [275.0f32, -75.0f32];
    // Approximate pixels_per_em for 300pt rendering
    let pixels_per_em = [0.147f32, 0.147f32];

    println!("  render_coord = {:?}", render_coord);
    println!("  pixels_per_em = {:?}", pixels_per_em);

    // Vertical ray casting (this is the one failing)
    println!("\n  --- Vertical ray casting ---");
    let mut ycov = 0.0f32;
    let mut ywgt = 0.0f32;

    for (i, (p1, p2, p3)) in curves.iter().enumerate() {
        let p12 = [
            p1[0] - render_coord[0],
            p1[1] - render_coord[1],
            p2[0] - render_coord[0],
            p2[1] - render_coord[1],
        ];
        let p3r = [p3[0] - render_coord[0], p3[1] - render_coord[1]];

        let max_y = p12[1].max(p12[3]).max(p3r[1]);
        println!("  curve {}: max_y={:.1}, early_exit={}", i, max_y,
            max_y * pixels_per_em[1] < -0.5);

        if max_y * pixels_per_em[1] < -0.5 {
            println!("    BREAK (early exit)");
            break;
        }

        let code = cpu_calc_root_code(p12[0], p12[2], p3r[0]);
        println!("    x-signs: ({:.1}, {:.1}, {:.1}) code=0x{:04X}",
            p12[0], p12[2], p3r[0], code);

        if code != 0 {
            let r = cpu_solve_vert_poly(p12, p3r);
            let r_scaled = [r[0] * pixels_per_em[1], r[1] * pixels_per_em[1]];
            println!("    r_raw={:?}, r_scaled={:?}", r, r_scaled);

            if (code & 1) != 0 {
                let contrib = (r_scaled[0] + 0.5).clamp(0.0, 1.0);
                ycov -= contrib;
                ywgt = ywgt.max((1.0 - r_scaled[0].abs() * 2.0).clamp(0.0, 1.0));
                println!("    root1: ycov -= {:.4} → ycov={:.4}", contrib, ycov);
            }
            if code > 1 {
                let contrib = (r_scaled[1] + 0.5).clamp(0.0, 1.0);
                ycov += contrib;
                ywgt = ywgt.max((1.0 - r_scaled[1].abs() * 2.0).clamp(0.0, 1.0));
                println!("    root2: ycov += {:.4} → ycov={:.4}", contrib, ycov);
            }
        } else {
            println!("    code=0, skip");
        }
    }

    println!("\n  RESULT: ycov={:.4}, ywgt={:.4}, abs(ycov)={:.4}", ycov, ywgt, ycov.abs());

    // Also do horizontal
    println!("\n  --- Horizontal ray casting ---");
    let mut xcov = 0.0f32;
    let mut xwgt = 0.0f32;

    for (i, (p1, p2, p3)) in curves.iter().enumerate() {
        let p12 = [
            p1[0] - render_coord[0],
            p1[1] - render_coord[1],
            p2[0] - render_coord[0],
            p2[1] - render_coord[1],
        ];
        let p3r = [p3[0] - render_coord[0], p3[1] - render_coord[1]];

        let code = cpu_calc_root_code(p12[1], p12[3], p3r[1]);
        println!("  curve {}: y-signs=({:.1}, {:.1}, {:.1}) code=0x{:04X}",
            i, p12[1], p12[3], p3r[1], code);

        if code != 0 {
            // SolveHorizPoly
            let ax = p12[0] - p12[2] * 2.0 + p3r[0];
            let ay = p12[1] - p12[3] * 2.0 + p3r[1];
            let bx = p12[0] - p12[2];
            let by = p12[1] - p12[3];

            let (t1, t2);
            if ay.abs() < 1.0 / 65536.0 {
                let rb = 0.5 / by;
                let lin = p12[1] * rb;
                t1 = lin;
                t2 = lin;
            } else {
                let ra = 1.0 / ay;
                let d = (by * by - ay * p12[1]).max(0.0).sqrt();
                t1 = (by - d) * ra;
                t2 = (by + d) * ra;
            }

            let rx = (ax * t1 - bx * 2.0) * t1 + p12[0];
            let ry = (ax * t2 - bx * 2.0) * t2 + p12[0];
            let r = [rx * pixels_per_em[0], ry * pixels_per_em[0]];
            println!("    t1={:.4}, t2={:.4}, r_scaled={:?}", t1, t2, r);

            if (code & 1) != 0 {
                let contrib = (r[0] + 0.5).clamp(0.0, 1.0);
                xcov += contrib;
                xwgt = xwgt.max((1.0 - r[0].abs() * 2.0).clamp(0.0, 1.0));
                println!("    root1: xcov += {:.4} → xcov={:.4}", contrib, xcov);
            }
            if code > 1 {
                let contrib = (r[1] + 0.5).clamp(0.0, 1.0);
                xcov -= contrib;
                xwgt = xwgt.max((1.0 - r[1].abs() * 2.0).clamp(0.0, 1.0));
                println!("    root2: xcov -= {:.4} → xcov={:.4}", contrib, xcov);
            }
        } else {
            println!("    code=0, skip");
        }
    }

    println!("\n  RESULT: xcov={:.4}, xwgt={:.4}", xcov, xwgt);

    let combined = (xcov * xwgt + ycov * ywgt).abs() / (xwgt + ywgt).max(1.0 / 65536.0);
    let fallback = xcov.abs().min(ycov.abs());
    let coverage = combined.max(fallback).clamp(0.0, 1.0);
    println!("  FINAL: combined={:.4}, fallback={:.4}, coverage={:.4}", combined, fallback, coverage);

    // Now simulate WITH band lookup to find the divergence
    println!("\n  --- Band lookup simulation ---");
    let band_transform = [0.013605442f32, 0.0070546735, -1.7414966, 2.5326278];
    let band_max = [3i32, 3];
    let band_idx_x = ((render_coord[0] * band_transform[0] + band_transform[2]) as i32).clamp(0, band_max[0]);
    let band_idx_y = ((render_coord[1] * band_transform[1] + band_transform[3]) as i32).clamp(0, band_max[1]);
    println!("  band_index = ({}, {})", band_idx_x, band_idx_y);

    // Band entries from debug output (relative texels)
    let band_texels: Vec<[u32; 4]> = vec![
        [3, 8, 0, 0],      // 0: hband 0
        [2, 11, 0, 0],     // 1: hband 1
        [2, 13, 0, 0],     // 2: hband 2
        [2, 15, 0, 0],     // 3: hband 3
        [2, 17, 0, 0],     // 4: vband 0
        [4, 19, 0, 0],     // 5: vband 1
        [4, 23, 0, 0],     // 6: vband 2
        [2, 27, 0, 0],     // 7: vband 3
        [156, 0, 0, 0],    // 8
        [158, 0, 0, 0],    // 9
        [152, 0, 0, 0],    // 10
        [156, 0, 0, 0],    // 11
        [152, 0, 0, 0],    // 12
        [156, 0, 0, 0],    // 13
        [152, 0, 0, 0],    // 14
        [156, 0, 0, 0],    // 15
        [152, 0, 0, 0],    // 16
        [152, 0, 0, 0],    // 17
        [158, 0, 0, 0],    // 18
        [152, 0, 0, 0],    // 19
        [154, 0, 0, 0],    // 20
        [156, 0, 0, 0],    // 21
        [158, 0, 0, 0],    // 22
        [152, 0, 0, 0],    // 23
        [154, 0, 0, 0],    // 24
        [156, 0, 0, 0],    // 25
        [158, 0, 0, 0],    // 26
        [154, 0, 0, 0],    // 27
        [156, 0, 0, 0],    // 28
    ];

    // Simulate vband lookup
    // Shader: vband_data = band_texels[band_max.y + 1 + band_idx_x]
    let vband_header_idx = (band_max[1] + 1 + band_idx_x) as usize;
    let vband_header = band_texels[vband_header_idx];
    println!("  vband header at relative idx {}: count={}, offset={}",
        vband_header_idx, vband_header[0], vband_header[1]);

    let curve_offset_base = 152u32; // comma's curve_offset
    println!("  vband curves:");
    for ci in 0..vband_header[0] {
        let ref_idx = vband_header[1] as usize + ci as usize;
        let curve_ref = band_texels[ref_idx];
        let curve_tex_x = curve_ref[0];
        let curve_idx = (curve_tex_x - curve_offset_base) / 2;
        println!("    ci={}: ref_idx={}, curve_tex_x={}, curve_idx={}",
            ci, ref_idx, curve_tex_x, curve_idx);
    }

    println!("========================\n");
}

fn main() {
    env_logger::init();

    // Quick sanity check: print some glyph info
    let test_chars = ['H', 'e', 'l', 'o', ',', '.'];
    for ch in test_chars {
        if let Some(gid) = char_to_glyph_id(FONT_BYTES, ch) {
            if let Some(outline) = extract_outline(FONT_BYTES, gid) {
                println!(
                    "'{}' (glyph {}): {} curves, bounds [{:.0}, {:.0}, {:.0}, {:.0}]",
                    ch,
                    gid,
                    outline.curves.len(),
                    outline.bounds[0],
                    outline.bounds[1],
                    outline.bounds[2],
                    outline.bounds[3]
                );
            }
        }
    }

    // Debug: trace comma outline extraction
    if let Some(comma_gid) = char_to_glyph_id(FONT_BYTES, ',') {
        println!("\n=== COMMA OUTLINE TRACE (glyph {}) ===", comma_gid);
        outline::extract_outline_debug(FONT_BYTES, comma_gid);
        println!("===================================\n");
    }
    if let Some(period_gid) = char_to_glyph_id(FONT_BYTES, '.') {
        println!("=== PERIOD OUTLINE TRACE (glyph {}) ===", period_gid);
        outline::extract_outline_debug(FONT_BYTES, period_gid);
        println!("=====================================\n");
    }

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App::new();
    let _ = event_loop.run_app(&mut app);
}
