//! Raster fallback for non-vector glyphs (emoji, bitmap fonts).
//!
//! Shared GPU resources (pipeline, atlas texture, glyph cache) live on
//! `RasterState`, owned by `TextAtlas`. Per-frame instance data lives on
//! `TextRenderer`, which draws using the shared resources.

use std::collections::HashMap;
use std::mem;

use wgpu::{
    BindGroup, BindGroupLayout, Buffer, DepthStencilState, Device, MultisampleState, Queue,
    RenderPass, RenderPipeline, TextureFormat,
};

const SHADER_SOURCE: &str = include_str!("raster_text.wgsl");
const INITIAL_ATLAS_SIZE: u32 = 1024;

/// Per-instance vertex data for a raster glyph quad.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RasterVertex {
    pub screen_pos: [f32; 2],
    pub screen_size: [f32; 2],
    pub atlas_pos: [f32; 2],
    pub atlas_size: [f32; 2],
    pub color: [f32; 4],
    pub depth: f32,
}

/// Data collected during the vector glyph loop for non-vector glyphs.
#[derive(Clone)]
pub(crate) struct NonVectorGlyph {
    pub physical: cosmic_text::PhysicalGlyph,
    pub color: [f32; 4],
    pub depth: f32,
    pub line_y_scaled_rounded: f32,
    pub clip_bounds: [i32; 4],
}

/// Cached atlas location and placement for a rasterized glyph.
#[derive(Copy, Clone)]
struct CachedGlyph {
    atlas_x: u16,
    atlas_y: u16,
    atlas_w: u16,
    atlas_h: u16,
    placement_left: i16,
    placement_top: i16,
    is_color: bool,
}

/// Simple row-packing atlas allocator.
struct RowPacker {
    width: u32,
    height: u32,
    cursor_x: u32,
    cursor_y: u32,
    row_height: u32,
}

impl RowPacker {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
        }
    }

    fn allocate(&mut self, w: u32, h: u32) -> Option<(u16, u16)> {
        if w == 0 || h == 0 || w > self.width {
            return None;
        }
        // 1px gutter between glyphs prevents bilinear filter bleeding
        let padded_w = w + 1;
        let padded_h = h + 1;
        if self.cursor_x + padded_w > self.width {
            self.cursor_y += self.row_height;
            self.cursor_x = 0;
            self.row_height = 0;
        }
        if self.cursor_y + padded_h > self.height {
            return None;
        }
        let pos = (self.cursor_x as u16, self.cursor_y as u16);
        self.cursor_x += padded_w;
        self.row_height = self.row_height.max(padded_h);
        Some(pos)
    }

    fn reset(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.cursor_x = 0;
        self.cursor_y = 0;
        self.row_height = 0;
    }
}

/// Shared raster GPU resources, owned by `TextAtlas`.
pub(crate) struct RasterState {
    device: Device,
    pipeline: RenderPipeline,
    atlas_texture: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,
    atlas_size: u32,
    packer: RowPacker,
    glyph_cache: HashMap<cosmic_text::CacheKey, CachedGlyph>,
    frame_used: usize,
}

impl RasterState {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        uniforms_layout: &wgpu::BindGroupLayout,
        depth_stencil: Option<DepthStencilState>,
        multisample: MultisampleState,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raster text shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raster text bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("raster text pipeline layout"),
                bind_group_layouts: &[uniforms_layout, &bind_group_layout],
                immediate_size: 0,
            });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raster text pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: mem::size_of::<RasterVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 16, shader_location: 2 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 3 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 4 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 48, shader_location: 5 },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::default(),
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil,
            multisample,
            multiview_mask: None,
            cache: None,
        });

        let atlas_texture = create_atlas_texture(device, INITIAL_ATLAS_SIZE);
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = create_sampler(device);
        let bind_group = create_bind_group(device, &bind_group_layout, &atlas_view, &sampler);

        Self {
            pipeline,
            device: device.clone(),
            atlas_texture,
            atlas_view,
            bind_group_layout,
            bind_group,
            atlas_size: INITIAL_ATLAS_SIZE,
            packer: RowPacker::new(INITIAL_ATLAS_SIZE, INITIAL_ATLAS_SIZE),
            glyph_cache: HashMap::new(),
            frame_used: 0,
        }
    }

    /// Rasterize non-vector glyphs and return per-instance vertex data.
    /// The atlas texture is shared; the returned instances reference positions in it.
    pub fn rasterize_glyphs(
        &mut self,
        queue: &Queue,
        font_system: &mut cosmic_text::FontSystem,
        swash_cache: &mut cosmic_text::SwashCache,
        glyphs: &[NonVectorGlyph],
    ) -> Vec<RasterVertex> {
        self.frame_used = 0;
        let mut instances = Vec::new();

        'restart: loop {
            instances.clear();
            let mut grew = false;

            for nv in glyphs {
                let cached = if let Some(&c) = self.glyph_cache.get(&nv.physical.cache_key) {
                    c
                } else {
                    let image = match swash_cache.get_image_uncached(font_system, nv.physical.cache_key) {
                        Some(img) => img,
                        None => continue,
                    };

                    let w = image.placement.width as u32;
                    let h = image.placement.height as u32;
                    if w == 0 || h == 0 {
                        continue;
                    }

                    let (ax, ay) = match self.packer.allocate(w, h) {
                        Some(pos) => pos,
                        None => {
                            if !self.grow_atlas() {
                                log::warn!("Raster atlas exhausted, dropping glyph");
                                continue;
                            }
                            grew = true;
                            match self.packer.allocate(w, h) {
                                Some(pos) => pos,
                                None => continue,
                            }
                        }
                    };

                    let rgba_data = to_premultiplied_rgba(&image);

                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.atlas_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d { x: ax as u32, y: ay as u32, z: 0 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        &rgba_data,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(w * 4),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    );

                    let is_color = matches!(image.content, cosmic_text::SwashContent::Color);
                    let cached = CachedGlyph {
                        atlas_x: ax,
                        atlas_y: ay,
                        atlas_w: w as u16,
                        atlas_h: h as u16,
                        placement_left: image.placement.left as i16,
                        placement_top: image.placement.top as i16,
                        is_color,
                    };
                    let _ = self.glyph_cache.insert(nv.physical.cache_key, cached);
                    cached
                };

                let x = (nv.physical.x + cached.placement_left as i32) as f32;
                let y = (nv.line_y_scaled_rounded as i32
                    + nv.physical.y
                    - cached.placement_top as i32) as f32;

                // Cull against clip bounds
                let w_f = cached.atlas_w as f32;
                let h_f = cached.atlas_h as f32;
                if x + w_f < nv.clip_bounds[0] as f32
                    || x > nv.clip_bounds[2] as f32
                    || y + h_f < nv.clip_bounds[1] as f32
                    || y > nv.clip_bounds[3] as f32
                {
                    continue;
                }

                let color = if cached.is_color {
                    // Color emoji: vertex color is alpha-only tint
                    let a = nv.color[3];
                    [a, a, a, a]
                } else {
                    // Mask glyph: vertex color is premultiplied RGBA
                    let [r, g, b, a] = nv.color;
                    [r * a, g * a, b * a, a]
                };

                let atlas_f = self.atlas_size as f32;
                self.frame_used += 1;
                instances.push(RasterVertex {
                    screen_pos: [x, y],
                    screen_size: [w_f, h_f],
                    atlas_pos: [cached.atlas_x as f32 / atlas_f, cached.atlas_y as f32 / atlas_f],
                    atlas_size: [w_f / atlas_f, h_f / atlas_f],
                    color,
                    depth: nv.depth,
                });
            }

            if grew {
                continue 'restart;
            }
            break;
        }

        instances
    }

    /// Set the raster pipeline and atlas bind group, then draw from the
    /// caller's vertex buffer.
    pub fn render_pass<'a>(
        &'a self,
        viewport_bind_group: &'a BindGroup,
        pass: &mut RenderPass<'a>,
        vertex_buffer: &'a Buffer,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, viewport_bind_group, &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.draw(0..4, 0..count);
    }

    pub fn trim(&mut self) {
        let cached = self.glyph_cache.len();
        if cached > 0 && self.frame_used < cached / 4 && self.atlas_size > INITIAL_ATLAS_SIZE {
            log::debug!(
                "raster_text: trim reset ({}/{} glyphs in use, atlas {}x{})",
                self.frame_used, cached, self.atlas_size, self.atlas_size
            );
            self.atlas_size = INITIAL_ATLAS_SIZE;
            self.atlas_texture = create_atlas_texture(&self.device, INITIAL_ATLAS_SIZE);
            self.atlas_view = self.atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = create_sampler(&self.device);
            self.bind_group = create_bind_group(&self.device, &self.bind_group_layout, &self.atlas_view, &sampler);
            self.packer.reset(INITIAL_ATLAS_SIZE, INITIAL_ATLAS_SIZE);
            self.glyph_cache.clear();
        }
    }

    fn grow_atlas(&mut self) -> bool {
        let max_dim = self.device.limits().max_texture_dimension_2d;
        let new_size = (self.atlas_size * 2).min(max_dim);
        if new_size == self.atlas_size {
            log::error!("Raster atlas at device max {max_dim}, cannot grow");
            return false;
        }
        log::debug!("Growing raster atlas: {0}x{0} → {1}x{1}", self.atlas_size, new_size);
        self.atlas_size = new_size;
        self.atlas_texture = create_atlas_texture(&self.device, new_size);
        self.atlas_view = self.atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = create_sampler(&self.device);
        self.bind_group = create_bind_group(&self.device, &self.bind_group_layout, &self.atlas_view, &sampler);
        self.packer.reset(new_size, new_size);
        self.glyph_cache.clear();
        true
    }
}

fn create_atlas_texture(device: &Device, size: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("raster text atlas"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn create_sampler(device: &Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("raster text sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

fn create_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raster text bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

fn linear_to_srgb(v: u8) -> u8 {
    let linear = v as f32 / 255.0;
    let srgb = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0 + 0.5) as u8
}

fn to_premultiplied_rgba(image: &cosmic_text::SwashImage) -> Vec<u8> {
    match image.content {
        cosmic_text::SwashContent::Color => {
            let mut data = image.data.clone();
            for pixel in data.chunks_exact_mut(4) {
                let a = pixel[3] as f32 / 255.0;
                pixel[0] = (pixel[0] as f32 * a) as u8;
                pixel[1] = (pixel[1] as f32 * a) as u8;
                pixel[2] = (pixel[2] as f32 * a) as u8;
            }
            data
        }
        _ => {
            // Mask: expand to premultiplied RGBA (white * alpha).
            // RGB channels are sRGB-encoded so hardware decode recovers
            // linear coverage correctly (Rgba8UnormSrgb decodes RGB but not A).
            let mut data = Vec::with_capacity(image.data.len() * 4);
            for &alpha in &image.data {
                let srgb = linear_to_srgb(alpha);
                data.extend_from_slice(&[srgb, srgb, srgb, alpha]);
            }
            data
        }
    }
}
