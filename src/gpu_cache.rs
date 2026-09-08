use std::mem;
use std::num::NonZeroU64;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
    BindingType, BlendState, Buffer, BufferBindingType, ColorTargetState, ColorWrites,
    DepthStencilState, Device, FragmentState, MultisampleState, PipelineCompilationOptions,
    PipelineLayout, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderPipeline,
    RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    TextureFormat, VertexFormat, VertexState,
};

use crate::GlyphInstance;

/// Shared GPU state for Slug text rendering.
///
/// Holds the shader module, bind group layouts, pipeline layout, and cached
/// render pipelines. Shared across all `TextAtlas` instances.
#[derive(Debug, Clone)]
pub struct Cache(Arc<Inner>);

#[derive(Debug)]
struct Inner {
    shader: ShaderModule,
    vertex_buffers: [wgpu::VertexBufferLayout<'static>; 1],
    pub(crate) atlas_layout: BindGroupLayout,
    pub(crate) uniforms_layout: BindGroupLayout,
    pipeline_layout: PipelineLayout,
    #[allow(clippy::type_complexity)]
    pipelines: Mutex<
        Vec<(
            TextureFormat,
            MultisampleState,
            Option<DepthStencilState>,
            RenderPipeline,
        )>,
    >,
    #[allow(clippy::type_complexity)]
    border: Mutex<
        Vec<(
            TextureFormat,
            MultisampleState,
            Option<DepthStencilState>,
            // Whether this variant's fragment supplies the fill (a Ring) or
            // only an underlay. They need different depth/stencil state.
            bool,
            BorderPipelineState,
        )>,
    >,
    /// The mask pipeline has one fixed target format, so it needs no key.
    mask: Mutex<Option<BorderPipelineState>>,
    /// The decoration uniform's layout, created once and shared by every
    /// border-shader pipeline variant. One bind group is built from it and
    /// bound under both the underlay and the fill-owning pipeline, so they
    /// must be the SAME layout rather than two structurally identical ones
    /// that happen to be interned as equivalent.
    border_uniforms_layout: BindGroupLayout,
}

#[derive(Debug, Clone)]
pub(crate) struct BorderPipelineState {
    pub pipeline: RenderPipeline,
    pub uniforms_layout: BindGroupLayout,
}

impl Cache {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sluggrs shader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(crate::SIMPLE_SHADER_WGSL)),
        });

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<GlyphInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // screen_rect: vec4<f32>
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                // color: vec4<f32>
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 1,
                },
                // glyph_offset, cmd_texel_count
                wgpu::VertexAttribute {
                    format: VertexFormat::Uint32x2,
                    offset: 32,
                    shader_location: 2,
                },
                // depth, ppem
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x2,
                    offset: 40,
                    shader_location: 3,
                },
            ],
        };

        // Bind group 0: unified glyph storage buffer
        let atlas_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sluggrs atlas bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                // Requires wgpu::DownlevelFlags::VERTEX_STORAGE. Baseline
                // WebGPU supports it; GLES-style adapters without vertex
                // storage are unsupported.
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Bind group 1: screen resolution uniform
        let uniforms_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sluggrs uniforms bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(mem::size_of::<Params>() as u64),
                },
                count: None,
            }],
        });

        // Shader layout: group 0 = params uniform, group 1 = textures
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sluggrs pipeline layout"),
            bind_group_layouts: &[Some(&uniforms_layout), Some(&atlas_layout)],
            immediate_size: 0,
        });

        Self(Arc::new(Inner {
            shader,
            vertex_buffers: [vertex_buffer_layout],
            atlas_layout,
            uniforms_layout,
            pipeline_layout,
            pipelines: Mutex::new(Vec::new()),
            border: Mutex::new(Vec::new()),
            mask: Mutex::new(None),
            border_uniforms_layout: device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("sluggrs border uniforms bind group layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: NonZeroU64::new(32),
                        },
                        count: None,
                    }],
                },
            ),
        }))
    }

    pub(crate) fn create_atlas_bind_group(
        &self,
        device: &Device,
        buffer: &wgpu::Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("sluggrs atlas bind group"),
            layout: &self.0.atlas_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }

    /// The bind group layout for the viewport/params uniform buffer.
    /// Shared with the raster fallback pipeline so both can use the same viewport.
    pub fn uniforms_layout(&self) -> &BindGroupLayout {
        &self.0.uniforms_layout
    }

    pub(crate) fn create_uniforms_bind_group(&self, device: &Device, buffer: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("sluggrs uniforms bind group"),
            layout: &self.0.uniforms_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }

    pub(crate) fn get_or_create_pipeline(
        &self,
        device: &Device,
        format: TextureFormat,
        multisample: MultisampleState,
        depth_stencil: Option<DepthStencilState>,
    ) -> RenderPipeline {
        let Inner {
            pipelines,
            pipeline_layout,
            shader,
            vertex_buffers,
            ..
        } = self.0.deref();

        let mut cache = pipelines.lock().expect("Write pipeline cache");

        cache
            .iter()
            .find(|(fmt, ms, ds, _)| fmt == &format && ms == &multisample && ds == &depth_stencil)
            .map(|(_, _, _, p)| p.clone())
            .unwrap_or_else(|| {
                let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
                    label: Some("sluggrs pipeline"),
                    layout: Some(pipeline_layout),
                    vertex: VertexState {
                        module: shader,
                        entry_point: Some("vs_main"),
                        buffers: vertex_buffers,
                        compilation_options: PipelineCompilationOptions::default(),
                    },
                    fragment: Some(FragmentState {
                        module: shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(ColorTargetState {
                            format,
                            blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                            write_mask: ColorWrites::default(),
                        })],
                        compilation_options: PipelineCompilationOptions::default(),
                    }),
                    primitive: PrimitiveState {
                        topology: PrimitiveTopology::TriangleStrip,
                        ..PrimitiveState::default()
                    },
                    depth_stencil: depth_stencil.clone(),
                    multisample,
                    multiview_mask: None,
                    cache: None,
                });

                cache.push((format, multisample, depth_stencil, pipeline.clone()));
                pipeline
            })
    }

    /// The border-shader pipeline.
    ///
    /// `fill_owning` selects between the two roles this shader has, which need
    /// DIFFERENT depth and stencil behaviour and therefore cannot share one
    /// pipeline:
    ///
    /// - `false` - an analytic underlay drawn beneath a fill that some other
    ///   draw will supply. Depth-tested, but writes neither depth nor stencil,
    ///   because the fill above it is what owns those.
    /// - `true` - a Ring, whose fragment emits the fill itself. It must keep
    ///   the caller's original state unchanged, or a glyph's fill would stop
    ///   writing depth and stencil purely because it was decorated, while the
    ///   COLR glyphs beside it still did.
    pub(crate) fn get_or_create_border_pipeline(
        &self,
        device: &Device,
        format: TextureFormat,
        multisample: MultisampleState,
        depth_stencil: Option<DepthStencilState>,
        fill_owning: bool,
    ) -> BorderPipelineState {
        let mut cached = self.0.border.lock().expect("Write border pipeline cache");
        if let Some((_, _, _, _, state)) = cached.iter().find(|(fmt, ms, ds, owning, _)| {
            fmt == &format && ms == &multisample && ds == &depth_stencil && *owning == fill_owning
        }) {
            return state.clone();
        }
        // Shared across both variants: one bind group is built from this and
        // bound under either pipeline.
        let border_uniforms_layout = &self.0.border_uniforms_layout;
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sluggrs border pipeline layout"),
            bind_group_layouts: &[
                Some(&self.0.uniforms_layout),
                Some(&self.0.atlas_layout),
                Some(border_uniforms_layout),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sluggrs border shader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(crate::BORDER_SHADER_WGSL)),
        });
        // A fill-owning Ring keeps the caller's state verbatim; only an
        // underlay strips writes.
        let mut underlay_depth = depth_stencil.clone();
        if let Some(state) = underlay_depth.as_mut().filter(|_| !fill_owning) {
            state.depth_write_enabled = Some(false);
            state.stencil.front.fail_op = wgpu::StencilOperation::Keep;
            state.stencil.front.depth_fail_op = wgpu::StencilOperation::Keep;
            state.stencil.front.pass_op = wgpu::StencilOperation::Keep;
            state.stencil.back.fail_op = wgpu::StencilOperation::Keep;
            state.stencil.back.depth_fail_op = wgpu::StencilOperation::Keep;
            state.stencil.back.pass_op = wgpu::StencilOperation::Keep;
            state.stencil.write_mask = 0;
        }
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("sluggrs border pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_border"),
                buffers: &self.0.vertex_buffers,
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_border"),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::default(),
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..PrimitiveState::default()
            },
            depth_stencil: underlay_depth,
            multisample,
            multiview_mask: None,
            cache: None,
        });
        let state = BorderPipelineState {
            pipeline,
            uniforms_layout: border_uniforms_layout.clone(),
        };
        cached.push((
            format,
            multisample,
            depth_stencil,
            fill_owning,
            state.clone(),
        ));
        state
    }
}

impl Cache {
    /// Pipeline that renders glyph coverage into a filtered shadow's source
    /// mask: the border vertex shader with a coverage-only fragment, drawn
    /// into a single-channel linear target.
    ///
    /// Blending is source-over union rather than additive, so glyphs that
    /// overlap do not push coverage past 1.0 and show as a bright core once
    /// blurred.
    pub(crate) fn get_or_create_mask_pipeline(&self, device: &Device) -> BorderPipelineState {
        let mut cached = self.0.mask.lock().expect("Write mask pipeline cache");
        if let Some(state) = cached.as_ref() {
            return state.clone();
        }
        let mask_uniforms_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sluggrs mask uniforms bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(32),
                    },
                    count: None,
                }],
            });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sluggrs mask pipeline layout"),
            bind_group_layouts: &[
                Some(&self.0.uniforms_layout),
                Some(&self.0.atlas_layout),
                Some(&mask_uniforms_layout),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sluggrs mask shader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(crate::BORDER_SHADER_WGSL)),
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("sluggrs mask pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_border"),
                buffers: &self.0.vertex_buffers,
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_mask"),
                targets: &[Some(ColorTargetState {
                    format: crate::blur::MASK_FORMAT,
                    blend: Some(crate::blur::mask_blend()),
                    write_mask: ColorWrites::RED,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let state = BorderPipelineState {
            pipeline,
            uniforms_layout: mask_uniforms_layout,
        };
        *cached = Some(state.clone());
        state
    }
}

/// Uniform params matching the shader's Params struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Params {
    pub screen_size: [f32; 2],
    pub scroll_offset: [f32; 2],
    pub flags: u32, // bit 0: enable MSAA+stem darkening
    pub _pad: u32,
}
