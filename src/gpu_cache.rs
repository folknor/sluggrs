use std::mem;
use std::num::NonZeroU64;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
    BindingResource, BindingType, BlendState, Buffer, BufferBindingType, ColorTargetState,
    ColorWrites, DepthStencilState, Device, FragmentState, MultisampleState,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, PrimitiveState,
    PrimitiveTopology, RenderPipeline, RenderPipelineDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, TextureFormat, TextureSampleType,
    TextureViewDimension, VertexFormat, VertexState,
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
                // em_rect: vec4<f32>
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 1,
                },
                // band_transform: vec4<f32>
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 2,
                },
                // glyph_data: vec4<u32>
                wgpu::VertexAttribute {
                    format: VertexFormat::Uint32x4,
                    offset: 48,
                    shader_location: 3,
                },
                // color: vec4<f32>
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 64,
                    shader_location: 4,
                },
                // depth: f32
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32,
                    offset: 80,
                    shader_location: 5,
                },
            ],
        };

        // Bind group 0: curve texture (Rgba32Float) + band texture (Rgba32Uint)
        let atlas_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sluggrs atlas bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
                        sample_type: TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
                        sample_type: TextureSampleType::Uint,
                    },
                    count: None,
                },
            ],
        });

        // Bind group 1: screen resolution uniform
        let uniforms_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sluggrs uniforms bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
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
            bind_group_layouts: &[&uniforms_layout, &atlas_layout],
            immediate_size: 0,
        });

        Self(Arc::new(Inner {
            shader,
            vertex_buffers: [vertex_buffer_layout],
            atlas_layout,
            uniforms_layout,
            pipeline_layout,
            pipelines: Mutex::new(Vec::new()),
        }))
    }

    pub(crate) fn create_atlas_bind_group(
        &self,
        device: &Device,
        curve_view: &wgpu::TextureView,
        band_view: &wgpu::TextureView,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("sluggrs atlas bind group"),
            layout: &self.0.atlas_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(curve_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(band_view),
                },
            ],
        })
    }

    pub(crate) fn create_uniforms_bind_group(
        &self,
        device: &Device,
        buffer: &Buffer,
    ) -> BindGroup {
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
}

/// Uniform params matching the shader's Params struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Params {
    pub screen_size: [f32; 2],
    pub scroll_offset: [f32; 2],
}
