//! Filtered (blurred) shadow jobs: mask render, separable blur, composite.
//!
//! A blurred shadow is a convolution of the area's whole glyph mask, not a
//! function of the distance to the nearest boundary, so it cannot ride the
//! analytic decoration path. The chain is:
//!
//! ```text
//! glyph coverage -> scalar linear mask
//!   -> horizontal Gaussian -> vertical Gaussian
//!   -> tint with the shadow colour -> premultiplied source-over composite
//! ```
//!
//! The blur runs on SCALAR LINEAR coverage and the colour is applied only at
//! composite. Every pixel of one decoration shares a colour, so tinting after
//! the convolution is algebraically identical to convolving premultiplied
//! colour, with less storage and no chance of blurring sRGB values.
//!
//! # Texture lifetime
//!
//! `prepare` encodes the mask and blur passes; `render` samples the result.
//! Between those two the textures must not be written by anything else, so
//! each job owns freshly created textures and nothing is recycled within or
//! across frames. Pooling them would require knowing when the submission that
//! reads them has completed - `Queue::on_submitted_work_done` or an
//! equivalent serial - because a second `prepare` before submit would
//! otherwise encode writes into storage an earlier encoded pass still reads.
//! That accounting is deliberately not attempted here: correctness first,
//! and the allocation cost is bounded by the number of blurred decorations.

use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
    BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer,
    BufferBindingType, BufferUsages, ColorTargetState, ColorWrites, CommandEncoder, Device,
    Extent3d, FilterMode, FragmentState, MultisampleState, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue, RenderPipeline,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
    TextureViewDimension, VertexState,
};

/// Single-channel linear coverage. 16-bit float rather than 8-bit unorm: the
/// mask is written once but read and rewritten by two blur passes, and
/// repeated 8-bit quantisation is visible as banding in broad faint shadows.
pub(crate) const MASK_FORMAT: TextureFormat = TextureFormat::R16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurUniform {
    texel: [f32; 2],
    direction: [f32; 2],
    sigma: f32,
    support: i32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowUniform {
    color: [f32; 4],
    rect: [f32; 4],
    screen_size: [f32; 2],
    flags: u32,
    _pad: f32,
}

/// Pipelines, layouts and the sampler shared by every blur job.
#[derive(Debug)]
pub(crate) struct BlurResources {
    blur_layout: BindGroupLayout,
    shadow_layout: BindGroupLayout,
    blur_pipeline: RenderPipeline,
    shadow_pipelines: Vec<(TextureFormat, RenderPipeline)>,
    sampler: Sampler,
    /// Held so the composite pipeline for a newly seen surface format can be
    /// created later without re-parsing the WGSL.
    shadow_module: ShaderModule,
}

impl BlurResources {
    pub fn new(device: &Device) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sluggrs blur shader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(crate::BLUR_SHADER_WGSL)),
        });
        let shadow_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sluggrs shadow composite shader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(crate::SHADOW_SHADER_WGSL)),
        });

        let entries = |uniform_size: u64| {
            [
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(uniform_size),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ]
        };

        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sluggrs blur bind group layout"),
            entries: &entries(std::mem::size_of::<BlurUniform>() as u64),
        });
        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sluggrs shadow composite bind group layout"),
            entries: &entries(std::mem::size_of::<ShadowUniform>() as u64),
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sluggrs blur pipeline layout"),
            bind_group_layouts: &[Some(&blur_layout)],
            immediate_size: 0,
        });

        let blur_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("sluggrs blur pipeline"),
            layout: Some(&blur_pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs_blur"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fs_blur"),
                targets: &[Some(ColorTargetState {
                    format: MASK_FORMAT,
                    // Each blur pass fully determines its target.
                    blend: None,
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

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("sluggrs blur sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..SamplerDescriptor::default()
        });

        Self {
            blur_layout,
            shadow_layout,
            blur_pipeline,
            shadow_pipelines: Vec::new(),
            sampler,
            shadow_module,
        }
    }

    /// The composite pipeline for a surface format, created on first use.
    pub fn shadow_pipeline(&mut self, device: &Device, format: TextureFormat) -> &RenderPipeline {
        if let Some(index) = self
            .shadow_pipelines
            .iter()
            .position(|(fmt, _)| *fmt == format)
        {
            return &self.shadow_pipelines[index].1;
        }
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sluggrs shadow composite pipeline layout"),
            bind_group_layouts: &[Some(&self.shadow_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("sluggrs shadow composite pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &self.shadow_module,
                entry_point: Some("vs_shadow"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &self.shadow_module,
                entry_point: Some("fs_shadow"),
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
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        self.shadow_pipelines.push((format, pipeline));
        &self.shadow_pipelines.last().expect("just pushed").1
    }
}

/// Source-over union blending for the coverage mask.
///
/// Overlapping glyphs must form a union, not a sum: additive blending would
/// push coverage past 1.0 wherever glyphs touch and show as a bright core in
/// the blurred result. The target has no alpha channel, so the factors use
/// `OneMinusSrc` on the colour rather than `OneMinusSrcAlpha`.
pub(crate) fn mask_blend() -> BlendState {
    let component = BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrc,
        operation: BlendOperation::Add,
    };
    BlendState {
        color: component,
        alpha: component,
    }
}

/// One blurred shadow, encoded during `prepare` and sampled during `render`.
#[derive(Debug)]
pub(crate) struct BlurJob {
    /// Bind group over the blurred result, for the composite draw. The
    /// destination rect travels inside its uniform rather than as a field,
    /// since the composite vertex shader builds the quad from it.
    pub bind_group: BindGroup,
    /// Kept alive so the encoded passes and the composite agree on storage.
    _result: wgpu::Texture,
    _uniform: Buffer,
}

/// A job's geometry, in physical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BlurGeometry {
    /// Where the finished shadow lands on screen.
    pub dest: [f32; 4],
    /// Where its source mask is sampled from: the destination pulled back by
    /// the offset and grown by the kernel support on every side, so every
    /// glyph that can contribute through the kernel is inside it.
    pub source: [f32; 4],
    pub support: f32,
    pub sigma: f32,
}

impl BlurGeometry {
    /// Compute a job's rectangles, or `None` when nothing would be visible.
    ///
    /// `glyph_bounds` is the union of the area's contributing glyph rects;
    /// `clip` is the area's bounds. The destination is the glyph bounds grown
    /// by support, translated by the offset, then clipped - the SHADOW is
    /// clipped, never the source mask, because a glyph outside the clip can
    /// still throw light into it through the kernel.
    pub fn new(
        glyph_bounds: [f32; 4],
        clip: [f32; 4],
        offset: [f32; 2],
        sigma: f32,
        support: f32,
    ) -> Option<Self> {
        let grown = [
            glyph_bounds[0] - support + offset[0],
            glyph_bounds[1] - support + offset[1],
            glyph_bounds[2] + support + offset[0],
            glyph_bounds[3] + support + offset[1],
        ];
        let dest = [
            grown[0].max(clip[0]),
            grown[1].max(clip[1]),
            grown[2].min(clip[2]),
            grown[3].min(clip[3]),
        ];
        if !(dest[2] > dest[0] && dest[3] > dest[1]) {
            return None;
        }
        // Pull back by the offset to reach the unshifted glyphs, then grow by
        // support so every contributor through the kernel is inside.
        let source = [
            dest[0] - offset[0] - support,
            dest[1] - offset[1] - support,
            dest[2] - offset[0] + support,
            dest[3] - offset[1] + support,
        ];
        Some(Self {
            dest: [dest[0], dest[1], dest[2] - dest[0], dest[3] - dest[1]],
            source: [
                source[0],
                source[1],
                source[2] - source[0],
                source[3] - source[1],
            ],
            support,
            sigma,
        })
    }

    pub fn source_size(&self) -> (u32, u32) {
        (
            (self.source[2].ceil() as u32).max(1),
            (self.source[3].ceil() as u32).max(1),
        )
    }
}

fn mask_texture(device: &Device, label: &str, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: MASK_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// Encode the two blur passes over `mask`, returning the blurred texture.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_blur(
    resources: &BlurResources,
    device: &Device,
    _queue: &Queue,
    encoder: &mut CommandEncoder,
    mask: &TextureView,
    geometry: BlurGeometry,
) -> (wgpu::Texture, wgpu::Texture) {
    let (width, height) = geometry.source_size();
    let horizontal = mask_texture(device, "sluggrs blur horizontal", width, height);
    let vertical = mask_texture(device, "sluggrs blur vertical", width, height);
    let horizontal_view = horizontal.create_view(&TextureViewDescriptor::default());
    let vertical_view = vertical.create_view(&TextureViewDescriptor::default());

    let texel = [1.0 / width as f32, 1.0 / height as f32];
    let support = geometry.support as i32;

    for (source, target, direction, label) in [
        (mask, &horizontal_view, [1.0f32, 0.0], "horizontal"),
        (&horizontal_view, &vertical_view, [0.0f32, 1.0], "vertical"),
    ] {
        // A fresh uniform buffer per pass, with its contents supplied at
        // creation rather than through a queue write: reusing one buffer and
        // rewriting it before submit would let the first encoded pass observe
        // the second's values.
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sluggrs blur uniform"),
            contents: bytemuck::bytes_of(&BlurUniform {
                texel,
                direction,
                sigma: geometry.sigma,
                support,
                _pad: [0.0; 2],
            }),
            usage: BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("sluggrs blur bind group"),
            layout: &resources.blur_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            multiview_mask: None,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&resources.blur_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..4, 0..1);
    }

    (horizontal, vertical)
}

/// Build the composite record for a finished blur.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_job(
    resources: &BlurResources,
    device: &Device,
    _queue: &Queue,
    result: wgpu::Texture,
    geometry: BlurGeometry,
    color: [f32; 4],
    screen_size: [f32; 2],
    flags: u32,
) -> BlurJob {
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sluggrs shadow uniform"),
        contents: bytemuck::bytes_of(&ShadowUniform {
            color,
            rect: geometry.dest,
            screen_size,
            flags,
            _pad: 0.0,
        }),
        usage: BufferUsages::UNIFORM,
    });
    let view = result.create_view(&TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("sluggrs shadow bind group"),
        layout: &resources.shadow_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&resources.sampler),
            },
        ],
    });
    BlurJob {
        bind_group,
        _result: result,
        _uniform: uniform,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shadow is clipped; the source mask is not. A glyph outside the
    /// clip still throws light into it through the kernel, so the source
    /// rectangle must reach past the destination by the full support.
    #[test]
    fn source_reaches_past_the_clipped_destination() {
        let geometry = BlurGeometry::new(
            [100.0, 100.0, 200.0, 140.0],
            [0.0, 0.0, 500.0, 500.0],
            [0.0, 0.0],
            2.0,
            8.0,
        )
        .expect("visible");
        // Destination is the glyph bounds grown by support on every side.
        assert_eq!(geometry.dest, [92.0, 92.0, 116.0, 56.0]);
        // Source grows by support again, since taps reach that far out.
        assert_eq!(geometry.source, [84.0, 84.0, 132.0, 72.0]);
    }

    /// The offset moves the destination but must be undone to find the
    /// source: the glyphs themselves never moved.
    #[test]
    fn offset_moves_destination_and_is_undone_for_source() {
        let geometry = BlurGeometry::new(
            [100.0, 100.0, 200.0, 140.0],
            [0.0, 0.0, 500.0, 500.0],
            [10.0, 5.0],
            1.0,
            4.0,
        )
        .expect("visible");
        assert_eq!(geometry.dest[0], 106.0, "96 shifted right by 10");
        assert_eq!(geometry.dest[1], 101.0, "96 shifted down by 5");
        // Undoing the offset returns to the glyph-space origin, minus support.
        assert_eq!(geometry.source[0], 92.0);
        assert_eq!(geometry.source[1], 92.0);
    }

    #[test]
    fn a_shadow_clipped_entirely_away_produces_no_job() {
        assert!(
            BlurGeometry::new(
                [100.0, 100.0, 200.0, 140.0],
                [0.0, 0.0, 50.0, 50.0],
                [0.0, 0.0],
                2.0,
                8.0,
            )
            .is_none()
        );
    }

    /// Clipping the destination must not shrink the source below what the
    /// kernel needs, or contributors just outside the clip would be lost.
    #[test]
    fn clipping_the_destination_keeps_the_source_padded() {
        let geometry = BlurGeometry::new(
            [0.0, 0.0, 200.0, 200.0],
            [50.0, 50.0, 150.0, 150.0],
            [0.0, 0.0],
            2.0,
            8.0,
        )
        .expect("visible");
        assert_eq!(geometry.dest, [50.0, 50.0, 100.0, 100.0]);
        assert_eq!(geometry.source, [42.0, 42.0, 116.0, 116.0]);
    }
}
