//! GPU regression coverage for the quadratic solver cancellation fix.

use sluggrs::{
    GlyphInstance, SIMPLE_SHADER_WGSL,
    outline::{GlyphOutline, QuadCurve},
    prep::{PrepScratch, prepare_mono},
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    screen_size: [f32; 2],
    scroll_offset: [f32; 2],
    flags: u32,
    _pad: u32,
}

#[derive(Copy, Clone)]
enum Axis {
    Horizontal,
    Vertical,
}

impl Axis {
    fn fragment_entry(self) -> &'static str {
        match self {
            Self::Horizontal => "fs_solver_horizontal",
            Self::Vertical => "fs_solver_vertical",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Horizontal => "horizontal",
            Self::Vertical => "vertical",
        }
    }
}

fn create_test_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: true,
    }))
    .expect("Failed to find adapter - this test requires a GPU or software renderer");

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("Failed to create device")
}

/// A box whose right (resp. top) edge is a genuine midpoint-controlled
/// quadratic with an exactly-linear active-axis coordinate: control values
/// (-1000, 0, 1000), so a = q1 - 2*q2 + q3 == 0 exactly in the stored
/// quarter-integer encoding. The three other edges are p2 = p1 lines.
/// Pre-fix shaders computed `a` from render_coord-shifted values, where
/// rounding yields a = -2^-15 instead of 0 at the witness sample, the
/// quadratic branch runs, and cancellation selects root t = 2.0 instead of
/// the correct linear root t ~= 0.744002.
fn outline(axis: Axis) -> GlyphOutline {
    let curves = match axis {
        Axis::Horizontal => vec![
            QuadCurve {
                p1: [0.0, -1000.0],
                p2: [500.0, 0.0],
                p3: [0.0, 1000.0],
            },
            QuadCurve {
                p1: [0.0, 1000.0],
                p2: [0.0, 1000.0],
                p3: [-1000.0, 1000.0],
            },
            QuadCurve {
                p1: [-1000.0, 1000.0],
                p2: [-1000.0, 1000.0],
                p3: [-1000.0, -1000.0],
            },
            QuadCurve {
                p1: [-1000.0, -1000.0],
                p2: [-1000.0, -1000.0],
                p3: [0.0, -1000.0],
            },
        ],
        Axis::Vertical => vec![
            QuadCurve {
                p1: [-1000.0, 0.0],
                p2: [0.0, 500.0],
                p3: [1000.0, 0.0],
            },
            QuadCurve {
                p1: [1000.0, 0.0],
                p2: [1000.0, 0.0],
                p3: [1000.0, -1000.0],
            },
            QuadCurve {
                p1: [1000.0, -1000.0],
                p2: [1000.0, -1000.0],
                p3: [-1000.0, -1000.0],
            },
            QuadCurve {
                p1: [-1000.0, -1000.0],
                p2: [-1000.0, -1000.0],
                p3: [-1000.0, 0.0],
            },
        ],
    };
    let bounds = match axis {
        Axis::Horizontal => [-1000.0, -1000.0, 500.0, 1000.0],
        Axis::Vertical => [-1000.0, -1000.0, 1000.0, 500.0],
    };

    GlyphOutline { curves, bounds }
}

fn glyph_data(axis: Axis) -> Vec<i32> {
    let mut scratch = PrepScratch::default();
    let prepared = prepare_mono(&outline(axis), 1, 1, 2000.0, &mut scratch)
        .expect("synthetic outline must fit the packed i16 representation");
    let mut data = vec![
        prepared.bounds[0].to_bits() as i32,
        prepared.bounds[1].to_bits() as i32,
        prepared.bounds[2].to_bits() as i32,
        prepared.bounds[3].to_bits() as i32,
        prepared.band_transform[0].to_bits() as i32,
        prepared.band_transform[1].to_bits() as i32,
        prepared.band_transform[2].to_bits() as i32,
        prepared.band_transform[3].to_bits() as i32,
        sluggrs::prep::pack_i16_pair(
            (prepared.band_count_x - 1) as i16,
            (prepared.band_count_y - 1) as i16,
        ),
        0,
    ];
    data.extend_from_slice(&prepared.blob_data);
    data
}

/// Appends test-only fragment entry points that call the real
/// render_single with the exact f32 witness sample. On the witness curve
/// y(t) = -1000 + 2000t, the sample's active coordinate 488.003997802734375
/// gives t ~= 0.744002 and curve x(t) ~= 190.463; the sample x
/// 187.9630126953125 sits 2.5 decoded units = 0.25 px inside
/// (pixels_per_em 0.1), so correct coverage is exactly 0.75. The broken
/// variant's wrong root drives coverage far from 0.75 (measured 0 on this
/// geometry). The vertical entry point swaps the two coordinates.
fn shader_source(broken: bool) -> String {
    let original = "let a = q12.xy - q12.zw * 2.0 + q3;\n            let b = q12.xy - q12.zw;";
    let shifted = "let a = p12.xy - p12.zw * 2.0 + p3;\n            let b = p12.xy - p12.zw;";
    let occurrences = SIMPLE_SHADER_WGSL.matches(original).count();
    assert_eq!(
        occurrences, 2,
        "solver regression patch must match both solver blocks, found {occurrences}"
    );

    let source = if broken {
        SIMPLE_SHADER_WGSL.replace(original, shifted)
    } else {
        SIMPLE_SHADER_WGSL.to_owned()
    };
    format!(
        "{source}\n\
@fragment\n\
fn fs_solver_horizontal(input: VertexOutput) -> @location(0) vec4<f32> {{\n\
    let coverage = render_single(\n\
        vec2<f32>(187.9630126953125, 488.003997802734375),\n\
        vec2<f32>(0.1, 0.1),\n\
        input.banding,\n\
        u32(input.glyph.x),\n\
        input.glyph.yz,\n\
    );\n\
    return vec4<f32>(coverage, coverage, coverage, coverage);\n\
}}\n\
@fragment\n\
fn fs_solver_vertical(input: VertexOutput) -> @location(0) vec4<f32> {{\n\
    let coverage = render_single(\n\
        vec2<f32>(488.003997802734375, 187.9630126953125),\n\
        vec2<f32>(0.1, 0.1),\n\
        input.banding,\n\
        u32(input.glyph.x),\n\
        input.glyph.yz,\n\
    );\n\
    return vec4<f32>(coverage, coverage, coverage, coverage);\n\
}}\n"
    )
}

fn render_coverage(device: &wgpu::Device, queue: &wgpu::Queue, axis: Axis, broken: bool) -> f32 {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("solver regression shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source(broken).into()),
    });
    let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("solver regression params layout"),
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
    let glyph_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("solver regression glyph layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("solver regression pipeline layout"),
        bind_group_layouts: &[Some(&params_bgl), Some(&glyph_bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("solver regression pipeline"),
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
                        format: wgpu::VertexFormat::Uint32x2,
                        offset: 32,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 40,
                        shader_location: 3,
                    },
                ],
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(axis.fragment_entry()),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                blend: None,
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

    let params = Params {
        screen_size: [1.0, 1.0],
        scroll_offset: [0.0, 0.0],
        flags: 0,
        _pad: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("solver regression params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let glyph_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("solver regression glyph data"),
        contents: bytemuck::cast_slice(&glyph_data(axis)),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let instance = GlyphInstance {
        screen_rect: [0.0, 0.0, 1.0, 1.0],
        color: [1.0; 4],
        glyph_offset: 0,
        cmd_texel_count: 0,
        depth: 0.0,
        ppem: 200.0,
    };
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("solver regression instance"),
        contents: bytemuck::bytes_of(&instance),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("solver regression params bind group"),
        layout: &params_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buffer.as_entire_binding(),
        }],
    });
    let glyph_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("solver regression glyph bind group"),
        layout: &glyph_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: glyph_buffer.as_entire_binding(),
        }],
    });

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("solver regression target"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("solver regression readback"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("solver regression encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("solver regression pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &params_bg, &[]);
        pass.set_bind_group(1, &glyph_bg, &[]);
        pass.set_vertex_buffer(0, instance_buffer.slice(..));
        pass.draw(0..4, 0..1);
    }
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).expect("readback receiver dropped");
        });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv()
        .expect("readback sender dropped")
        .expect("readback map failed");
    let mapped = readback.slice(..).get_mapped_range();
    let coverage = f32::from(mapped[3]) / 255.0;
    drop(mapped);
    readback.unmap();
    coverage
}

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn solver_cancellation_regression() {
    let (device, queue) = create_test_device();
    let tolerance = 2.0 / 255.0;

    for axis in [Axis::Horizontal, Axis::Vertical] {
        let fixed = render_coverage(&device, &queue, axis, false);
        let broken = render_coverage(&device, &queue, axis, true);
        assert!(
            (fixed - 0.75).abs() <= tolerance,
            "{} fixed coverage {fixed}, broken coverage {broken}; expected fixed coverage near 0.75",
            axis.name(),
        );
        assert!(
            (broken - 0.75).abs() > tolerance,
            "{} fixed coverage {fixed}, broken coverage {broken}; expected broken coverage away from 0.75",
            axis.name(),
        );
    }
}
