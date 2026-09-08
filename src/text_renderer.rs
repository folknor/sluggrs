use crate::GlyphInstance;
use crate::glyph_cache::GlyphKey;
use crate::raster_text::{NonVectorGlyph, RasterVertex};
use crate::text_atlas::TextAtlas;
use crate::types::{
    DecorationExtents, DecorationMode, PhysicalDecoration, PrepareError, RenderError, TextArea,
};
use crate::viewport::Viewport;

use rustc_hash::FxHashMap;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, Buffer, BufferBinding, BufferDescriptor,
    BufferUsages, COPY_BUFFER_ALIGNMENT, CommandEncoder, DepthStencilState, Device,
    MultisampleState, Queue, RenderPass, RenderPipeline,
};

use crate::types::TextBounds;

type BufferPtr = *const cosmic_text::Buffer;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TextAreaCacheKey {
    buffer_ptr: BufferPtr,
    occurrence: usize,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrepareStats {
    pub direct_hits: usize,
    pub reculls: usize,
    pub misses: usize,
}

/// Cached prepared output for a single TextArea. Reusable when the text
/// content, styling, and atlas state haven't changed.
struct CachedTextArea {
    left: f32,
    top: f32,
    scroll: [f32; 2],
    scale: f32,
    bounds: TextBounds,
    default_color: cosmic_text::Color,
    /// Resolved decorations, in input order. Color changes here are paint
    /// only; spread and offset changes move `extents` and so are geometry.
    decorations: Vec<PhysicalDecoration>,
    /// Culling envelope the cached candidate set was selected against.
    extents: DecorationExtents,
    /// Per normal instance: whether it is a MONOCHROME VECTOR glyph, the only
    /// kind a decoration covers. Parallel to `instances`, in glyph order.
    ///
    /// This records glyph kind, never whether the area happens to be
    /// decorated right now: the vector is cached and survives placement
    /// changes, so a presence flag would go stale the moment an area gains a
    /// decoration without its text changing.
    ///
    /// A Ring draw supplies the fill for exactly these, so they are withheld
    /// from the normal pipeline - and the ones NOT covered (COLRv0 layers,
    /// COLRv1) must keep their position relative to them.
    decorated: Vec<bool>,
    atlas_generation: u32,
    instances: Vec<GlyphInstance>,
    border_instances: Vec<GlyphInstance>,
    distinct_keys: Vec<GlyphKey>,
    non_vector_glyphs: Vec<NonVectorGlyph>,
    /// Whether the cached candidates cover the entire area, so a later
    /// placement-only change can re-cull them without re-walking the buffer.
    complete: bool,
}

/// Per-decoration-draw paint. 32 bytes, matching `BorderParams` in the
/// border shader; the stride is padded to the device's dynamic-offset
/// alignment at upload.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BorderUniform {
    color: [f32; 4],
    spread: f32,
    /// Matches `MODE_RING` in the border shader: 0 Solid, 1 Ring.
    mode: u32,
    offset: [f32; 2],
}

impl BorderUniform {
    fn of(decoration: &PhysicalDecoration) -> Self {
        Self {
            color: decoration.color,
            spread: decoration.spread,
            mode: match decoration.mode {
                DecorationMode::Solid => 0,
                DecorationMode::Ring => 1,
            },
            offset: decoration.offset,
        }
    }
}

#[derive(Clone, Copy)]
enum DrawStream {
    Normal,
    Border,
    /// A pre-blurred shadow composited from its own texture. Carries an index
    /// into `blur_jobs` rather than an instance range.
    Shadow,
}

#[derive(Clone, Copy)]
enum DrawMode {
    Fill,
    /// Painted beneath a fill that another draw supplies; writes no depth or
    /// stencil.
    Underlay,
    /// A Ring run, whose fragment emits the fill itself and therefore keeps
    /// the caller's depth and stencil behaviour.
    RingFill,
}

struct OrderedDraw {
    stream: DrawStream,
    range: std::ops::Range<u32>,
    mode: DrawMode,
    uniform_offset: u32,
}

/// Render one area's glyph coverage into a mask, blur it, and build the
/// composite record.
///
/// The mask is rendered in MASK-LOCAL space: the params uniform gives it the
/// mask's own size as the screen, and a scroll offset shifted by the source
/// origin, so a glyph lands at the same place relative to the mask that it
/// occupies on screen. The decoration's offset is deliberately NOT applied
/// here - it moves the finished shadow at composite time, and applying it
/// twice would double it.
#[allow(clippy::too_many_arguments)]
fn encode_blur_job(
    resources: &mut crate::blur::BlurResources,
    device: &Device,
    queue: &Queue,
    encoder: &mut CommandEncoder,
    atlas: &TextAtlas,
    instances: &[GlyphInstance],
    geometry: crate::blur::BlurGeometry,
    color: [f32; 4],
    resolution: crate::types::Resolution,
    scroll: [f32; 2],
    flags: u32,
    depth: f32,
) -> crate::blur::BlurJob {
    let cache = atlas.cache();
    let (width, height) = geometry.source_size();

    let mask = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sluggrs shadow mask"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::blur::MASK_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let mask_view = mask.create_view(&wgpu::TextureViewDescriptor::default());

    let state = cache.get_or_create_mask_pipeline(device);

    // Per-job buffers, never reused: rewriting a shared buffer before submit
    // would let an already-encoded pass observe a later job's values.
    // Mask-local space. `instance.screen_rect` is UNSCROLLED and vs_border
    // adds params.scroll_offset, while `geometry.source` was measured in
    // scrolled space by instance_bounds - so the viewport scroll has to be
    // carried here too, or the mask is displaced by exactly -scroll.
    let params = crate::gpu_cache::Params {
        screen_size: [width as f32, height as f32],
        scroll_offset: [
            scroll[0] - geometry.source[0],
            scroll[1] - geometry.source[1],
        ],
        flags,
        _pad: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sluggrs shadow mask params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let params_group = cache.create_uniforms_bind_group(device, &params_buffer);

    let decoration = BorderUniform {
        color: [1.0, 1.0, 1.0, 1.0],
        spread: 0.0,
        mode: 0,
        offset: [0.0, 0.0],
    };
    let decoration_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sluggrs shadow mask decoration"),
        contents: bytemuck::bytes_of(&decoration),
        usage: BufferUsages::UNIFORM,
    });
    let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
    let stride = 32u64.next_multiple_of(alignment.max(1));
    let decoration_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("sluggrs shadow mask decoration bind group"),
        layout: &state.uniforms_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(BufferBinding {
                buffer: &decoration_buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(stride.min(decoration_buffer.size())),
            }),
        }],
    });

    // Only the instances at this job's depth: the composite draws one quad at
    // one depth, so a mask holding glyphs from another depth would place them
    // at the wrong one.
    let at_depth: Vec<GlyphInstance> = instances
        .iter()
        .filter(|instance| instance.depth.to_bits() == depth.to_bits())
        .copied()
        .collect();
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sluggrs shadow mask vertices"),
        contents: bytemuck::cast_slice(&at_depth),
        usage: BufferUsages::VERTEX,
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sluggrs shadow mask"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &mask_view,
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
        pass.set_pipeline(&state.pipeline);
        pass.set_bind_group(0, &params_group, &[]);
        pass.set_bind_group(1, atlas.bind_group(), &[]);
        pass.set_bind_group(2, &decoration_group, &[0]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.draw(0..4, 0..at_depth.len() as u32);
    }

    let (_horizontal, vertical) =
        crate::blur::encode_blur(resources, device, queue, encoder, &mask_view, geometry);

    crate::blur::finish_job(
        resources,
        device,
        queue,
        vertical,
        geometry,
        color,
        [resolution.width as f32, resolution.height as f32],
        flags,
        depth,
    )
}

/// A filtered decoration queued during instance emission, encoded once the
/// atlas storage buffer has been flushed.
struct PendingBlur {
    instances: std::ops::Range<u32>,
    geometry: crate::blur::BlurGeometry,
    color: [f32; 4],
    /// The mask is built from only the instances at this depth, and the
    /// composite is drawn at it.
    depth: f32,
}

/// The area's clip rect in physical pixels, for clipping a shadow.
fn plan_bounds(plan: &AreaPlan<'_>) -> [f32; 4] {
    let bounds = match plan {
        AreaPlan::Miss(area) => [
            area.bounds_min_x,
            area.bounds_min_y,
            area.bounds_max_x,
            area.bounds_max_y,
        ],
        AreaPlan::ReCull { bounds, .. } | AreaPlan::HitDirect { bounds, .. } => *bounds,
    };
    [
        bounds[0] as f32,
        bounds[1] as f32,
        bounds[2] as f32,
        bounds[3] as f32,
    ]
}

/// The distinct depths present among these instances, farthest first.
///
/// A composite quad carries one depth, so a blurred shadow is built once per
/// depth. Compared by bit pattern: these are copied verbatim from what the
/// caller's `metadata_to_depth` produced, never arithmetic results.
///
/// # Known limitation
///
/// Partitioning is what lets each shadow be occluded at its glyphs' own
/// depth, but it is not the same picture as blurring the union. Where two
/// partitions' shadows overlap on screen they composite source-over, giving
/// `a + b(1-a)` rather than the single blur of the combined mask that CSS
/// `text-shadow` specifies, so the overlap reads darker. One quad cannot do
/// both, and occlusion was judged the more visible of the two errors. The
/// case needs an area whose glyphs carry different metadata AND whose
/// shadows overlap; sorting farthest-first at least makes the result
/// deterministic instead of dependent on glyph order.
fn distinct_depths(instances: &[GlyphInstance]) -> Vec<f32> {
    let mut depths: Vec<f32> = Vec::new();
    for instance in instances {
        if !depths
            .iter()
            .any(|seen| seen.to_bits() == instance.depth.to_bits())
        {
            depths.push(instance.depth);
        }
    }
    depths.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    depths
}

/// Union of the rects of the instances at one depth, in scrolled screen
/// space, or `None` when none of them is at that depth.
fn instance_bounds_at_depth(
    instances: &[GlyphInstance],
    scroll: [f32; 2],
    depth: f32,
) -> Option<[f32; 4]> {
    let mut bounds: Option<[f32; 4]> = None;
    for instance in instances {
        if instance.depth.to_bits() != depth.to_bits() {
            continue;
        }
        let [x, y, w, h] = instance.screen_rect;
        // vs_border rasterises the mask half a pixel outside screen_rect on
        // every side, so the union has to include that or the destination is
        // clipped just inside the convolution's true support.
        let rect = [
            x + scroll[0] - DECORATION_AA,
            y + scroll[1] - DECORATION_AA,
            x + scroll[0] + w + DECORATION_AA,
            y + scroll[1] + h + DECORATION_AA,
        ];
        bounds = Some(match bounds {
            None => rect,
            Some(b) => [
                b[0].min(rect[0]),
                b[1].min(rect[1]),
                b[2].max(rect[2]),
                b[3].max(rect[3]),
            ],
        });
    }
    bounds
}

/// Split an area into draw runs that preserve glyph order.
///
/// `decorated[i]` says whether normal instance `i` is covered by the Ring
/// draw. Emitting all covered instances first and all uncovered ones after
/// would reorder them, which is observable wherever quads overlap - combining
/// marks, negative letter spacing, overhanging bounds, or merely overlapping
/// antialiasing fringes - and can change depth/stencil results. So adjacent
/// instances of the same kind coalesce into a run and the runs stay in order.
///
/// Yields `(decorated, normal_range, border_range)`, where `border_range`
/// indexes the decoration stream (the covered subset, same glyph order).
fn decoration_runs(decorated: &[bool]) -> Vec<(bool, std::ops::Range<u32>, std::ops::Range<u32>)> {
    let mut runs = Vec::new();
    let mut index = 0usize;
    let mut covered_seen = 0u32;
    while index < decorated.len() {
        let kind = decorated[index];
        let start = index;
        let border_start = covered_seen;
        while index < decorated.len() && decorated[index] == kind {
            if kind {
                covered_seen += 1;
            }
            index += 1;
        }
        runs.push((kind, start as u32..index as u32, border_start..covered_seen));
    }
    runs
}

/// Antialiasing allowance the border shader adds to every dilation, in
/// physical pixels. The same value drives quad dilation, the distance-grid
/// radius, the fragment coverage threshold, and the CPU culling extents;
/// they must agree or the fragment falloff is clipped at the quad edge.
pub(crate) const DECORATION_AA: f32 = 0.5;

/// The largest distance-query radius any of these decorations needs, or
/// `None` when there is nothing to draw. Offset is excluded on purpose.
fn widest_spread(decorations: &[PhysicalDecoration]) -> Option<f32> {
    decorations
        .iter()
        .map(|d| d.spread + DECORATION_AA)
        .fold(None, |acc: Option<f32>, r| {
            Some(acc.map_or(r, |a| a.max(r)))
        })
}

/// Find the border-eligible glyph key behind an emitted instance, with its
/// units-per-em. Monochrome vector glyphs only: COLRv0 layers flatten into
/// ordinary-looking instances, so eligibility is checked on the entry.
fn bordered_key_for(
    atlas: &TextAtlas,
    distinct_keys: &[GlyphKey],
    glyph_offset: u32,
) -> Option<(GlyphKey, f32)> {
    distinct_keys.iter().find_map(|key| {
        atlas.glyph(key).and_then(|entry| {
            (!entry.is_non_vector()
                && !entry.is_color_vector()
                && !entry.is_color_v1_vector()
                && entry.glyph_offset == glyph_offset)
                .then_some((*key, entry.units_per_em))
        })
    })
}

/// The two independent capacities one border blob must satisfy across
/// every use of its glyph in a frame. See the aggregation pre-pass in
/// `prepare_with_depth`.
#[derive(Clone, Copy, Default)]
struct BorderCapacity {
    /// Largest ppem the glyph is drawn at, bounding boundary accuracy.
    ppem: f32,
    /// Largest distance-query radius in FONT UNITS. A pixel radius cannot
    /// be compared across ppems, which is the whole point of this type.
    radius_units: f32,
}

impl BorderCapacity {
    /// Fold in one use of the glyph: `radius_px` at `ppem`, given the
    /// glyph's units-per-em.
    fn extend(&mut self, ppem: f32, radius_px: f32, units_per_em: f32) {
        self.ppem = self.ppem.max(ppem);
        let units = radius_px * units_per_em / ppem.max(f32::MIN_POSITIVE);
        self.radius_units = self.radius_units.max(units);
    }
}

fn instance_stream_unchanged(current: &[GlyphInstance], previous: &[GlyphInstance]) -> bool {
    bytemuck::cast_slice::<_, u8>(current) == bytemuck::cast_slice::<_, u8>(previous)
}

/// One glyph queued for instance packing in pass 3 (cache-miss areas only).
struct WorkItem<'a> {
    glyph: &'a cosmic_text::LayoutGlyph,
    line_y: f32,
    key: GlyphKey,
}

/// Per-area context for a cache-miss area. Built in pass 1, consumed in pass 3.
struct MissArea<'a> {
    cache_key: TextAreaCacheKey,
    text_area: TextArea<'a>,
    work_start: usize,
    work_end: usize,
    bounds_min_x: i32,
    bounds_min_y: i32,
    bounds_max_x: i32,
    bounds_max_y: i32,
    default_color: [f32; 4],
    all_runs_included: bool,
    decorations: Vec<PhysicalDecoration>,
    extents: DecorationExtents,
}

/// Plan record per text area produced by pass 1 and consumed by pass 3 in input order.
enum AreaPlan<'a> {
    HitDirect {
        cache_key: TextAreaCacheKey,
        decorations: Vec<PhysicalDecoration>,
        /// Carried so a blurred shadow is still clipped to its area. There is
        /// no per-area scissor - one pass draws every area - so falling back
        /// to the viewport would let a cached blur paint outside its bounds
        /// on exactly the frames that hit the cache.
        bounds: [i32; 4],
    },
    ReCull {
        cache_key: TextAreaCacheKey,
        dx: f32,
        dy: f32,
        left: f32,
        top: f32,
        bounds: [i32; 4],
        scroll: [f32; 2],
        decorations: Vec<PhysicalDecoration>,
        extents: DecorationExtents,
    },
    Miss(MissArea<'a>),
}

impl AreaPlan<'_> {
    /// The area's decorations, whatever the plan kind.
    fn decorations(&self) -> &[PhysicalDecoration] {
        match self {
            AreaPlan::HitDirect { decorations, .. } | AreaPlan::ReCull { decorations, .. } => {
                decorations
            }
            AreaPlan::Miss(area) => &area.decorations,
        }
    }
}

/// How a cached area relates to the requested placement (`left`, `top`,
/// viewport scroll). See `classify_placement`.
enum PlacementClass {
    Direct,
    ReCull,
    Miss,
}

/// A text renderer that uses the Slug algorithm to render text into an
/// existing render pass.
pub struct TextRenderer {
    vertex_buffer: Buffer,
    vertex_buffer_size: u64,
    pipeline: RenderPipeline,
    instances: Vec<GlyphInstance>,
    border_instances: Vec<GlyphInstance>,
    border_vertex_buffer: Option<Buffer>,
    border_vertex_buffer_size: u64,
    border_uniform_buffer: Option<Buffer>,
    border_uniform_bind_group: Option<BindGroup>,
    border_uniform_stride: u64,
    border_pipeline: Option<RenderPipeline>,
    /// The fill-owning variant, used for a Ring run. Keeps the caller's depth
    /// and stencil state, because that draw supplies the glyph's fill.
    ring_pipeline: Option<RenderPipeline>,
    /// Blur pipelines and sampler, created on first filtered decoration so a
    /// caller that never blurs pays nothing.
    blur: Option<crate::blur::BlurResources>,
    /// Composite pipeline for the surface format, resolved during prepare so
    /// render does not need the device.
    shadow_pipeline: Option<RenderPipeline>,
    /// This frame's blurred shadows. Rebuilt every prepare: a blurred result
    /// depends on the whole area mask, so an ordinary glyph-cache hit does
    /// not establish that a previous one is still valid.
    blur_jobs: Vec<crate::blur::BlurJob>,
    multisample: MultisampleState,
    depth_stencil: Option<DepthStencilState>,
    draws: Vec<OrderedDraw>,
    glyphs_to_render: u32,
    /// Per-TextArea retained cache, keyed by buffer pointer and occurrence.
    text_area_cache: FxHashMap<TextAreaCacheKey, CachedTextArea>,
    /// Per-frame occurrence counters for shared buffers.
    text_area_occurrences: FxHashMap<BufferPtr, usize>,
    last_prepare_stats: PrepareStats,
    /// Resolution from last frame, for cache invalidation.
    cached_resolution: crate::types::Resolution,
    /// Atlas generation at last prepare() - detects trim(reset) between prepare and render.
    prepared_atlas_generation: u32,
    /// Atlas identity recorded at construction; instance offsets cannot cross atlas instances.
    atlas_id: u64,
    // Raster fallback: per-frame instances drawn using TextAtlas's shared raster resources
    raster_instances: Vec<RasterVertex>,
    raster_vertex_buffer: Buffer,
    raster_vertex_buffer_size: u64,
    raster_glyphs_to_render: u32,
}

impl TextRenderer {
    pub fn new(
        atlas: &mut TextAtlas,
        device: &Device,
        multisample: MultisampleState,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let vertex_buffer_size = next_copy_buffer_size(4096);
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sluggrs vertices"),
            size: vertex_buffer_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = atlas.get_or_create_pipeline(device, multisample, depth_stencil.clone());

        atlas.init_raster(device, depth_stencil.clone(), multisample);

        let raster_vertex_buffer_size = 4096u64;
        let raster_vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sluggrs raster vertices"),
            size: raster_vertex_buffer_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer,
            vertex_buffer_size,
            pipeline,
            instances: Vec::new(),
            border_instances: Vec::new(),
            border_vertex_buffer: None,
            border_vertex_buffer_size: 0,
            border_uniform_buffer: None,
            border_uniform_bind_group: None,
            border_uniform_stride: 0,
            border_pipeline: None,
            ring_pipeline: None,
            blur: None,
            shadow_pipeline: None,
            blur_jobs: Vec::new(),
            multisample,
            depth_stencil,
            draws: Vec::new(),
            glyphs_to_render: 0,
            text_area_cache: FxHashMap::default(),
            text_area_occurrences: FxHashMap::default(),
            last_prepare_stats: PrepareStats::default(),
            cached_resolution: crate::types::Resolution {
                width: 0,
                height: 0,
            },
            prepared_atlas_generation: 0,
            atlas_id: atlas.id(),
            raster_instances: Vec::new(),
            raster_vertex_buffer,
            raster_vertex_buffer_size,
            raster_glyphs_to_render: 0,
        }
    }

    /// Prepare text areas for rendering, with per-glyph depth mapping.
    ///
    /// `encoder` and `cache` are unused - they exist for cryoglyph API
    /// compatibility. sluggrs uses `queue.write_texture` (no encoder needed)
    /// and extracts outlines via skrifa (no swash rasterization).
    ///
    /// Three-pass structure:
    /// 1. Classify each text area as cache-hit (direct/shifted) or miss;
    ///    for misses, walk visible runs and collect work items + distinct keys.
    /// 2. Resolve distinct missing glyph keys (extract+upload). Future-parallel.
    /// 3. Walk plans in input order; emit instances per area; populate cache.
    #[allow(clippy::too_many_arguments)]
    #[hotpath::measure]
    /// `encoder` is mutable because a filtered (blurred) decoration encodes
    /// its own render passes here: the source mask and the two blur passes.
    /// They cannot go in `render`, which runs inside the caller's already
    /// open pass, and a shared `&CommandEncoder` cannot begin a pass at all.
    ///
    /// # The caller MUST submit this encoder
    ///
    /// It carries real work now. Preparing into one encoder and then rendering
    /// into a second, submitting only the second, silently drops every mask
    /// and blur pass and leaves blurred shadows empty - the encoder was
    /// previously unused, so that pattern used to be harmless. Either use one
    /// encoder for both phases, or submit the prepare encoder first.
    pub fn prepare_with_depth<'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        font_system: &mut cosmic_text::FontSystem,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        _cache: &mut cosmic_text::SwashCache,
        mut metadata_to_depth: impl FnMut(usize) -> f32,
    ) -> Result<(), PrepareError> {
        assert_eq!(
            atlas.id(),
            self.atlas_id,
            "TextRenderer must be prepared with the TextAtlas used to construct it"
        );
        let previous_instances = std::mem::take(&mut self.instances);
        let previous_border_instances = std::mem::take(&mut self.border_instances);
        self.draws.clear();
        // Every early return below - validation, glyph resolution, border
        // capacity - leaves the instance streams emptied but the vertex
        // buffer holding the previous frame. Zero the counts up front so a
        // caller that logs the error and renders anyway draws nothing rather
        // than last frame's glyphs from a stale buffer.
        self.glyphs_to_render = 0;
        self.raster_glyphs_to_render = 0;
        self.text_area_occurrences.clear();
        let mut non_vector_collector: Vec<NonVectorGlyph> = Vec::new();

        let resolution = viewport.resolution();
        let scroll = viewport.scroll_offset();
        let atlas_gen = atlas.generation();

        if resolution != self.cached_resolution {
            self.text_area_cache.clear();
            self.cached_resolution = resolution;
        }

        // Only an all-direct-hit frame may reuse the vector vertex buffer.
        let mut all_direct_hits = true;
        let mut prepare_stats = PrepareStats::default();

        let mut plans: Vec<AreaPlan<'a>> = Vec::new();
        let mut work: Vec<WorkItem<'a>> = Vec::new();
        let mut distinct_misses: Vec<GlyphKey> = Vec::new();

        // ===== Pass 1: classify areas, collect work items =====
        for text_area in text_areas {
            // Enforced here, not left to the caller: these combinations are
            // ones the execution model cannot draw, so accepting them would
            // mean rendering something other than what was asked for.

            let buffer_ptr = text_area.buffer as BufferPtr;
            let count = self.text_area_occurrences.entry(buffer_ptr).or_default();
            let cache_key = TextAreaCacheKey {
                buffer_ptr,
                occurrence: *count,
            };
            *count += 1;

            if let Some(cached) = self.text_area_cache.get(&cache_key)
                && !text_area.buffer.redraw()
                && cached.scale == text_area.scale
                && cached.bounds == text_area.bounds
                && cached.default_color == text_area.default_color
                && cached.atlas_generation == atlas_gen
            {
                let glyphs_valid = cached
                    .distinct_keys
                    .iter()
                    .all(|k| atlas.glyph_mark_used(k).is_some());

                if glyphs_valid {
                    let dx = text_area.left - cached.left;
                    let dy = text_area.top - cached.top;
                    let decorations = text_area.physical_decorations()?;
                    let extents = DecorationExtents::of(&decorations, DECORATION_AA);
                    // Color is paint only - it rides in the uniform and never
                    // moves a candidate. Spread and offset are geometry, and
                    // they are exactly what `extents` summarizes, so the
                    // envelope is the placement-validity test.
                    let extents_match = cached.extents == extents;
                    match classify_placement(
                        dx,
                        dy,
                        cached.scroll == scroll && extents_match,
                        cached.complete,
                        !cached.non_vector_glyphs.is_empty(),
                    ) {
                        PlacementClass::Direct => {
                            prepare_stats.direct_hits += 1;
                            plans.push(AreaPlan::HitDirect {
                                cache_key,
                                decorations,
                                bounds: clipped_bounds(text_area.bounds, resolution),
                            });
                            continue;
                        }
                        PlacementClass::ReCull => {
                            all_direct_hits = false;
                            prepare_stats.reculls += 1;
                            plans.push(AreaPlan::ReCull {
                                cache_key,
                                dx,
                                dy,
                                left: text_area.left,
                                top: text_area.top,
                                bounds: clipped_bounds(text_area.bounds, resolution),
                                scroll,
                                decorations,
                                extents,
                            });
                            continue;
                        }
                        // Fall through to the miss path below.
                        PlacementClass::Miss => {}
                    }
                }
            }

            all_direct_hits = false;
            prepare_stats.misses += 1;
            let [bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y] =
                clipped_bounds(text_area.bounds, resolution);

            let default_color = color_to_f32(text_area.default_color);
            let decorations = text_area.physical_decorations()?;
            let extents = DecorationExtents::of(&decorations, DECORATION_AA);
            let work_start = work.len();

            let mut all_runs_included = true;
            let mut started_visible_range = false;
            for run in text_area.buffer.layout_runs() {
                if !run_is_visible(
                    text_area.top,
                    text_area.scale,
                    scroll[1],
                    &run,
                    bounds_min_y,
                    bounds_max_y,
                    extents,
                ) {
                    all_runs_included = false;
                    if started_visible_range {
                        break;
                    }
                    continue;
                }
                started_visible_range = true;
                let line_y = run.line_y;
                for glyph in run.glyphs {
                    let key = GlyphKey::from_layout_glyph(glyph);
                    if atlas.glyph_mark_used(&key).is_none() {
                        distinct_misses.push(key);
                    }
                    work.push(WorkItem { glyph, line_y, key });
                }
            }
            let work_end = work.len();

            plans.push(AreaPlan::Miss(MissArea {
                cache_key,
                text_area,
                work_start,
                work_end,
                bounds_min_x,
                bounds_min_y,
                bounds_max_x,
                bounds_max_y,
                default_color,
                all_runs_included,
                decorations,
                extents,
            }));
        }

        // ===== Pass 2: resolve distinct misses =====
        // Sort+dedup so each glyph is extracted+uploaded once even if it
        // appears across many text areas. Future: parallelize the extract
        // step with serial atlas commit.
        distinct_misses.sort_unstable();
        distinct_misses.dedup();
        for key in &distinct_misses {
            atlas.resolve_glyph(font_system, *key)?;
        }

        // Resolve each glyph's largest requirement for this frame before any
        // descriptor is emitted. This prevents an early instance from naming
        // a border blob superseded by a later, larger use in the same frame.
        //
        // The two capacities are tracked INDEPENDENTLY. Maximizing ppem and
        // pixel radius separately and then resolving that pair does not
        // work: the required grid radius in font units is
        // radius_px * units_per_em / ppem, so pairing the largest ppem with
        // the largest pixel radius yields the SMALLEST unit radius, and the
        // same glyph drawn at a smaller size in the same frame is
        // under-provisioned - which forced a rebuild during emission, the
        // exact supersession this pre-pass exists to prevent.
        // One blob per glyph serves EVERY decoration of every area, so the
        // requirement is the widest spread any of them asks for. Offset is
        // deliberately absent: it translates the quad, it does not change
        // which boundary is nearest a fragment in glyph space.
        let mut border_requirements: FxHashMap<GlyphKey, BorderCapacity> = FxHashMap::default();
        for plan in &plans {
            let Some(radius_px) = widest_spread(plan.decorations()) else {
                continue;
            };
            match plan {
                AreaPlan::HitDirect { cache_key, .. } | AreaPlan::ReCull { cache_key, .. } => {
                    let cached = &self.text_area_cache[cache_key];
                    for instance in &cached.instances {
                        if let Some((key, units_per_em)) =
                            bordered_key_for(atlas, &cached.distinct_keys, instance.glyph_offset)
                        {
                            border_requirements.entry(key).or_default().extend(
                                instance.ppem,
                                radius_px,
                                units_per_em,
                            );
                        }
                    }
                }
                AreaPlan::Miss(area) => {
                    for wi in &work[area.work_start..area.work_end] {
                        let entry = atlas.glyph(&wi.key).expect("glyph resolved");
                        if !entry.is_non_vector()
                            && !entry.is_color_vector()
                            && !entry.is_color_v1_vector()
                        {
                            border_requirements.entry(wi.key).or_default().extend(
                                wi.glyph.font_size * area.text_area.scale,
                                radius_px,
                                entry.units_per_em,
                            );
                        }
                    }
                }
            }
        }
        for (key, capacity) in border_requirements {
            atlas.resolve_border_glyph(key, capacity.ppem, capacity.radius_units)?;
        }

        // ===== Pass 3: emit instances per area in input order =====
        let bordered_frame = plans.iter().any(|plan| !plan.decorations().is_empty());
        let mut border_uniforms = Vec::new();
        let mut pending_blurs: Vec<PendingBlur> = Vec::new();
        self.blur_jobs.clear();
        if plans.iter().any(|plan| {
            plan.decorations()
                .iter()
                .any(PhysicalDecoration::is_filtered)
        }) {
            if self.blur.is_none() {
                self.blur = Some(crate::blur::BlurResources::new(device));
            }
            let format = atlas.format();
            let multisample = self.multisample;
            let depth_stencil = self.depth_stencil.clone();
            let resources = self.blur.as_mut().expect("just created");
            self.shadow_pipeline = Some(
                resources
                    .shadow_pipeline(device, format, multisample, depth_stencil)
                    .clone(),
            );
        }
        for plan in &plans {
            let normal_start = self.instances.len() as u32;
            let border_start = self.border_instances.len() as u32;
            let decorations = plan.decorations();
            // A Ring, if present, is the FIRST entry (validation enforces it),
            // so it is encoded last among decorations and paints on top of
            // them - and it also carries the fill for the glyphs it covers.
            let ring = decorations
                .first()
                .filter(|d| d.mode == DecorationMode::Ring)
                .copied();
            // Only a ring needs the per-instance coverage flags, to split the
            // area into order-preserving runs.
            let mut decorated_for_runs: Vec<bool> = Vec::new();
            match plan {
                AreaPlan::HitDirect { cache_key, .. } => {
                    let cached = self
                        .text_area_cache
                        .get_mut(cache_key)
                        .expect("direct-hit cache entry exists");
                    self.instances.extend_from_slice(&cached.instances);
                    // Rebuilt when decorated and CLEARED when not. Leaving a
                    // previous frame's stream behind would keep the frame on
                    // the ordered-draw path with nothing ordered to draw,
                    // and the area's text would vanish the frame its
                    // decorations were removed.
                    if decorations.is_empty() {
                        cached.border_instances.clear();
                    } else {
                        let mut refreshed = Vec::new();
                        for instance in &cached.instances {
                            if let Some((key, _)) = bordered_key_for(
                                atlas,
                                &cached.distinct_keys,
                                instance.glyph_offset,
                            ) {
                                refreshed.push(GlyphInstance {
                                    glyph_offset: atlas
                                        .border_descriptor(&key)
                                        .expect("border capacity resolved in the pre-pass"),
                                    ..*instance
                                });
                            }
                        }
                        cached.border_instances = refreshed;
                    }
                    self.border_instances
                        .extend_from_slice(&cached.border_instances);
                    non_vector_collector.extend_from_slice(&cached.non_vector_glyphs);
                    cached.decorations = decorations.to_vec();
                    if ring.is_some() {
                        decorated_for_runs = cached.decorated.clone();
                    }
                }
                AreaPlan::ReCull {
                    cache_key,
                    dx,
                    dy,
                    left,
                    top,
                    bounds,
                    scroll,
                    extents,
                    ..
                } => {
                    let cached = self
                        .text_area_cache
                        .get_mut(cache_key)
                        .expect("re-cull cache entry exists");
                    let (instances, decorated, complete) = re_cull_vector_instances(
                        &cached.instances,
                        &cached.decorated,
                        cached.complete,
                        *dx,
                        *dy,
                        *scroll,
                        *bounds,
                        *extents,
                    );
                    non_vector_collector.extend_from_slice(&cached.non_vector_glyphs);
                    self.instances.extend_from_slice(&instances);
                    let mut border_instances = Vec::new();
                    if !decorations.is_empty() {
                        for instance in &instances {
                            if let Some((key, _)) = bordered_key_for(
                                atlas,
                                &cached.distinct_keys,
                                instance.glyph_offset,
                            ) {
                                border_instances.push(GlyphInstance {
                                    glyph_offset: atlas
                                        .border_descriptor(&key)
                                        .expect("border capacity resolved in the pre-pass"),
                                    ..*instance
                                });
                            }
                        }
                    }
                    self.border_instances.extend_from_slice(&border_instances);
                    cached.left = *left;
                    cached.top = *top;
                    cached.scroll = *scroll;
                    cached.decorations = decorations.to_vec();
                    cached.extents = *extents;
                    cached.instances = instances;
                    if ring.is_some() {
                        decorated_for_runs = decorated.clone();
                    }
                    cached.decorated = decorated;
                    cached.border_instances = border_instances;
                    cached.complete = complete;
                }
                AreaPlan::Miss(area) => {
                    let mut area_instances: Vec<GlyphInstance> = Vec::new();
                    let mut area_decorated: Vec<bool> = Vec::new();
                    let mut area_non_vector: Vec<NonVectorGlyph> = Vec::new();
                    let mut area_border_instances: Vec<GlyphInstance> = Vec::new();
                    let mut area_keys: Vec<GlyphKey> = Vec::new();
                    let mut complete = area.all_runs_included;

                    let text_area = &area.text_area;
                    let bounds = [
                        area.bounds_min_x,
                        area.bounds_min_y,
                        area.bounds_max_x,
                        area.bounds_max_y,
                    ];

                    for wi in &work[area.work_start..area.work_end] {
                        let glyph = wi.glyph;
                        let entry = atlas.glyph(&wi.key).expect("miss resolved in pass 2");
                        area_keys.push(wi.key);

                        if entry.is_non_vector() {
                            let physical =
                                glyph.physical((text_area.left, text_area.top), text_area.scale);
                            let color = match glyph.color_opt {
                                Some(c) => color_to_f32(c),
                                None => area.default_color,
                            };
                            area_non_vector.push(NonVectorGlyph {
                                physical,
                                color,
                                depth: metadata_to_depth(glyph.metadata),
                                line_y_scaled_rounded: (wi.line_y * text_area.scale).round(),
                                clip_bounds: [
                                    area.bounds_min_x,
                                    area.bounds_min_y,
                                    area.bounds_max_x,
                                    area.bounds_max_y,
                                ],
                            });
                            continue;
                        }

                        if entry.is_color_v1_vector() {
                            if let Some(v1_entry) = atlas.color_v1_glyph(&wi.key) {
                                let scale =
                                    glyph.font_size * text_area.scale / v1_entry.units_per_em;
                                let glyph_x =
                                    text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                                let glyph_y =
                                    text_area.top + (wi.line_y + glyph.y_offset) * text_area.scale;
                                let [min_x, min_y, max_x, max_y] = v1_entry.bounds;
                                let screen_x = glyph_x + min_x * scale;
                                let screen_y = glyph_y - max_y * scale;
                                let screen_w = (max_x - min_x) * scale;
                                let screen_h = (max_y - min_y) * scale;

                                let screen_rect = [screen_x, screen_y, screen_w, screen_h];
                                if vector_rect_visible(
                                    screen_rect,
                                    scroll,
                                    bounds,
                                    DecorationExtents::default(),
                                ) {
                                    area_instances.push(GlyphInstance {
                                        screen_rect,
                                        color: match glyph.color_opt {
                                            Some(c) => color_to_f32(c),
                                            None => area.default_color,
                                        },
                                        glyph_offset: v1_entry.glyph_offset,
                                        cmd_texel_count: v1_entry.cmd_texel_count,
                                        depth: metadata_to_depth(glyph.metadata),
                                        ppem: glyph.font_size * text_area.scale,
                                    });
                                    // COLRv1 is never decorated.
                                    area_decorated.push(false);
                                } else {
                                    complete = false;
                                }
                            } else {
                                complete = false;
                            }
                            continue;
                        }

                        if entry.is_color_vector() {
                            if let Some(color_entry) = atlas.color_glyph(&wi.key) {
                                let foreground_color = match glyph.color_opt {
                                    Some(c) => color_to_f32(c),
                                    None => area.default_color,
                                };
                                let scale =
                                    glyph.font_size * text_area.scale / color_entry.units_per_em;
                                let glyph_x =
                                    text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                                let glyph_y =
                                    text_area.top + (wi.line_y + glyph.y_offset) * text_area.scale;
                                let depth = metadata_to_depth(glyph.metadata);
                                let ppem = glyph.font_size * text_area.scale;

                                for layer in &color_entry.layers {
                                    if layer.entry.is_non_vector() {
                                        continue;
                                    }
                                    let [min_x, min_y, max_x, max_y] = layer.entry.bounds;
                                    let screen_x = glyph_x + min_x * scale;
                                    let screen_y = glyph_y - max_y * scale;
                                    let screen_w = (max_x - min_x) * scale;
                                    let screen_h = (max_y - min_y) * scale;

                                    let screen_rect = [screen_x, screen_y, screen_w, screen_h];
                                    if !vector_rect_visible(
                                        screen_rect,
                                        scroll,
                                        bounds,
                                        DecorationExtents::default(),
                                    ) {
                                        complete = false;
                                        continue;
                                    }

                                    let color = if layer.use_foreground {
                                        foreground_color
                                    } else {
                                        layer.color
                                    };

                                    area_instances.push(GlyphInstance {
                                        screen_rect,
                                        color,
                                        glyph_offset: layer.entry.glyph_offset,
                                        cmd_texel_count: 0,
                                        depth,
                                        ppem,
                                    });
                                    // COLRv0 layers are never decorated.
                                    area_decorated.push(false);
                                }
                            } else {
                                complete = false;
                            }
                            continue;
                        }

                        let scale = glyph.font_size * text_area.scale / entry.units_per_em;
                        let [min_x, min_y, max_x, max_y] = entry.bounds;

                        let glyph_x = text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                        let glyph_y =
                            text_area.top + (wi.line_y + glyph.y_offset) * text_area.scale;

                        let screen_x = glyph_x + min_x * scale;
                        let screen_y = glyph_y - max_y * scale;
                        let screen_w = (max_x - min_x) * scale;
                        let screen_h = (max_y - min_y) * scale;

                        let screen_rect = [screen_x, screen_y, screen_w, screen_h];
                        if !vector_rect_visible(screen_rect, scroll, bounds, area.extents) {
                            complete = false;
                            continue;
                        }

                        let color = match glyph.color_opt {
                            Some(c) => color_to_f32(c),
                            None => area.default_color,
                        };

                        let fill_instance = GlyphInstance {
                            screen_rect,
                            color,
                            glyph_offset: entry.glyph_offset,
                            cmd_texel_count: 0,
                            depth: metadata_to_depth(glyph.metadata),
                            ppem: glyph.font_size * text_area.scale,
                        };
                        area_instances.push(fill_instance);
                        // Glyph KIND, not "this area currently has
                        // decorations". The flag is cached and survives a
                        // re-cull, so recording presence would leave an area
                        // that gained a Ring after being cached undecorated
                        // treating its mono glyphs as ordinary fills.
                        area_decorated.push(true);
                        if !decorations.is_empty() {
                            area_border_instances.push(GlyphInstance {
                                glyph_offset: atlas
                                    .border_descriptor(&wi.key)
                                    .expect("border capacity resolved in the pre-pass"),
                                ..fill_instance
                            });
                        }
                    }

                    area_keys.sort_unstable();
                    area_keys.dedup();

                    self.instances.extend_from_slice(&area_instances);
                    self.border_instances
                        .extend_from_slice(&area_border_instances);
                    non_vector_collector.extend_from_slice(&area_non_vector);

                    self.text_area_cache.insert(
                        area.cache_key,
                        CachedTextArea {
                            left: text_area.left,
                            top: text_area.top,
                            scale: text_area.scale,
                            bounds: text_area.bounds,
                            default_color: text_area.default_color,
                            decorations: decorations.to_vec(),
                            extents: area.extents,
                            atlas_generation: atlas_gen,
                            instances: area_instances,
                            decorated: {
                                if ring.is_some() {
                                    decorated_for_runs = area_decorated.clone();
                                }
                                area_decorated
                            },
                            border_instances: area_border_instances,
                            distinct_keys: area_keys,
                            non_vector_glyphs: area_non_vector,
                            scroll,
                            complete,
                        },
                    );
                }
            }
            let normal_end = self.instances.len() as u32;
            let border_end = self.border_instances.len() as u32;
            // Every decoration of this area draws the SAME instance range -
            // offset and dilation live in the uniform and are applied in the
            // vertex shader, so N decorations cost N uniform rebinds rather
            // than N copies of the instance stream.
            //
            // ONE traversal in reverse, emitting whichever kind each entry
            // is. The list is back-to-front like CSS text-shadow, so the
            // first entry paints on top and is encoded last. Walking the
            // filtered entries and the analytic ones as two separate groups
            // would reorder them relative to each other: [blur, solid] must
            // paint solid first, but two groups put the blur first.
            //
            // Every analytic decoration draws the SAME instance range -
            // offset and dilation live in the uniform and are applied in the
            // vertex shader - so N of them cost N uniform rebinds rather than
            // N copies of the instance stream.
            if border_end > border_start {
                let area_bounds = plan_bounds(plan);
                // Index 0 is skipped when it is the ring: that one is emitted
                // below, interleaved with the COLR runs, so mono and COLR
                // glyphs keep their relative order at the fill stage.
                let analytic_end = decorations.len();
                let analytic_start = usize::from(ring.is_some());
                for decoration in decorations[analytic_start..analytic_end].iter().rev() {
                    if decoration.is_filtered() {
                        let instances =
                            &self.border_instances[border_start as usize..border_end as usize];
                        let support = crate::types::blur_support(decoration.blur);
                        // One composite quad carries one depth, so a mask
                        // spanning glyphs at different depths would have to
                        // pick one and be occluded wrongly for the rest.
                        // Partition instead: one job per distinct depth,
                        // which is a single job for the usual case where an
                        // area's glyphs share a depth.
                        for depth in distinct_depths(instances) {
                            let Some(glyph_bounds) =
                                instance_bounds_at_depth(instances, scroll, depth)
                            else {
                                continue;
                            };
                            let Some(geometry) = crate::blur::BlurGeometry::new(
                                glyph_bounds,
                                area_bounds,
                                decoration.offset,
                                decoration.blur,
                                support,
                            ) else {
                                continue;
                            };
                            // Queued rather than encoded here: the atlas
                            // storage buffer is not flushed until after this
                            // pass, and the mask reads glyph data out of it.
                            self.draws.push(OrderedDraw {
                                stream: DrawStream::Shadow,
                                range: pending_blurs.len() as u32..pending_blurs.len() as u32 + 1,
                                mode: DrawMode::Underlay,
                                uniform_offset: 0,
                            });
                            pending_blurs.push(PendingBlur {
                                instances: border_start..border_end,
                                geometry,
                                color: decoration.color,
                                depth,
                            });
                        }
                    } else {
                        let uniform_offset = border_uniforms.len() as u32;
                        border_uniforms.push(BorderUniform::of(decoration));
                        self.draws.push(OrderedDraw {
                            stream: DrawStream::Border,
                            range: border_start..border_end,
                            mode: DrawMode::Underlay,
                            uniform_offset,
                        });
                    }
                }
            }

            match ring {
                // No ring: the fill draws as one range, exactly as before.
                None => {
                    if bordered_frame && normal_end > normal_start {
                        self.draws.push(OrderedDraw {
                            stream: DrawStream::Normal,
                            range: normal_start..normal_end,
                            mode: DrawMode::Fill,
                            uniform_offset: 0,
                        });
                    }
                }
                // A ring owns the fill of every glyph it covers, so those
                // instances must not also run the normal pipeline - a
                // zero-alpha fill would still write depth and stencil. The
                // uncovered instances (COLRv0 layers, COLRv1) still draw
                // normally, and the runs interleave to preserve glyph order.
                Some(ring) => {
                    let uniform_offset = border_uniforms.len() as u32;
                    border_uniforms.push(BorderUniform::of(&ring));
                    for (covered, normal_range, ring_range) in decoration_runs(&decorated_for_runs)
                    {
                        if covered {
                            self.draws.push(OrderedDraw {
                                stream: DrawStream::Border,
                                range: border_start + ring_range.start
                                    ..border_start + ring_range.end,
                                // This run supplies the fill, so it uses the
                                // fill-owning pipeline and keeps the caller's
                                // depth and stencil behaviour.
                                mode: DrawMode::RingFill,
                                uniform_offset,
                            });
                        } else {
                            self.draws.push(OrderedDraw {
                                stream: DrawStream::Normal,
                                range: normal_start + normal_range.start
                                    ..normal_start + normal_range.end,
                                mode: DrawMode::Fill,
                                uniform_offset: 0,
                            });
                        }
                    }
                }
            }
        }

        let occurrences = &self.text_area_occurrences;
        self.text_area_cache.retain(|key, _| {
            occurrences
                .get(&key.buffer_ptr)
                .is_some_and(|count| key.occurrence < *count)
        });

        atlas.flush_uploads(queue);

        // Only now is the atlas storage buffer complete, so only now can a
        // mask pass read glyph data out of it.
        for pending in pending_blurs {
            let resources = self.blur.as_mut().expect("blur resources created above");
            let instances = &self.border_instances
                [pending.instances.start as usize..pending.instances.end as usize];
            self.blur_jobs.push(encode_blur_job(
                resources,
                device,
                queue,
                encoder,
                atlas,
                instances,
                pending.geometry,
                pending.color,
                resolution,
                scroll,
                viewport.flags(),
                pending.depth,
            ));
        }

        let normal_order_unchanged =
            instance_stream_unchanged(&self.instances, &previous_instances);
        let border_order_unchanged =
            instance_stream_unchanged(&self.border_instances, &previous_border_instances);
        self.upload_border_resources(
            device,
            queue,
            atlas,
            &border_uniforms,
            !(all_direct_hits && border_order_unchanged),
        );
        // Stamped with the generation these entries were VALIDATED against,
        // sampled before pass 2, not with whatever the atlas reports now. The
        // two are equal today because only `trim()` bumps the generation and
        // it is unreachable during prepare, but re-reading it here would mark
        // entries holding pre-compaction glyph offsets as current and defeat
        // the RemovedFromAtlas guard the moment that changed.
        for cached in self.text_area_cache.values_mut() {
            cached.atlas_generation = atlas_gen;
        }

        self.raster_instances =
            atlas.rasterize_glyphs(queue, font_system, &non_vector_collector, scroll);
        self.raster_glyphs_to_render = self.raster_instances.len() as u32;

        // Only direct hits leave the vector vertex buffer unchanged.
        if all_direct_hits
            && normal_order_unchanged
            && self.instances.len() == self.glyphs_to_render as usize
        {
            self.upload_raster_vertices(device, queue);
            self.prepared_atlas_generation = atlas.generation();
            self.last_prepare_stats = prepare_stats;
            return Ok(());
        }

        self.upload_vertices(device, queue);
        self.upload_raster_vertices(device, queue);
        self.prepared_atlas_generation = atlas.generation();
        self.last_prepare_stats = prepare_stats;
        Ok(())
    }

    /// Upload the instance buffer to the GPU.
    fn upload_vertices(&mut self, device: &Device, queue: &Queue) {
        self.glyphs_to_render = self.instances.len() as u32;

        if self.instances.is_empty() {
            return;
        }

        let vertices_raw = bytemuck::cast_slice(&self.instances);

        if self.vertex_buffer_size >= vertices_raw.len() as u64 {
            queue.write_buffer(&self.vertex_buffer, 0, vertices_raw);
        } else {
            self.vertex_buffer.destroy();

            let new_size = next_copy_buffer_size(vertices_raw.len() as u64);
            self.vertex_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("sluggrs vertices"),
                size: new_size,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            self.vertex_buffer
                .slice(..)
                .get_mapped_range_mut()
                .slice(..vertices_raw.len())
                .copy_from_slice(vertices_raw);
            self.vertex_buffer.unmap();
            self.vertex_buffer_size = new_size;
        }
    }

    fn upload_border_resources(
        &mut self,
        device: &Device,
        queue: &Queue,
        atlas: &TextAtlas,
        uniforms: &[BorderUniform],
        upload_instances: bool,
    ) {
        if self.border_instances.is_empty() {
            return;
        }
        let raw = bytemuck::cast_slice(&self.border_instances);
        let mut recreated = false;
        if self.border_vertex_buffer_size < raw.len() as u64 {
            if let Some(buffer) = self.border_vertex_buffer.take() {
                buffer.destroy();
            }
            self.border_vertex_buffer_size = next_copy_buffer_size(raw.len() as u64);
            self.border_vertex_buffer = Some(device.create_buffer(&BufferDescriptor {
                label: Some("sluggrs border vertices"),
                size: self.border_vertex_buffer_size,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            recreated = true;
        }
        if upload_instances || recreated {
            queue.write_buffer(
                self.border_vertex_buffer
                    .as_ref()
                    .expect("border vertex buffer"),
                0,
                raw,
            );
        }

        // Two variants of the same shader. An underlay must not write depth
        // or stencil - the fill above it owns those - while a Ring's fragment
        // IS the fill and must keep the caller's state verbatim.
        let state = atlas.get_or_create_border_pipeline(
            device,
            self.multisample,
            self.depth_stencil.clone(),
            false,
        );
        self.border_pipeline = Some(state.pipeline);
        self.ring_pipeline = Some(
            atlas
                .get_or_create_border_pipeline(
                    device,
                    self.multisample,
                    self.depth_stencil.clone(),
                    true,
                )
                .pipeline,
        );
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        self.border_uniform_stride = 32u64.next_multiple_of(alignment.max(1));
        let required = self.border_uniform_stride * uniforms.len() as u64;
        let recreate = self
            .border_uniform_buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < required);
        if recreate {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("sluggrs border uniforms"),
                size: required.max(32),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("sluggrs border uniforms bind group"),
                layout: &state.uniforms_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(32),
                    }),
                }],
            });
            self.border_uniform_buffer = Some(buffer);
            self.border_uniform_bind_group = Some(bind_group);
        }
        let buffer = self
            .border_uniform_buffer
            .as_ref()
            .expect("border buffer created");
        for (index, uniform) in uniforms.iter().enumerate() {
            queue.write_buffer(
                buffer,
                self.border_uniform_stride * index as u64,
                bytemuck::bytes_of(uniform),
            );
        }
    }

    fn upload_raster_vertices(&mut self, device: &Device, queue: &Queue) {
        if self.raster_instances.is_empty() {
            self.raster_glyphs_to_render = 0;
            return;
        }

        let data = bytemuck::cast_slice(&self.raster_instances);

        if self.raster_vertex_buffer_size >= data.len() as u64 {
            queue.write_buffer(&self.raster_vertex_buffer, 0, data);
        } else {
            self.raster_vertex_buffer.destroy();
            let new_size = (data.len() as u64).next_power_of_two().max(4096);
            self.raster_vertex_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("sluggrs raster vertices"),
                size: new_size,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&self.raster_vertex_buffer, 0, data);
            self.raster_vertex_buffer_size = new_size;
        }
    }

    /// Prepares all of the provided text areas for rendering.
    #[allow(clippy::too_many_arguments)] // matches cryoglyph's API
    pub fn prepare<'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        font_system: &mut cosmic_text::FontSystem,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        cache: &mut cosmic_text::SwashCache,
    ) -> Result<(), PrepareError> {
        self.prepare_with_depth(
            device,
            queue,
            encoder,
            font_system,
            atlas,
            viewport,
            text_areas,
            cache,
            zero_depth,
        )
    }

    /// Renders all layouts that were previously provided to `prepare`.
    pub fn render(
        &self,
        atlas: &TextAtlas,
        viewport: &Viewport,
        pass: &mut RenderPass<'_>,
    ) -> Result<(), RenderError> {
        if atlas.id() != self.atlas_id {
            return Err(RenderError::RemovedFromAtlas);
        }

        // Detect trim(compaction) between prepare() and render(): the atlas was
        // recreated so our instance buffer references stale glyph offsets.
        if atlas.generation() != self.prepared_atlas_generation {
            return Err(RenderError::RemovedFromAtlas);
        }

        if self.border_instances.is_empty() && self.glyphs_to_render > 0 {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &viewport.bind_group, &[]);
            pass.set_bind_group(1, atlas.bind_group(), &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..4, 0..self.glyphs_to_render);
        } else if !self.border_instances.is_empty() {
            pass.set_bind_group(0, &viewport.bind_group, &[]);
            pass.set_bind_group(1, atlas.bind_group(), &[]);
            for draw in &self.draws {
                match (draw.stream, draw.mode) {
                    (DrawStream::Border, mode @ (DrawMode::Underlay | DrawMode::RingFill)) => {
                        let pipeline = match mode {
                            DrawMode::RingFill => {
                                self.ring_pipeline.as_ref().expect("ring pipeline")
                            }
                            _ => self.border_pipeline.as_ref().expect("border pipeline"),
                        };
                        pass.set_pipeline(pipeline);
                        let offset =
                            (u64::from(draw.uniform_offset) * self.border_uniform_stride) as u32;
                        pass.set_bind_group(
                            2,
                            self.border_uniform_bind_group
                                .as_ref()
                                .expect("border bind group"),
                            &[offset],
                        );
                        pass.set_vertex_buffer(
                            0,
                            self.border_vertex_buffer
                                .as_ref()
                                .expect("border vertex buffer")
                                .slice(..),
                        );
                    }
                    (DrawStream::Normal, DrawMode::Fill) => {
                        pass.set_pipeline(&self.pipeline);
                        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                    }
                    // A blurred shadow is already rendered into its own
                    // texture; compositing is one quad, not an instance run.
                    (DrawStream::Shadow, _) => {
                        let job = &self.blur_jobs[draw.range.start as usize];
                        let pipeline = self
                            .shadow_pipeline
                            .as_ref()
                            .expect("shadow pipeline created during prepare");
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, &job.bind_group, &[]);
                        pass.draw(0..4, 0..1);
                        // The composite borrows group 0 for its own layout;
                        // restore the shared bindings or the next glyph draw
                        // validates against the shadow's bind group.
                        pass.set_bind_group(0, &viewport.bind_group, &[]);
                        pass.set_bind_group(1, atlas.bind_group(), &[]);
                        continue;
                    }
                    _ => unreachable!("draw stream and mode agree"),
                }
                pass.draw(0..4, draw.range.clone());
            }
        }

        // Raster fallback (emoji, bitmap fonts)
        if self.raster_glyphs_to_render > 0 {
            atlas.render_raster_pass(
                viewport,
                pass,
                &self.raster_vertex_buffer,
                self.raster_glyphs_to_render,
            );
        }

        Ok(())
    }

    pub fn trim(&mut self) {
        // Raster trim is handled by TextAtlas::trim()
    }

    /// Vector instances emitted by the last `prepare()` call.
    ///
    /// This exists for integration tests that assert placement without a
    /// render pass; it is not part of the stable API.
    #[doc(hidden)]
    pub fn prepared_instances(&self) -> &[GlyphInstance] {
        &self.instances
    }

    #[doc(hidden)]
    pub fn last_prepare_stats(&self) -> PrepareStats {
        self.last_prepare_stats
    }
}

fn clipped_bounds(bounds: TextBounds, resolution: crate::types::Resolution) -> [i32; 4] {
    [
        bounds.left.max(0),
        bounds.top.max(0),
        bounds.right.min(resolution.width as i32),
        bounds.bottom.min(resolution.height as i32),
    ]
}

/// A run whose own box is outside the bounds can still be needed: its
/// decorations reach further. `extents.top` widens the top edge and
/// `extents.bottom` the bottom, independently - a downward shadow must not
/// resurrect a run above the viewport.
fn run_is_visible(
    top: f32,
    scale: f32,
    scroll_y: f32,
    run: &cosmic_text::LayoutRun,
    bounds_min_y: i32,
    bounds_max_y: i32,
    extents: DecorationExtents,
) -> bool {
    let start_y = top + run.line_top * scale + scroll_y;
    let end_y = start_y + run.line_height * scale;
    start_y - extents.top <= bounds_max_y as f32 && bounds_min_y as f32 <= end_y + extents.bottom
}

fn vector_rect_visible(
    screen_rect: [f32; 4],
    scroll: [f32; 2],
    bounds: [i32; 4],
    extents: DecorationExtents,
) -> bool {
    let [x, y, width, height] = screen_rect;
    let x = x + scroll[0];
    let y = y + scroll[1];
    // The extra pixel is the long-standing conservative slack on the fill
    // rect itself; decoration reach is added per side on top of it.
    const SLACK: f32 = 1.0;
    x + width + extents.right + SLACK >= bounds[0] as f32
        && x - extents.left - SLACK <= bounds[2] as f32
        && y + height + extents.bottom + SLACK >= bounds[1] as f32
        && y - extents.top - SLACK <= bounds[3] as f32
}

/// Classify a placement-valid cache hit. `Direct` when nothing about the
/// placement changed; `ReCull` when the cached candidate set is complete and
/// can be shifted/re-culled (raster candidates forbid a left/top shift
/// because `LayoutGlyph::physical` recomputes integer placement and subpixel
/// bins from the origin); `Miss` otherwise.
fn classify_placement(
    dx: f32,
    dy: f32,
    scroll_matches: bool,
    complete: bool,
    has_raster_candidates: bool,
) -> PlacementClass {
    if dx == 0.0 && dy == 0.0 && scroll_matches {
        return PlacementClass::Direct;
    }
    if complete && (!has_raster_candidates || (dx == 0.0 && dy == 0.0)) {
        return PlacementClass::ReCull;
    }
    PlacementClass::Miss
}

/// Re-cull cached instances after a placement change. `decorated` is filtered
/// in lockstep so it stays parallel to the surviving instances - the run
/// splitter depends on that correspondence.
#[allow(clippy::too_many_arguments)]
fn re_cull_vector_instances(
    instances: &[GlyphInstance],
    decorated: &[bool],
    complete: bool,
    dx: f32,
    dy: f32,
    scroll: [f32; 2],
    bounds: [i32; 4],
    extents: DecorationExtents,
) -> (Vec<GlyphInstance>, Vec<bool>, bool) {
    let mut visible = Vec::with_capacity(instances.len());
    let mut visible_decorated = Vec::with_capacity(instances.len());
    let mut complete = complete;
    for (index, instance) in instances.iter().enumerate() {
        let mut adjusted = *instance;
        adjusted.screen_rect[0] += dx;
        adjusted.screen_rect[1] += dy;
        if vector_rect_visible(adjusted.screen_rect, scroll, bounds, extents) {
            visible.push(adjusted);
            visible_decorated.push(decorated.get(index).copied().unwrap_or(false));
        } else {
            complete = false;
        }
    }
    (visible, visible_decorated, complete)
}

/// Convert a cosmic_text Color to normalized [f32; 4].
pub(crate) fn color_to_f32(c: cosmic_text::Color) -> [f32; 4] {
    [
        c.r() as f32 / 255.0,
        c.g() as f32 / 255.0,
        c.b() as f32 / 255.0,
        c.a() as f32 / 255.0,
    ]
}

fn next_copy_buffer_size(size: u64) -> u64 {
    let align_mask = COPY_BUFFER_ALIGNMENT - 1;
    ((size.next_power_of_two() + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT)
}

fn zero_depth(_: usize) -> f32 {
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The aggregation bug this type exists to prevent: one glyph drawn at
    /// two sizes in one frame, same border width. The required grid radius
    /// in font units is radius_px * units_per_em / ppem, so the SMALLER
    /// ppem sets the larger requirement. Maximizing ppem and pixel radius
    /// separately and pairing them picks the largest ppem, which yields the
    /// smallest unit radius and under-provisions the small-size use.
    #[test]
    fn border_capacity_takes_unit_radius_from_the_smallest_ppem() {
        let mut capacity = BorderCapacity::default();
        capacity.extend(48.0, 4.5, 1000.0);
        capacity.extend(12.0, 4.5, 1000.0);

        assert_eq!(capacity.ppem, 48.0, "boundary accuracy follows max ppem");
        assert_eq!(
            capacity.radius_units, 375.0,
            "4.5px at 12ppem needs 375 units, not the 93.75 the 48ppem use needs"
        );
    }

    #[test]
    fn border_capacity_takes_the_widest_decoration_at_each_size() {
        let mut capacity = BorderCapacity::default();
        capacity.extend(24.0, 1.0, 2048.0);
        capacity.extend(24.0, 6.0, 2048.0);
        assert_eq!(capacity.radius_units, 512.0);
    }

    #[test]
    fn order_changed_direct_hit_stream_requires_upload() {
        let instance = |x| GlyphInstance {
            screen_rect: [x, 0.0, 1.0, 1.0],
            color: [1.0; 4],
            glyph_offset: 1,
            cmd_texel_count: 0,
            depth: 0.0,
            ppem: 12.0,
        };
        let previous = [instance(1.0), instance(2.0)];
        let reordered = [instance(2.0), instance(1.0)];
        let normal_may_skip = instance_stream_unchanged(&reordered, &previous);
        let border_may_skip = instance_stream_unchanged(&reordered, &previous);
        assert!(!normal_may_skip);
        assert!(!border_may_skip);
    }

    fn run(line_top: f32, line_height: f32) -> cosmic_text::LayoutRun<'static> {
        cosmic_text::LayoutRun {
            line_i: 0,
            text: "",
            rtl: false,
            glyphs: &[],
            decorations: &[],
            line_y: 0.0,
            line_top,
            line_height,
            line_w: 0.0,
        }
    }

    fn no_extents() -> DecorationExtents {
        DecorationExtents::default()
    }

    #[test]
    fn run_visibility_applies_fractional_and_negative_scroll_at_inclusive_edges() {
        let layout_run = run(10.0, 5.0);
        let e = no_extents();
        assert!(run_is_visible(0.0, 1.0, -15.0, &layout_run, 0, 10, e));
        assert!(run_is_visible(0.5, 1.0, -10.5, &layout_run, 0, 5, e));
        assert!(!run_is_visible(0.0, 1.0, -15.1, &layout_run, 0, 10, e));
    }

    /// A run pushed off the top by scroll must be revived by an UPWARD
    /// decoration reach, not by a downward one. A single scalar margin
    /// cannot tell those apart.
    #[test]
    fn run_visibility_uses_the_side_the_decoration_actually_reaches() {
        let layout_run = run(10.0, 5.0);
        let up = DecorationExtents {
            bottom: 4.0,
            ..DecorationExtents::default()
        };
        let down = DecorationExtents {
            top: 4.0,
            ..DecorationExtents::default()
        };
        // Run sits just above the bounds: only `bottom` reach can save it.
        assert!(!run_is_visible(0.0, 1.0, -17.0, &layout_run, 0, 10, down));
        assert!(run_is_visible(0.0, 1.0, -17.0, &layout_run, 0, 10, up));
    }

    #[test]
    fn vector_culling_uses_scroll_and_one_pixel_margin() {
        let rect = [10.0, 10.0, 5.0, 5.0];
        let e = no_extents();
        assert!(vector_rect_visible(rect, [-16.0, 0.0], [0, 0, 10, 20], e));
        assert!(!vector_rect_visible(rect, [-16.1, 0.0], [0, 0, 10, 20], e));
        let right = DecorationExtents {
            right: 2.0,
            ..DecorationExtents::default()
        };
        assert!(vector_rect_visible(
            rect,
            [-18.0, 0.0],
            [0, 0, 10, 20],
            right
        ));
    }

    /// The rect is off the LEFT of the bounds, so only rightward reach can
    /// bring it back. Equal-magnitude leftward reach must not.
    #[test]
    fn vector_culling_extents_are_directional() {
        let rect = [10.0, 10.0, 5.0, 5.0];
        let bounds = [0, 0, 10, 20];
        let scroll = [-18.0, 0.0];
        let right = DecorationExtents {
            right: 2.0,
            ..DecorationExtents::default()
        };
        let left = DecorationExtents {
            left: 2.0,
            ..DecorationExtents::default()
        };
        assert!(vector_rect_visible(rect, scroll, bounds, right));
        assert!(!vector_rect_visible(rect, scroll, bounds, left));
    }

    #[test]
    fn placement_classification_covers_all_transitions() {
        use PlacementClass::{Direct, Miss, ReCull};
        // Exact placement: direct hit regardless of completeness or raster.
        assert!(matches!(
            classify_placement(0.0, 0.0, true, false, true),
            Direct
        ));
        // Scroll-only change: re-cull, even with raster candidates.
        assert!(matches!(
            classify_placement(0.0, 0.0, false, true, true),
            ReCull
        ));
        // Position change, complete, vector-only: re-cull.
        assert!(matches!(
            classify_placement(2.5, -1.0, true, true, false),
            ReCull
        ));
        // Position change with raster candidates: miss.
        assert!(matches!(
            classify_placement(0.0, 1.0, true, true, true),
            Miss
        ));
        // Incomplete cache: any placement change is a miss.
        assert!(matches!(
            classify_placement(1.0, 0.0, true, false, false),
            Miss
        ));
        assert!(matches!(
            classify_placement(0.0, 0.0, false, false, false),
            Miss
        ));
    }

    #[test]
    fn re_cull_keeps_completeness_only_when_every_vector_survives() {
        let instance = GlyphInstance {
            screen_rect: [2.0, 2.0, 2.0, 2.0],
            color: [0.0; 4],
            glyph_offset: 0,
            cmd_texel_count: 0,
            depth: 0.0,
            ppem: 0.0,
        };
        let (visible, decorated, complete) = re_cull_vector_instances(
            &[instance],
            &[true],
            true,
            1.5,
            -0.5,
            [0.0, 0.0],
            [0, 0, 10, 10],
            no_extents(),
        );
        assert!(complete);
        assert_eq!(visible[0].screen_rect[..2], [3.5, 1.5]);
        assert_eq!(decorated, vec![true], "flags follow their instances");

        let (visible, decorated, complete) = re_cull_vector_instances(
            &[instance],
            &[true],
            true,
            -10.0,
            0.0,
            [0.0, 0.0],
            [0, 0, 10, 10],
            no_extents(),
        );
        assert!(visible.is_empty());
        assert!(decorated.is_empty(), "a culled instance drops its flag");
        assert!(!complete);
    }

    /// The flags must stay parallel to the surviving instances, or the run
    /// splitter pairs a COLR glyph with a ring range.
    #[test]
    fn re_cull_keeps_flags_aligned_when_some_instances_die() {
        let at = |x: f32| GlyphInstance {
            screen_rect: [x, 2.0, 2.0, 2.0],
            color: [0.0; 4],
            glyph_offset: 0,
            cmd_texel_count: 0,
            depth: 0.0,
            ppem: 0.0,
        };
        // The middle instance is far off to the left and will be culled.
        let instances = [at(2.0), at(-500.0), at(4.0)];
        let (visible, decorated, complete) = re_cull_vector_instances(
            &instances,
            &[true, false, true],
            true,
            0.0,
            0.0,
            [0.0, 0.0],
            [0, 0, 10, 10],
            no_extents(),
        );
        assert_eq!(visible.len(), 2);
        assert_eq!(decorated, vec![true, true]);
        assert!(!complete);
    }

    /// Runs must alternate in glyph order rather than grouping all covered
    /// instances together, and the ring ranges must index the decoration
    /// stream, which holds only the covered subset.
    #[test]
    fn decoration_runs_preserve_glyph_order() {
        let runs = decoration_runs(&[true, true, false, true, false, false]);
        let shape: Vec<(bool, u32, u32, u32, u32)> = runs
            .iter()
            .map(|(covered, normal, ring)| {
                (*covered, normal.start, normal.end, ring.start, ring.end)
            })
            .collect();
        assert_eq!(
            shape,
            vec![
                // Two covered glyphs: decoration instances 0..2.
                (true, 0, 2, 0, 2),
                // One uncovered glyph, drawn between them.
                (false, 2, 3, 2, 2),
                // One more covered glyph: decoration instance 2.
                (true, 3, 4, 2, 3),
                (false, 4, 6, 3, 3),
            ]
        );
    }

    /// One composite quad carries one depth, so a mask spanning depths must
    /// be split rather than picking one and occluding the rest wrongly.
    #[test]
    fn masks_are_partitioned_by_depth() {
        let at = |depth: f32, x: f32| GlyphInstance {
            screen_rect: [x, 0.0, 4.0, 4.0],
            color: [1.0; 4],
            glyph_offset: 0,
            cmd_texel_count: 0,
            depth,
            ppem: 16.0,
        };
        let instances = [at(0.25, 0.0), at(0.75, 10.0), at(0.25, 20.0)];
        assert_eq!(
            distinct_depths(&instances),
            vec![0.75, 0.25],
            "farthest first, so the order does not depend on glyph order"
        );

        // Each partition's bounds cover only its own glyphs, grown by the
        // half pixel the mask is actually rasterised over.
        let aa = f64::from(DECORATION_AA) as f32;
        let near = instance_bounds_at_depth(&instances, [0.0, 0.0], 0.25).expect("has glyphs");
        assert_eq!(near, [-aa, -aa, 24.0 + aa, 4.0 + aa]);
        let far = instance_bounds_at_depth(&instances, [0.0, 0.0], 0.75).expect("has glyphs");
        assert_eq!(far, [10.0 - aa, -aa, 14.0 + aa, 4.0 + aa]);
    }

    #[test]
    fn a_depth_with_no_glyphs_has_no_bounds() {
        let instance = GlyphInstance {
            screen_rect: [0.0, 0.0, 4.0, 4.0],
            color: [1.0; 4],
            glyph_offset: 0,
            cmd_texel_count: 0,
            depth: 0.5,
            ppem: 16.0,
        };
        assert!(instance_bounds_at_depth(&[instance], [0.0, 0.0], 0.9).is_none());
    }

    #[test]
    fn decoration_runs_handle_uniform_and_empty_input() {
        assert!(decoration_runs(&[]).is_empty());
        let all = decoration_runs(&[true, true, true]);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].2, 0..3);
        let none = decoration_runs(&[false, false]);
        assert_eq!(none.len(), 1);
        assert_eq!(none[0].2, 0..0);
    }
}
