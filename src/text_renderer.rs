use crate::GlyphInstance;
use crate::glyph_cache::GlyphKey;
use crate::raster_text::{NonVectorGlyph, RasterVertex};
use crate::text_atlas::TextAtlas;
use crate::types::{DecorationExtents, PhysicalDecoration, PrepareError, RenderError, TextArea};
use crate::viewport::Viewport;

use rustc_hash::FxHashMap;
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
    _pad0: f32,
    offset: [f32; 2],
}

#[derive(Clone, Copy)]
enum DrawStream {
    Normal,
    Border,
}

#[derive(Clone, Copy)]
enum DrawMode {
    Fill,
    Underlay,
}

struct OrderedDraw {
    stream: DrawStream,
    range: std::ops::Range<u32>,
    mode: DrawMode,
    uniform_offset: u32,
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
    pub fn prepare_with_depth<'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        _encoder: &CommandEncoder,
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
                    let decorations = text_area.physical_decorations();
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
            let decorations = text_area.physical_decorations();
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
        for plan in &plans {
            let normal_start = self.instances.len() as u32;
            let border_start = self.border_instances.len() as u32;
            let decorations = plan.decorations();
            match plan {
                AreaPlan::HitDirect { cache_key, .. } => {
                    let cached = self
                        .text_area_cache
                        .get_mut(cache_key)
                        .expect("direct-hit cache entry exists");
                    self.instances.extend_from_slice(&cached.instances);
                    if !decorations.is_empty() {
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
                    let (instances, complete) = re_cull_vector_instances(
                        &cached.instances,
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
                    cached.border_instances = border_instances;
                    cached.complete = complete;
                }
                AreaPlan::Miss(area) => {
                    let mut area_instances: Vec<GlyphInstance> = Vec::new();
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
            // Iterated in REVERSE because the list is back-to-front like CSS
            // text-shadow: the first entry must paint on top, so it is
            // encoded last.
            if border_end > border_start {
                for decoration in decorations.iter().rev() {
                    let uniform_offset = border_uniforms.len() as u32;
                    border_uniforms.push(BorderUniform {
                        color: decoration.color,
                        spread: decoration.spread,
                        _pad0: 0.0,
                        offset: decoration.offset,
                    });
                    self.draws.push(OrderedDraw {
                        stream: DrawStream::Border,
                        range: border_start..border_end,
                        mode: DrawMode::Underlay,
                        uniform_offset,
                    });
                }
            }
            if bordered_frame && normal_end > normal_start {
                self.draws.push(OrderedDraw {
                    stream: DrawStream::Normal,
                    range: normal_start..normal_end,
                    mode: DrawMode::Fill,
                    uniform_offset: 0,
                });
            }
        }

        let occurrences = &self.text_area_occurrences;
        self.text_area_cache.retain(|key, _| {
            occurrences
                .get(&key.buffer_ptr)
                .is_some_and(|count| key.occurrence < *count)
        });

        atlas.flush_uploads(queue);

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
        let current_generation = atlas.generation();
        for cached in self.text_area_cache.values_mut() {
            cached.atlas_generation = current_generation;
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

        let state = atlas.get_or_create_border_pipeline(
            device,
            self.multisample,
            self.depth_stencil.clone(),
        );
        self.border_pipeline = Some(state.pipeline);
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
        encoder: &CommandEncoder,
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
                    (DrawStream::Border, DrawMode::Underlay) => {
                        pass.set_pipeline(self.border_pipeline.as_ref().expect("border pipeline"));
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

fn re_cull_vector_instances(
    instances: &[GlyphInstance],
    complete: bool,
    dx: f32,
    dy: f32,
    scroll: [f32; 2],
    bounds: [i32; 4],
    extents: DecorationExtents,
) -> (Vec<GlyphInstance>, bool) {
    let mut visible = Vec::with_capacity(instances.len());
    let mut complete = complete;
    for instance in instances {
        let mut adjusted = *instance;
        adjusted.screen_rect[0] += dx;
        adjusted.screen_rect[1] += dy;
        if vector_rect_visible(adjusted.screen_rect, scroll, bounds, extents) {
            visible.push(adjusted);
        } else {
            complete = false;
        }
    }
    (visible, complete)
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
        let (visible, complete) = re_cull_vector_instances(
            &[instance],
            true,
            1.5,
            -0.5,
            [0.0, 0.0],
            [0, 0, 10, 10],
            no_extents(),
        );
        assert!(complete);
        assert_eq!(visible[0].screen_rect[..2], [3.5, 1.5]);

        let (visible, complete) = re_cull_vector_instances(
            &[instance],
            true,
            -10.0,
            0.0,
            [0.0, 0.0],
            [0, 0, 10, 10],
            no_extents(),
        );
        assert!(visible.is_empty());
        assert!(!complete);
    }
}
