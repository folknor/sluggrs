use crate::band::{BandScratch, CurveLocation};
use crate::blob_cache::{
    BlobCacheStats, BlobKind, CachedBlob, CachedBorderBlob, CachedColorLayer, GlyphBlobCache,
    ResidentBlob, ResidentBorderBlob,
};
use crate::glyph_cache::{
    COLOR_V1_VECTOR_GLYPH, COLOR_VECTOR_GLYPH, ColorGlyphEntry, ColorGlyphLayer, ColorV1GlyphEntry,
    GlyphEntry, GlyphKey, GlyphMap,
};
use crate::gpu_cache::Cache;
use crate::outline::{ColorGlyphInfo, extract_color_info, extract_outline};
use crate::prep::{PrepScratch, prepare_mono};
use crate::prepare::apply_italic_shear;
use crate::raster_text::{NonVectorGlyph, RasterState, RasterVertex};
use crate::types::ColorMode;
use crate::viewport::Viewport;
use rustc_hash::FxHashMap;
use skrifa::setting::VariationSetting;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use wgpu::{
    BindGroup, Buffer, DepthStencilState, Device, MultisampleState, Queue, RenderPass,
    RenderPipeline, TextureFormat,
};

/// An atlas containing cached glyph curve and band data for GPU rendering.
/// Initial buffer capacity in packed texels (8 bytes each).
const INITIAL_BUFFER_CAPACITY: u32 = 131_072;
/// A grown atlas is eligible for trimming once it reaches this multiple of its
/// initial allocation.
const TRIM_GROWTH_FACTOR: u32 = 4;
const GLYPH_HEADER_TEXELS: u32 = 5;
const BLOB_CACHE_BUDGET: usize = 4 * 1024 * 1024;
static NEXT_ATLAS_ID: AtomicU64 = AtomicU64::new(0);

/// Cached per-font data to avoid re-parsing font tables on every glyph miss.
struct CachedFont {
    font: Arc<cosmic_text::Font>,
    face_index: u32,
    units_per_em: f32,
    has_colr: bool,
}

pub struct TextAtlas {
    cache: Cache,
    device: Device,
    glyph_buffer: wgpu::Buffer,
    bind_group: BindGroup,
    format: TextureFormat,
    id: u64,

    // Buffer state - packed layout: each logical texel (4 i16 values) is stored
    // as 2 i32 elements (each i32 packs a pair of i16 values). All capacity/
    // cursor/offsets are in texel units; physical buffer is 2x in i32 units.
    initial_buffer_capacity: u32, // in texels, clamped to the device limit
    max_buffer_capacity: u32,     // in texels
    buffer_capacity: u32,         // in texels
    buffer_cursor: u32,           // append cursor in texels
    buffer_data: Vec<i32>,        // CPU-side copy (2 i32 per texel)

    // Scratch buffers retained for upload_color_v1's serial sub-glyph builder.
    scratch_band_entries: Vec<i16>,
    band_scratch: BandScratch,
    /// How much of buffer_data is already on the GPU. A grow resets this to 0.
    gpu_flush_cursor: u32,

    // Glyph cache
    glyphs: GlyphMap,
    /// COLRv0 color glyph layers, keyed by the same GlyphKey as the main map.
    color_glyphs: FxHashMap<GlyphKey, ColorGlyphEntry>,
    /// COLRv1 color glyph command sequences.
    color_v1_glyphs: FxHashMap<GlyphKey, ColorV1GlyphEntry>,
    resident_blobs: FxHashMap<GlyphKey, ResidentBlob>,
    /// Superseded supplemental spans. They remain immutable until the next
    /// compaction so already-prepared renderers keep valid descriptors.
    reclaimable_border_spans: Vec<(u32, u32)>,
    blob_cache: GlyphBlobCache,
    /// Monotonic counter incremented on atlas compaction. Used by TextRenderer's
    /// retained cache to detect when cached glyph offsets are invalidated.
    generation: u32,

    // Raster fallback for non-vector glyphs (emoji, bitmap fonts)
    raster: Option<RasterState>,
    swash_cache: cosmic_text::SwashCache,
    font_cache: FxHashMap<(cosmic_text::fontdb::ID, cosmic_text::Weight), CachedFont>,
    /// One scratch set while resolution is serial; future rayon work uses one per worker.
    prep_scratch: PrepScratch,
}

impl TextAtlas {
    pub fn new(device: &Device, queue: &Queue, cache: &Cache, format: TextureFormat) -> Self {
        Self::with_color_mode(device, queue, cache, format, ColorMode::Accurate)
    }

    pub fn with_color_mode(
        device: &Device,
        _queue: &Queue,
        cache: &Cache,
        format: TextureFormat,
        _color_mode: ColorMode,
    ) -> Self {
        Self::with_initial_buffer_capacity(
            device,
            cache,
            format,
            _color_mode,
            INITIAL_BUFFER_CAPACITY,
        )
    }

    /// Construct an atlas with a caller-selected initial storage capacity.
    ///
    /// This exists for lifecycle tests that need to exercise growth without
    /// allocating a production-sized atlas.
    #[doc(hidden)]
    pub fn with_initial_buffer_capacity(
        device: &Device,
        cache: &Cache,
        format: TextureFormat,
        _color_mode: ColorMode,
        initial_buffer_capacity: u32,
    ) -> Self {
        let max_buffer_capacity = max_buffer_capacity(device);
        // Zero would create a storage binding that fails wgpu validation and
        // a zero trim threshold; clamp to at least one texel.
        let initial_buffer_capacity = initial_buffer_capacity.clamp(1, max_buffer_capacity);
        let glyph_buffer = create_glyph_buffer(device, initial_buffer_capacity);
        let bind_group = cache.create_atlas_bind_group(device, &glyph_buffer);

        Self {
            cache: cache.clone(),
            device: device.clone(),
            glyph_buffer,
            bind_group,
            format,
            id: NEXT_ATLAS_ID.fetch_add(1, Ordering::Relaxed),
            initial_buffer_capacity,
            max_buffer_capacity,
            buffer_capacity: initial_buffer_capacity,
            buffer_cursor: 0,
            buffer_data: Vec::new(),
            scratch_band_entries: Vec::new(),
            band_scratch: BandScratch::default(),
            gpu_flush_cursor: 0,
            generation: 0,
            glyphs: GlyphMap::new(),
            color_glyphs: FxHashMap::default(),
            color_v1_glyphs: FxHashMap::default(),
            resident_blobs: FxHashMap::default(),
            reclaimable_border_spans: Vec::new(),
            blob_cache: GlyphBlobCache::new(BLOB_CACHE_BUDGET),
            raster: None,
            swash_cache: cosmic_text::SwashCache::new(),
            font_cache: FxHashMap::default(),
            prep_scratch: PrepScratch::default(),
        }
    }

    /// Number of cached glyph entries (including non-vector sentinels).
    pub fn glyph_count(&self) -> usize {
        self.glyphs.len()
    }

    /// Read-only access to the glyph cache, for querying non-vector classification.
    pub fn glyph_map(&self) -> &GlyphMap {
        &self.glyphs
    }

    /// Current atlas generation.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn glyph_mark_used(&mut self, key: &GlyphKey) -> Option<GlyphEntry> {
        self.glyphs.get_and_mark_used(key)
    }

    pub(crate) fn glyph(&self, key: &GlyphKey) -> Option<GlyphEntry> {
        self.glyphs.get(key)
    }

    /// Lazily append or grow a mono glyph's border descriptor and payload.
    /// The returned offset addresses the border descriptor, not the fill blob.
    pub(crate) fn resolve_border_blob(
        &mut self,
        key: GlyphKey,
        outline: &crate::outline::GlyphOutline,
        ppem: f32,
        radius_units: f32,
    ) -> Result<u32, crate::types::PrepareError> {
        let entry = self
            .glyphs
            .get(&key)
            .ok_or(crate::types::PrepareError::AtlasFull)?;
        // The two capacities are independent: ppem bounds the boundary
        // approximation's error, radius_units bounds the distance query.
        let mut ppem = ppem;
        let mut radius_units = radius_units;
        if let Some(border) = self
            .resident_blobs
            .get(&key)
            .and_then(|blob| blob.border.as_ref())
        {
            if border.ppem_ceiling >= ppem && border.grid_radius_units >= radius_units {
                return Ok(border.start_texel);
            }
            // Growth only: a rebuild triggered by one capacity must not
            // shrink the other below what an existing descriptor promised.
            // The builder derives its ceiling as next_pow2(2 * ppem), so
            // feeding the old ceiling back in would double it every
            // rebuild; half of it reproduces exactly the old ceiling.
            ppem = ppem.max(border.ppem_ceiling / 2.0);
            if border.grid_radius_units.is_finite() {
                radius_units = radius_units.max(border.grid_radius_units);
            }
        }
        let prepared = crate::border::prepare_border(
            outline,
            entry.glyph_offset,
            entry.units_per_em,
            ppem,
            radius_units,
        )?;
        let start = self.buffer_cursor;
        self.buffer_cursor = self.checked_buffer_end(prepared.texel_len)?;
        self.buffer_data.extend_from_slice(&prepared.data);
        let resident = self
            .resident_blobs
            .get_mut(&key)
            .ok_or(crate::types::PrepareError::AtlasFull)?;
        let replaced = resident.border.replace(ResidentBorderBlob {
            start_texel: start,
            texel_len: prepared.texel_len,
            ppem_ceiling: prepared.descriptor.ppem_ceiling,
            grid_radius_units: prepared.descriptor.grid_radius_units,
        });
        if let Some(old) = replaced {
            self.reclaimable_border_spans
                .push((old.start_texel, old.texel_len));
        }
        Ok(start)
    }

    /// Look up an already-resolved border descriptor. Instance emission
    /// uses this: every capacity must be resolved in one pre-pass before
    /// any descriptor is emitted, or a later, larger use in the same frame
    /// supersedes a blob an earlier instance already named.
    pub(crate) fn border_descriptor(&self, key: &GlyphKey) -> Option<u32> {
        self.resident_blobs
            .get(key)
            .and_then(|blob| blob.border.as_ref())
            .map(|border| border.start_texel)
    }

    pub(crate) fn resolve_border_glyph(
        &mut self,
        key: GlyphKey,
        ppem: f32,
        radius_units: f32,
    ) -> Result<u32, crate::types::PrepareError> {
        let weight = cosmic_text::Weight(key.font_weight);
        let cached = self
            .font_cache
            .get(&(key.font_id, weight))
            .expect("resolved mono glyph has cached font");
        let location = [VariationSetting::new(
            skrifa::Tag::new(b"wght"),
            key.font_weight as f32,
        )];
        let mut outline = extract_outline(
            cached.font.data(),
            cached.face_index,
            key.glyph_id,
            &location,
        )
        .ok_or(crate::types::PrepareError::AtlasFull)?;
        if key
            .cache_key_flags
            .contains(cosmic_text::CacheKeyFlags::FAKE_ITALIC)
        {
            apply_italic_shear(&mut outline);
        }
        self.resolve_border_blob(key, &outline, ppem, radius_units)
    }

    pub(crate) fn color_glyph(&self, key: &GlyphKey) -> Option<&ColorGlyphEntry> {
        self.color_glyphs.get(key)
    }

    pub(crate) fn color_v1_glyph(&self, key: &GlyphKey) -> Option<&ColorV1GlyphEntry> {
        self.color_v1_glyphs.get(key)
    }

    pub(crate) fn bind_group(&self) -> &BindGroup {
        &self.bind_group
    }

    pub fn buffer_elements_used(&self) -> u32 {
        self.buffer_cursor
    }

    #[doc(hidden)]
    pub fn blob_cache_stats(&self) -> BlobCacheStats {
        self.blob_cache.stats()
    }

    /// End-of-frame cache management.
    ///
    /// Clears per-frame usage tracking. When the buffer reaches four times
    /// its initial size AND fewer than a quarter of cached glyphs are in use,
    /// compacts current glyphs and moves inactive vector groups to the
    /// bounded CPU blob cache.
    ///
    /// With the default 1 MiB initial allocation, compaction eligibility begins at
    /// 4 MiB. This is pressure-based:
    /// a stable document with many glyphs will not trigger reset as long as
    /// the buffer has not substantially grown. Only when GPU memory has
    /// expanded AND the working set
    /// has shifted does eviction fire.
    pub fn trim(&mut self) {
        // Only consider reset when buffer has grown substantially.
        let substantial_growth = self.buffer_capacity
            >= self
                .initial_buffer_capacity
                .saturating_mul(TRIM_GROWTH_FACTOR);

        if substantial_growth {
            let cached = self.glyphs.len();
            let in_use = self.glyphs.in_use_count();

            if cached > 0 && in_use < cached / 4 {
                self.compact_atlas();
            } else {
                log::trace!(
                    "trim: retained ({in_use}/{cached} glyphs in use, \
                     buffer={}/{})",
                    self.buffer_cursor,
                    self.buffer_capacity,
                );
            }
        }

        self.glyphs.next_frame();

        if let Some(raster) = &mut self.raster {
            raster.trim();
        }
    }

    /// Lazily initialize raster pipeline. Called from TextRenderer::new().
    pub(crate) fn init_raster(
        &mut self,
        device: &Device,
        depth_stencil: Option<DepthStencilState>,
        multisample: MultisampleState,
    ) {
        if self.raster.is_none() {
            self.raster = Some(RasterState::new(
                device,
                self.format,
                self.cache.uniforms_layout(),
                depth_stencil,
                multisample,
            ));
        }
    }

    /// Rasterize non-vector glyphs and return per-instance vertex data.
    pub(crate) fn rasterize_glyphs(
        &mut self,
        queue: &Queue,
        font_system: &mut cosmic_text::FontSystem,
        glyphs: &[NonVectorGlyph],
        scroll: [f32; 2],
    ) -> Vec<RasterVertex> {
        if glyphs.is_empty() {
            return Vec::new();
        }
        let raster = match &mut self.raster {
            Some(r) => r,
            None => return Vec::new(),
        };
        raster.rasterize_glyphs(queue, font_system, &mut self.swash_cache, glyphs, scroll)
    }

    /// Set the raster pipeline and atlas bind group, then draw from the
    /// caller's vertex buffer.
    pub(crate) fn render_raster_pass(
        &self,
        viewport: &Viewport,
        pass: &mut RenderPass<'_>,
        vertex_buffer: &Buffer,
        count: u32,
    ) {
        if let Some(raster) = &self.raster {
            raster.render_pass(&viewport.bind_group, pass, vertex_buffer, count);
        }
    }

    fn compact_atlas(&mut self) {
        log::debug!(
            "trim: compacting atlas ({}/{} glyphs in use, buffer={}/{})",
            self.glyphs.in_use_count(),
            self.glyphs.len(),
            self.buffer_cursor,
            self.buffer_capacity,
        );

        let old_data = std::mem::take(&mut self.buffer_data);
        let mut residents: Vec<_> = self.resident_blobs.drain().collect();
        residents.sort_unstable_by_key(|(_, blob)| blob.start_texel);

        // Pass 1: keep current-frame blobs resident (copy into the compact
        // buffer); collect inactive blobs as lightweight candidates - spans
        // only, no payload copies until the budget selection has run.
        enum Source {
            Slice {
                start: usize,
                end: usize,
                kind: BlobKind,
                border: Option<(usize, usize, CachedBorderBlob)>,
            },
            Ready(CachedBlob),
        }
        let mut cand_keys: Vec<GlyphKey> = Vec::new();
        let mut cand_meta: Vec<(u64, usize)> = Vec::new();
        let mut cand_src: Vec<Source> = Vec::new();

        let mut compact = Vec::new();
        for (key, blob) in residents {
            let epoch = self.glyphs.last_used_epoch(&key).unwrap_or(0);
            let start = usize::try_from(blob.start_texel).expect("atlas offset") * 2;
            let end = start + usize::try_from(blob.texel_len).expect("atlas size") * 2;
            if self.glyphs.is_current_frame(&key) {
                let new_start = u32::try_from(compact.len() / 2).expect("atlas size");
                compact.extend_from_slice(&old_data[start..end]);
                let new_border = blob.border.as_ref().map(|border| {
                    let start = usize::try_from(border.start_texel).expect("atlas offset") * 2;
                    let end = start + usize::try_from(border.texel_len).expect("atlas size") * 2;
                    let new_border_start = u32::try_from(compact.len() / 2).expect("atlas size");
                    compact.extend_from_slice(&old_data[start..end]);
                    compact[new_border_start as usize * 2] = new_start as i32;
                    ResidentBorderBlob {
                        start_texel: new_border_start,
                        ..border.clone()
                    }
                });
                self.install_offsets(key, new_start, &blob.kind);
                self.resident_blobs.insert(
                    key,
                    ResidentBlob {
                        start_texel: new_start,
                        texel_len: blob.texel_len,
                        kind: blob.kind,
                        border: new_border,
                    },
                );
            } else {
                self.remove_glyph_group(&key);
                let cached_border = blob.border.map(|border| {
                    let border_start = border.start_texel as usize * 2;
                    let border_end = border_start + border.texel_len as usize * 2;
                    (
                        border_start,
                        border_end,
                        CachedBorderBlob {
                            relative_offset: blob.texel_len,
                            texel_len: border.texel_len,
                            ppem_ceiling: border.ppem_ceiling,
                            grid_radius_units: border.grid_radius_units,
                        },
                    )
                });
                cand_keys.push(key);
                let border_bytes = cached_border
                    .as_ref()
                    .map_or(0, |(start, end, _)| end - start);
                cand_meta.push((
                    epoch,
                    (end - start + border_bytes) * std::mem::size_of::<i32>(),
                ));
                cand_src.push(Source::Slice {
                    start,
                    end,
                    kind: blob.kind,
                    border: cached_border,
                });
            }
        }
        for (key, blob) in self.blob_cache.drain() {
            cand_keys.push(key);
            cand_meta.push((
                blob.last_used_epoch,
                blob.data.len() * std::mem::size_of::<i32>(),
            ));
            cand_src.push(Source::Ready(blob));
        }
        for key in self.glyphs.keys() {
            if !self.resident_blobs.contains_key(&key) && !self.glyphs.is_current_frame(&key) {
                self.remove_glyph_group(&key);
            }
        }

        // Pass 2: batch-select newest-first within the byte budget, then
        // materialize only the winners (losers are never copied).
        let (selected, budget_evictions, oversized_drops) =
            crate::blob_cache::select_within_budget(&cand_meta, self.blob_cache.budget());
        let mut kept = Vec::with_capacity(selected.len());
        let mut sources: Vec<Option<Source>> = cand_src.into_iter().map(Some).collect();
        for idx in selected {
            let source = sources[idx].take().expect("selection indices are unique");
            let (epoch, _) = cand_meta[idx];
            let blob = match source {
                Source::Ready(blob) => blob,
                Source::Slice {
                    start,
                    end,
                    kind,
                    border,
                } => {
                    let mut data = old_data[start..end].to_vec();
                    let border_meta = border.map(|(border_start, border_end, meta)| {
                        data.extend_from_slice(&old_data[border_start..border_end]);
                        meta
                    });
                    CachedBlob {
                        data: data.into_boxed_slice(),
                        kind,
                        border: border_meta,
                        last_used_epoch: epoch,
                    }
                }
            };
            kept.push((cand_keys[idx], blob));
        }
        self.blob_cache
            .replace_entries(kept, budget_evictions, oversized_drops);
        self.buffer_data = compact;
        self.reclaimable_border_spans.clear();
        self.buffer_cursor = u32::try_from(self.buffer_data.len() / 2).expect("atlas size");
        self.gpu_flush_cursor = 0;
        self.generation = self.generation.wrapping_add(1);

        self.buffer_capacity = planned_buffer_capacity(
            self.initial_buffer_capacity,
            self.buffer_cursor.max(self.initial_buffer_capacity),
            self.max_buffer_capacity,
        )
        .expect("live atlas fits");
        self.glyph_buffer = create_glyph_buffer(&self.device, self.buffer_capacity);
        self.bind_group = self
            .cache
            .create_atlas_bind_group(&self.device, &self.glyph_buffer);
    }

    fn remove_glyph_group(&mut self, key: &GlyphKey) {
        self.glyphs.remove(key);
        self.color_glyphs.remove(key);
        self.color_v1_glyphs.remove(key);
    }

    fn install_offsets(&mut self, key: GlyphKey, start: u32, kind: &BlobKind) {
        match kind {
            BlobKind::Mono { .. } => self.glyphs.replace_offset(&key, start),
            BlobKind::ColorV1 {
                cmd_texel_count,
                bounds,
                units_per_em,
            } => {
                self.color_v1_glyphs.insert(
                    key,
                    ColorV1GlyphEntry {
                        glyph_offset: start,
                        cmd_texel_count: *cmd_texel_count,
                        bounds: *bounds,
                        units_per_em: *units_per_em,
                    },
                );
            }
            BlobKind::ColorV0 {
                units_per_em,
                layers,
            } => {
                let layers = layers
                    .iter()
                    .map(|layer| ColorGlyphLayer {
                        entry: layer
                            .offset
                            .map_or(crate::glyph_cache::NON_VECTOR_GLYPH, |offset| {
                                GlyphEntry::new(start + offset, layer.bounds, *units_per_em)
                            }),
                        color: layer.color,
                        use_foreground: layer.use_foreground,
                    })
                    .collect();
                self.color_glyphs.insert(
                    key,
                    ColorGlyphEntry {
                        layers,
                        units_per_em: *units_per_em,
                    },
                );
            }
        }
    }

    /// Resolve a glyph: return its resident entry or restore, extract, and upload it.
    /// This is a cold path; the resident lookup keeps calls safe independent of pass 1.
    pub(crate) fn resolve_glyph(
        &mut self,
        font_system: &mut cosmic_text::FontSystem,
        key: GlyphKey,
    ) -> Result<GlyphEntry, crate::types::PrepareError> {
        if let Some(entry) = self.glyph_mark_used(&key) {
            return Ok(entry);
        }
        if let Some(entry) = self.restore_cached_glyph(key)? {
            return Ok(entry);
        }

        let font_weight = cosmic_text::Weight(key.font_weight);
        let cache_key = (key.font_id, font_weight);
        if let std::collections::hash_map::Entry::Vacant(slot) = self.font_cache.entry(cache_key) {
            let face_index = font_system
                .db()
                .face(key.font_id)
                .map(|info| info.index)
                .unwrap_or(0);
            let font = match font_system.get_font(key.font_id, font_weight) {
                Some(font) => font,
                None => {
                    log::warn!("Font not found for glyph {key:?}");
                    return Ok(self
                        .glyphs
                        .insert_and_mark_used(key, crate::glyph_cache::NON_VECTOR_GLYPH));
                }
            };
            let skrifa_font = skrifa::FontRef::from_index(font.data(), face_index).ok();
            let units_per_em = skrifa_font
                .as_ref()
                .and_then(|font| {
                    use skrifa::raw::TableProvider;
                    font.head().map(|head| head.units_per_em() as f32).ok()
                })
                .unwrap_or(1000.0);
            let has_colr = skrifa_font
                .as_ref()
                .map(|font| {
                    use skrifa::raw::TableProvider;
                    font.colr().is_ok()
                })
                .unwrap_or(false);
            slot.insert(CachedFont {
                font,
                face_index,
                units_per_em,
                has_colr,
            });
        }

        let (font, face_index, units_per_em, has_colr) = {
            let cached = &self.font_cache[&cache_key];
            (
                Arc::clone(&cached.font),
                cached.face_index,
                cached.units_per_em,
                cached.has_colr,
            )
        };
        let font_data = font.data();
        let location = [VariationSetting::new(
            skrifa::Tag::new(b"wght"),
            key.font_weight as f32,
        )];
        let color_info = has_colr
            .then(|| extract_color_info(font_data, face_index, key.glyph_id, &location))
            .flatten();
        let entry = match color_info {
            Some(ColorGlyphInfo::V0Layers(layers)) => {
                let fake_italic = key
                    .cache_key_flags
                    .contains(cosmic_text::CacheKeyFlags::FAKE_ITALIC);
                self.upload_colr_v0_layers(
                    font_data,
                    face_index,
                    units_per_em,
                    &location,
                    &layers,
                    fake_italic,
                    key,
                )
                .unwrap_or(crate::glyph_cache::NON_VECTOR_GLYPH)
            }
            Some(ColorGlyphInfo::V1(mut v1_data)) => {
                match self.upload_color_v1(key, &mut v1_data, units_per_em) {
                    Ok(entry) => {
                        self.color_v1_glyphs.insert(key, entry);
                        COLOR_V1_VECTOR_GLYPH
                    }
                    Err(_) => crate::glyph_cache::NON_VECTOR_GLYPH,
                }
            }
            None => match extract_outline(font_data, face_index, key.glyph_id, &location) {
                Some(mut outline) => {
                    if key
                        .cache_key_flags
                        .contains(cosmic_text::CacheKeyFlags::FAKE_ITALIC)
                    {
                        apply_italic_shear(&mut outline);
                    }
                    let band_count = band_count_for_curves(outline.curves.len());
                    match prepare_mono(
                        &outline,
                        band_count,
                        band_count,
                        units_per_em,
                        &mut self.prep_scratch,
                    ) {
                        Some(prepared) => self.commit_mono(key, &prepared)?,
                        None => crate::glyph_cache::NON_VECTOR_GLYPH,
                    }
                }
                None => crate::glyph_cache::NON_VECTOR_GLYPH,
            },
        };
        Ok(self.glyphs.insert_and_mark_used(key, entry))
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_colr_v0_layers(
        &mut self,
        font_data: &[u8],
        face_index: u32,
        units_per_em: f32,
        location: &[VariationSetting],
        layers: &[crate::outline::ColorLayer],
        fake_italic: bool,
        key: GlyphKey,
    ) -> Result<GlyphEntry, crate::types::PrepareError> {
        let mut prepared_layers = Vec::with_capacity(layers.len());
        for layer in layers {
            let Some(mut outline) =
                extract_outline(font_data, face_index, layer.glyph_id, location)
            else {
                continue;
            };
            if fake_italic {
                apply_italic_shear(&mut outline);
            }
            let band_count = band_count_for_curves(outline.curves.len());
            let prepared = prepare_mono(
                &outline,
                band_count,
                band_count,
                units_per_em,
                &mut self.prep_scratch,
            );
            prepared_layers.push((prepared, layer.color, layer.use_foreground));
        }
        if prepared_layers.is_empty() {
            return Ok(crate::glyph_cache::NON_VECTOR_GLYPH);
        }
        let entry = self.commit_color_v0(key, &prepared_layers, units_per_em)?;
        self.color_glyphs.insert(key, entry);
        Ok(COLOR_VECTOR_GLYPH)
    }

    fn restore_cached_glyph(
        &mut self,
        key: GlyphKey,
    ) -> Result<Option<GlyphEntry>, crate::types::PrepareError> {
        let Some(texel_len) = self.blob_cache.texel_len(&key) else {
            return Ok(None);
        };
        let new_end = self.checked_buffer_end(texel_len)?;
        let mut blob = self.blob_cache.take(&key).expect("cache entry checked");
        let start = self.buffer_cursor;
        if let Some(border) = &blob.border {
            blob.data[border.relative_offset as usize * 2] = start as i32;
        }
        self.buffer_data.extend_from_slice(&blob.data);
        self.buffer_cursor = new_end;
        let entry = match &blob.kind {
            BlobKind::Mono {
                bounds,
                units_per_em,
                ..
            } => GlyphEntry::new(start, *bounds, *units_per_em),
            BlobKind::ColorV0 { .. } => COLOR_VECTOR_GLYPH,
            BlobKind::ColorV1 { .. } => COLOR_V1_VECTOR_GLYPH,
        };
        self.install_offsets(key, start, &blob.kind);
        let primary_texel_len = blob
            .border
            .as_ref()
            .map_or(texel_len, |border| border.relative_offset);
        let resident_border = blob.border.as_ref().map(|border| ResidentBorderBlob {
            start_texel: start + border.relative_offset,
            texel_len: border.texel_len,
            ppem_ceiling: border.ppem_ceiling,
            grid_radius_units: border.grid_radius_units,
        });
        self.resident_blobs.insert(
            key,
            ResidentBlob {
                start_texel: start,
                texel_len: primary_texel_len,
                kind: blob.kind,
                border: resident_border,
            },
        );
        let entry = self.glyphs.insert_and_mark_used(key, entry);
        self.blob_cache
            .add_repopulated_bytes(u64::from(texel_len) * BYTES_PER_TEXEL);
        Ok(Some(entry))
    }

    /// Flush all pending glyph uploads to the GPU in a single write_buffer call.
    /// Call this once per frame after all upload_glyph calls are complete.
    pub(crate) fn flush_uploads(&mut self, queue: &Queue) {
        if self.buffer_cursor > self.buffer_capacity {
            self.grow_buffer(self.buffer_cursor);
        }

        let start = self.gpu_flush_cursor as usize;
        let end = self.buffer_cursor as usize;
        if start < end {
            let byte_offset = start as u64 * BYTES_PER_TEXEL;
            // buffer_data has 2 i32s per texel
            let i32_start = start * 2;
            let i32_end = end * 2;
            let blob_bytes: &[u8] =
                bytemuck::cast_slice::<i32, u8>(&self.buffer_data[i32_start..i32_end]);
            queue.write_buffer(&self.glyph_buffer, byte_offset, blob_bytes);
            self.gpu_flush_cursor = self.buffer_cursor;
        }
    }

    /// Commit a CPU-prepared mono glyph blob to the atlas storage buffer.
    /// Pure write side: appends a universal header then `prepared.blob_data`.
    #[hotpath::measure]
    fn commit_mono(
        &mut self,
        key: GlyphKey,
        prepared: &crate::prep::PreparedMono,
    ) -> Result<GlyphEntry, crate::types::PrepareError> {
        if prepared.blob_size > 65535 {
            return Err(crate::types::PrepareError::AtlasFull);
        }

        let glyph_offset = self.buffer_cursor;
        let total_size = GLYPH_HEADER_TEXELS
            .checked_add(prepared.blob_size)
            .ok_or(crate::types::PrepareError::AtlasFull)?;
        let new_end = self.checked_buffer_end(total_size)?;

        self.buffer_data.extend_from_slice(&encode_glyph_header(
            prepared.bounds,
            prepared.band_transform,
            prepared.band_count_x.saturating_sub(1),
            prepared.band_count_y.saturating_sub(1),
        ));
        self.buffer_data.extend_from_slice(&prepared.blob_data);
        self.buffer_cursor = new_end;

        let entry = GlyphEntry {
            glyph_offset,
            bounds: prepared.bounds,
            units_per_em: prepared.units_per_em,
            last_used_epoch: 0,
        };
        self.resident_blobs.insert(
            key,
            ResidentBlob {
                start_texel: glyph_offset,
                texel_len: total_size,
                kind: BlobKind::Mono {
                    bounds: prepared.bounds,
                    units_per_em: prepared.units_per_em,
                },
                border: None,
            },
        );
        Ok(entry)
    }

    /// Upload a COLRv1 color glyph command blob.
    ///
    /// Layout: [header] [commands...] [sub_glyph_0: header + bands + curves] [sub_glyph_1: ...] ...
    /// Sub-glyph header (3 texels): band_max (packed) + band_transform (4 raw i32).
    /// Command DRAW opcodes reference sub-glyphs by command-payload-relative
    /// texel offset (relative to the first command texel, after the header).
    ///
    /// COLRv1 commands and headers store raw i32 values (not i16-packed) since
    /// they contain bitcast f32 and packed color data that uses full i32 range.
    /// Each raw `[i32; 4]` occupies 2 packed texels (4 i32 slots).
    /// Band and curve data within sub-glyphs are i16-packed as usual.
    fn upload_color_v1(
        &mut self,
        key: GlyphKey,
        v1: &mut crate::outline::ColorV1Data,
        units_per_em: f32,
    ) -> Result<ColorV1GlyphEntry, crate::types::PrepareError> {
        // Phase 1: build each sub-glyph's blob (header + bands + curves).
        struct SubGlyphBlob {
            /// Header: 3 packed texels (6 i32 slots).
            /// [0-1]: band_max_x, band_max_y (packed i16 pair + padding)
            /// [2-5]: band_transform (4 raw i32, bitcast f32)
            header: [i32; 6],
            band_entries_packed: Vec<i32>, // 2 i32 per texel (packed i16 pairs)
            curve_texels: Vec<[i32; 4]>,   // intermediate; packed at append time
        }

        // Commands occupy 2 packed texels each (4 raw i32 values per command)
        let cmd_texel_count = u32::try_from(v1.commands.len())
            .ok()
            .and_then(|count| count.checked_mul(2))
            .ok_or(crate::types::PrepareError::AtlasFull)?;
        let mut sub_blobs: Vec<SubGlyphBlob> = Vec::with_capacity(v1.sub_glyphs.len());
        let mut union_bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];

        for sub in &v1.sub_glyphs {
            let outline = &sub.outline;
            let num_curves = u32::try_from(outline.curves.len())
                .map_err(|_| crate::types::PrepareError::AtlasFull)?;

            // Union bounds
            union_bounds[0] = union_bounds[0].min(outline.bounds[0]);
            union_bounds[1] = union_bounds[1].min(outline.bounds[1]);
            union_bounds[2] = union_bounds[2].max(outline.bounds[2]);
            union_bounds[3] = union_bounds[3].max(outline.bounds[3]);

            // Build curve texels
            let q = |v: f32| -> i32 { (v * 4.0).round() as i32 };
            let curve_capacity = usize::try_from(num_curves)
                .ok()
                .and_then(|count| count.checked_mul(2))
                .ok_or(crate::types::PrepareError::AtlasFull)?;
            let mut curve_texels = Vec::with_capacity(curve_capacity);
            let mut curve_locations = Vec::with_capacity(
                usize::try_from(num_curves).map_err(|_| crate::types::PrepareError::AtlasFull)?,
            );

            for (i, curve) in outline.curves.iter().enumerate() {
                let is_continuation = i > 0 && curve.p1 == outline.curves[i - 1].p3;
                if is_continuation {
                    let last: &mut [i32; 4] = curve_texels.last_mut().expect("continuation");
                    last[2] = q(curve.p2[0]);
                    last[3] = q(curve.p2[1]);
                } else {
                    curve_texels.push([
                        q(curve.p1[0]),
                        q(curve.p1[1]),
                        q(curve.p2[0]),
                        q(curve.p2[1]),
                    ]);
                }
                curve_locations.push(CurveLocation {
                    offset: u32::try_from(curve_texels.len())
                        .ok()
                        .and_then(|length| length.checked_sub(1))
                        .ok_or(crate::types::PrepareError::AtlasFull)?,
                });
                curve_texels.push([q(curve.p3[0]), q(curve.p3[1]), 0, 0]);
            }

            let band_count_x = (num_curves).clamp(1, 16);
            let band_count_y = band_count_x;
            let band_data = crate::band::build_bands(
                outline,
                &curve_locations,
                band_count_x,
                band_count_y,
                self.scratch_band_entries.split_off(0),
                &mut self.band_scratch,
            );
            self.scratch_band_entries = band_data.entries;

            let (band_chunks, _) = self.scratch_band_entries.as_chunks::<4>();
            let band_entries_packed: Vec<i32> = band_chunks
                .iter()
                .flat_map(|c| [pack_i16_pair(c[0], c[1]), pack_i16_pair(c[2], c[3])])
                .collect();

            let bt = band_data.band_transform;
            // Sub-glyph header: 3 packed texels (6 i32 slots).
            let header: [i32; 6] = [
                // Texel 0: band_max (packed i16 pair) + padding
                pack_i16_pair(
                    band_count_x.saturating_sub(1) as i16,
                    band_count_y.saturating_sub(1) as i16,
                ),
                0, // padding
                // Texels 1-2: band_transform as raw i32 (bitcast f32)
                f32::to_bits(bt[0]) as i32,
                f32::to_bits(bt[1]) as i32,
                f32::to_bits(bt[2]) as i32,
                f32::to_bits(bt[3]) as i32,
            ];

            sub_blobs.push(SubGlyphBlob {
                header,
                band_entries_packed,
                curve_texels,
            });
        }

        // Phase 2: compute sub-glyph offsets within the blob.
        // Blob layout: [commands (2 texels each)] [sub0: header(3) + bands + curves] [sub1: ...]
        let mut offset = cmd_texel_count;
        for (i, blob) in sub_blobs.iter().enumerate() {
            v1.sub_glyphs[i].blob_offset = offset;
            // header(3 texels) + bands (already in texel units) + curves
            let band_texels = u32::try_from(blob.band_entries_packed.len())
                .map_err(|_| crate::types::PrepareError::AtlasFull)?
                / 2;
            let curve_texels = u32::try_from(blob.curve_texels.len())
                .map_err(|_| crate::types::PrepareError::AtlasFull)?;
            offset = offset
                .checked_add(3)
                .and_then(|value| value.checked_add(band_texels))
                .and_then(|value| value.checked_add(curve_texels))
                .ok_or(crate::types::PrepareError::AtlasFull)?;
        }
        let total_blob_size = offset;

        // Phase 3: fixup command sub-glyph indices → command-payload-relative offsets.
        for cmd in &mut v1.commands {
            let opcode = cmd[0];
            if opcode == crate::outline::CMD_DRAW_SOLID
                || opcode == crate::outline::CMD_DRAW_GRADIENT
            {
                let sub_idx = cmd[1] as usize;
                if sub_idx < v1.sub_glyphs.len() {
                    cmd[1] = i32::try_from(v1.sub_glyphs[sub_idx].blob_offset)
                        .map_err(|_| crate::types::PrepareError::AtlasFull)?;
                }
            }
        }

        // Phase 4: validate capacity and append to CPU storage.
        let glyph_offset = self.buffer_cursor;
        let total_size = GLYPH_HEADER_TEXELS
            .checked_add(total_blob_size)
            .ok_or(crate::types::PrepareError::AtlasFull)?;
        let new_end = self.checked_buffer_end(total_size)?;

        self.buffer_data
            .extend_from_slice(&encode_glyph_header(union_bounds, [0.0; 4], 0, 0));
        // Append commands: each [i32; 4] command → 4 raw i32 values (2 packed texels)
        for cmd in &v1.commands {
            self.buffer_data.extend_from_slice(cmd);
        }
        // Append sub-glyph blobs
        for blob in &sub_blobs {
            self.buffer_data.extend_from_slice(&blob.header);
            self.buffer_data
                .extend_from_slice(&blob.band_entries_packed);
            // Pack curve texels
            for v in &blob.curve_texels {
                self.buffer_data
                    .push(pack_i16_pair(v[0] as i16, v[1] as i16));
                self.buffer_data
                    .push(pack_i16_pair(v[2] as i16, v[3] as i16));
            }
        }

        self.buffer_cursor = new_end;

        let entry = ColorV1GlyphEntry {
            glyph_offset,
            cmd_texel_count,
            bounds: union_bounds,
            units_per_em,
        };
        self.resident_blobs.insert(
            key,
            ResidentBlob {
                start_texel: glyph_offset,
                texel_len: total_size,
                kind: BlobKind::ColorV1 {
                    bounds: union_bounds,
                    units_per_em,
                    cmd_texel_count,
                },
                border: None,
            },
        );
        Ok(entry)
    }

    fn commit_color_v0(
        &mut self,
        key: GlyphKey,
        layers: &[(Option<crate::prep::PreparedMono>, [f32; 4], bool)],
        units_per_em: f32,
    ) -> Result<ColorGlyphEntry, crate::types::PrepareError> {
        if layers.iter().any(|(prepared, _, _)| {
            prepared
                .as_ref()
                .is_some_and(|prepared| prepared.blob_size > 65535)
        }) {
            return Err(crate::types::PrepareError::AtlasFull);
        }
        let total = layers
            .iter()
            .try_fold(0_u32, |size, (prepared, _, _)| {
                prepared.as_ref().map_or(Some(size), |prepared| {
                    GLYPH_HEADER_TEXELS
                        .checked_add(prepared.blob_size)
                        .and_then(|part| size.checked_add(part))
                })
            })
            .ok_or(crate::types::PrepareError::AtlasFull)?;
        if total == 0 {
            let layers = layers
                .iter()
                .map(|(_, color, use_foreground)| ColorGlyphLayer {
                    entry: crate::glyph_cache::NON_VECTOR_GLYPH,
                    color: *color,
                    use_foreground: *use_foreground,
                })
                .collect();
            return Ok(ColorGlyphEntry {
                layers,
                units_per_em,
            });
        }
        let new_end = self.checked_buffer_end(total)?;
        let start = self.buffer_cursor;
        let mut cached_layers = Vec::with_capacity(layers.len());
        let mut installed_layers = Vec::with_capacity(layers.len());
        for (prepared, color, use_foreground) in layers {
            if let Some(prepared) = prepared {
                let relative = self.buffer_cursor - start;
                self.buffer_data.extend_from_slice(&encode_glyph_header(
                    prepared.bounds,
                    prepared.band_transform,
                    prepared.band_count_x.saturating_sub(1),
                    prepared.band_count_y.saturating_sub(1),
                ));
                self.buffer_data.extend_from_slice(&prepared.blob_data);
                self.buffer_cursor += GLYPH_HEADER_TEXELS + prepared.blob_size;
                cached_layers.push(CachedColorLayer {
                    offset: Some(relative),
                    bounds: prepared.bounds,
                    color: *color,
                    use_foreground: *use_foreground,
                });
                installed_layers.push(ColorGlyphLayer {
                    entry: GlyphEntry::new(start + relative, prepared.bounds, units_per_em),
                    color: *color,
                    use_foreground: *use_foreground,
                });
            } else {
                cached_layers.push(CachedColorLayer {
                    offset: None,
                    bounds: [0.0; 4],
                    color: *color,
                    use_foreground: *use_foreground,
                });
                installed_layers.push(ColorGlyphLayer {
                    entry: crate::glyph_cache::NON_VECTOR_GLYPH,
                    color: *color,
                    use_foreground: *use_foreground,
                });
            }
        }
        self.buffer_cursor = new_end;
        let kind = BlobKind::ColorV0 {
            units_per_em,
            layers: cached_layers.into_boxed_slice(),
        };
        self.resident_blobs.insert(
            key,
            ResidentBlob {
                start_texel: start,
                texel_len: total,
                kind,
                border: None,
            },
        );
        Ok(ColorGlyphEntry {
            layers: installed_layers,
            units_per_em,
        })
    }

    fn checked_buffer_end(&self, blob_size: u32) -> Result<u32, crate::types::PrepareError> {
        checked_buffer_end(self.buffer_cursor, blob_size, self.max_buffer_capacity)
    }

    fn grow_buffer(&mut self, min_capacity: u32) {
        let new_cap =
            planned_buffer_capacity(self.buffer_capacity, min_capacity, self.max_buffer_capacity)
                .expect("validated buffer capacity");

        log::debug!(
            "Growing glyph buffer: {} → {} elements (cursor: {})",
            self.buffer_capacity,
            new_cap,
            self.buffer_cursor,
        );

        self.glyph_buffer = create_glyph_buffer(&self.device, new_cap);
        self.buffer_capacity = new_cap;

        // Mark all data as needing flush to the new buffer
        self.gpu_flush_cursor = 0;

        self.bind_group = self
            .cache
            .create_atlas_bind_group(&self.device, &self.glyph_buffer);
    }

    pub(crate) fn get_or_create_pipeline(
        &self,
        device: &Device,
        multisample: MultisampleState,
        depth_stencil: Option<DepthStencilState>,
    ) -> RenderPipeline {
        self.cache
            .get_or_create_pipeline(device, self.format, multisample, depth_stencil)
    }

    pub(crate) fn get_or_create_border_pipeline(
        &self,
        device: &Device,
        multisample: MultisampleState,
        depth_stencil: Option<DepthStencilState>,
    ) -> crate::gpu_cache::BorderPipelineState {
        self.cache
            .get_or_create_border_pipeline(device, self.format, multisample, depth_stencil)
    }
}

/// Determine the band count for a glyph based on its curve complexity.
/// Matches harfbuzz: 1:1 up to a cap of 16 bands.
fn band_count_for_curves(num_curves: usize) -> u32 {
    (num_curves as u32).clamp(1, 16)
}

/// Encode a universal 5-texel glyph header. Float values retain their exact
/// bits because the vertex shader reads these raw i32 slots and bitcasts them.
fn encode_glyph_header(
    bounds: [f32; 4],
    band_transform: [f32; 4],
    band_max_x: u32,
    band_max_y: u32,
) -> [i32; 10] {
    [
        bounds[0].to_bits() as i32,
        bounds[1].to_bits() as i32,
        bounds[2].to_bits() as i32,
        bounds[3].to_bits() as i32,
        band_transform[0].to_bits() as i32,
        band_transform[1].to_bits() as i32,
        band_transform[2].to_bits() as i32,
        band_transform[3].to_bits() as i32,
        pack_i16_pair(band_max_x as i16, band_max_y as i16),
        0,
    ]
}

/// Bytes per texel in the packed layout: 2 i32 values = 8 bytes.
const BYTES_PER_TEXEL: u64 = 8;

fn max_buffer_capacity(device: &Device) -> u32 {
    let limits = device.limits();
    // A storage binding is constrained by both the binding-size limit and
    // the overall buffer-size limit; honor whichever is smaller.
    let max_bytes = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let max_texels = max_bytes / BYTES_PER_TEXEL;
    max_texels.min(u64::from(u32::MAX)) as u32
}

/// Choose a single allocation that can hold `required_capacity` texels.
///
/// Kept independent of wgpu so the overflow and device-limit behavior can be
/// tested without a device.
fn planned_buffer_capacity(
    current_capacity: u32,
    required_capacity: u32,
    max_capacity: u32,
) -> Result<u32, crate::types::PrepareError> {
    if required_capacity > max_capacity {
        return Err(crate::types::PrepareError::AtlasFull);
    }

    let rounded = required_capacity
        .checked_next_power_of_two()
        .unwrap_or(max_capacity)
        .min(max_capacity);
    Ok(current_capacity.max(rounded).min(max_capacity))
}

fn checked_buffer_end(
    cursor: u32,
    blob_size: u32,
    max_capacity: u32,
) -> Result<u32, crate::types::PrepareError> {
    let end = cursor
        .checked_add(blob_size)
        .ok_or(crate::types::PrepareError::AtlasFull)?;
    if end > max_capacity {
        return Err(crate::types::PrepareError::AtlasFull);
    }
    Ok(end)
}

fn create_glyph_buffer(device: &Device, capacity_texels: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sluggrs glyph buffer"),
        size: capacity_texels as u64 * BYTES_PER_TEXEL,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Pack two i16 values into a single i32.
/// Layout: low 16 bits = first value, high 16 bits = second value.
/// Matches the shader's `unpack_lo/unpack_hi` extraction.
fn pack_i16_pair(a: i16, b: i16) -> i32 {
    (a as u16 as u32 | ((b as u16 as u32) << 16)) as i32
}

#[cfg(test)]
mod tests {
    use super::{checked_buffer_end, encode_glyph_header, planned_buffer_capacity};
    use crate::types::PrepareError;

    #[test]
    fn capacity_plan_rounds_up_once() {
        assert_eq!(planned_buffer_capacity(128, 129, 4096), Ok(256));
        assert_eq!(planned_buffer_capacity(128, 1025, 4096), Ok(2048));
    }

    #[test]
    fn capacity_plan_clamps_to_device_limit() {
        assert_eq!(planned_buffer_capacity(128, 3000, 3072), Ok(3072));
    }

    #[test]
    fn capacity_plan_rejects_limit_misses() {
        assert_eq!(
            planned_buffer_capacity(128, u32::MAX, u32::MAX - 1),
            Err(PrepareError::AtlasFull)
        );
        assert_eq!(
            planned_buffer_capacity(128, u32::MAX, u32::MAX),
            Ok(u32::MAX)
        );
    }

    #[test]
    fn buffer_end_rejects_overflow_and_limit_misses() {
        assert_eq!(
            checked_buffer_end(u32::MAX, 1, u32::MAX),
            Err(PrepareError::AtlasFull)
        );
        assert_eq!(
            checked_buffer_end(100, 1, 100),
            Err(PrepareError::AtlasFull)
        );
    }

    #[test]
    fn glyph_header_preserves_float_bits_and_packs_maxima() {
        let header = encode_glyph_header(
            [-0.0, f32::NAN, 3.5, -4.25],
            [1.0, -2.0, 0.25, f32::INFINITY],
            12,
            255,
        );
        assert_eq!(header[0] as u32, (-0.0f32).to_bits());
        assert_eq!(header[1] as u32, f32::NAN.to_bits());
        assert_eq!(header[2] as u32, 3.5f32.to_bits());
        assert_eq!(header[3] as u32, (-4.25f32).to_bits());
        assert_eq!(header[4] as u32, 1.0f32.to_bits());
        assert_eq!(header[5] as u32, (-2.0f32).to_bits());
        assert_eq!(header[6] as u32, 0.25f32.to_bits());
        assert_eq!(header[7] as u32, f32::INFINITY.to_bits());
        assert_eq!(header[8] as u32, 12 | (255 << 16));
        assert_eq!(header[9], 0);
    }
}
