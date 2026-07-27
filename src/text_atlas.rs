use crate::band::{BandScratch, CurveLocation};
use crate::glyph_cache::{ColorGlyphEntry, ColorV1GlyphEntry, GlyphEntry, GlyphKey, GlyphMap};
use crate::gpu_cache::Cache;
use crate::raster_text::{NonVectorGlyph, RasterState, RasterVertex};
use crate::types::ColorMode;
use crate::viewport::Viewport;
use rustc_hash::FxHashMap;

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

pub struct TextAtlas {
    pub(crate) cache: Cache,
    device: Device,
    pub(crate) glyph_buffer: wgpu::Buffer,
    pub(crate) bind_group: BindGroup,
    pub(crate) format: TextureFormat,

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
    pub(crate) glyphs: GlyphMap,
    /// COLRv0 color glyph layers, keyed by the same GlyphKey as the main map.
    pub(crate) color_glyphs: FxHashMap<GlyphKey, ColorGlyphEntry>,
    /// COLRv1 color glyph command sequences.
    pub(crate) color_v1_glyphs: FxHashMap<GlyphKey, ColorV1GlyphEntry>,
    /// Monotonic counter incremented on atlas reset. Used by TextRenderer's
    /// retained cache to detect when cached glyph offsets are invalidated.
    generation: u32,

    // Raster fallback for non-vector glyphs (emoji, bitmap fonts)
    raster: Option<RasterState>,
    swash_cache: cosmic_text::SwashCache,
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
            raster: None,
            swash_cache: cosmic_text::SwashCache::new(),
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

    pub fn buffer_elements_used(&self) -> u32 {
        self.buffer_cursor
    }

    /// End-of-frame cache management.
    ///
    /// Clears per-frame usage tracking. When the buffer reaches four times
    /// its initial size AND fewer than a quarter of cached glyphs are in use,
    /// performs a full reset: recreates buffer at initial size (reclaiming
    /// GPU memory immediately), clears the glyph cache. The next prepare()
    /// re-extracts only the visible glyphs.
    ///
    /// With the default 1 MiB initial allocation, reset eligibility begins at
    /// 4 MiB and a reset returns the atlas to 1 MiB. This is pressure-based:
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
                self.reset_atlas();
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
    ) -> Vec<RasterVertex> {
        if glyphs.is_empty() {
            return Vec::new();
        }
        let raster = match &mut self.raster {
            Some(r) => r,
            None => return Vec::new(),
        };
        raster.rasterize_glyphs(queue, font_system, &mut self.swash_cache, glyphs)
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

    /// Full atlas reset: recreate buffer at initial size, clear all caches.
    /// GPU memory is reclaimed immediately (old buffer is dropped).
    fn reset_atlas(&mut self) {
        log::debug!(
            "trim: resetting atlas ({}/{} glyphs in use, buffer={}/{})",
            self.glyphs.in_use_count(),
            self.glyphs.len(),
            self.buffer_cursor,
            self.buffer_capacity,
        );

        self.glyphs.clear();
        self.color_glyphs.clear();
        self.color_v1_glyphs.clear();
        self.buffer_cursor = 0;
        self.buffer_data = Vec::new(); // reclaim CPU memory (clear() keeps capacity)
        self.gpu_flush_cursor = 0;
        self.generation = self.generation.wrapping_add(1);

        self.buffer_capacity = self.initial_buffer_capacity;
        self.glyph_buffer = create_glyph_buffer(&self.device, self.buffer_capacity);
        self.bind_group = self
            .cache
            .create_atlas_bind_group(&self.device, &self.glyph_buffer);
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
    /// Pure write side: appends `prepared.blob_data` and advances the cursor.
    #[hotpath::measure]
    pub(crate) fn commit_mono(
        &mut self,
        prepared: &crate::prep::PreparedMono,
    ) -> Result<GlyphEntry, crate::types::PrepareError> {
        if prepared.blob_size > 65535 {
            return Err(crate::types::PrepareError::AtlasFull);
        }

        let glyph_offset = self.buffer_cursor;
        let new_end = self.checked_buffer_end(prepared.blob_size)?;

        self.buffer_data.extend_from_slice(&prepared.blob_data);
        self.buffer_cursor = new_end;

        Ok(GlyphEntry {
            band_offset: glyph_offset,
            band_max_x: prepared.band_count_x.saturating_sub(1),
            band_max_y: prepared.band_count_y.saturating_sub(1),
            band_transform: prepared.band_transform,
            bounds: prepared.bounds,
            units_per_em: prepared.units_per_em,
            last_used_epoch: 0,
        })
    }

    /// Upload a COLRv1 color glyph command blob.
    ///
    /// Layout: [commands...] [sub_glyph_0: header + bands + curves] [sub_glyph_1: ...] ...
    /// Sub-glyph header (3 texels): band_max (packed) + band_transform (4 raw i32).
    /// Command DRAW opcodes reference sub-glyphs by blob-relative texel offset.
    ///
    /// COLRv1 commands and headers store raw i32 values (not i16-packed) since
    /// they contain bitcast f32 and packed color data that uses full i32 range.
    /// Each raw `[i32; 4]` occupies 2 packed texels (4 i32 slots).
    /// Band and curve data within sub-glyphs are i16-packed as usual.
    pub(crate) fn upload_color_v1(
        &mut self,
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

        // Phase 3: fixup command sub-glyph indices → blob-relative offsets.
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
        let new_end = self.checked_buffer_end(total_blob_size)?;

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

        Ok(ColorV1GlyphEntry {
            blob_offset: glyph_offset,
            cmd_count: cmd_texel_count,
            bounds: union_bounds,
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
    use super::{checked_buffer_end, planned_buffer_capacity};
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
}
