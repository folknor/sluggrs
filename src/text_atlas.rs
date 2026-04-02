use crate::band::CurveLocation;
use crate::glyph_cache::{GlyphEntry, GlyphMap};
use crate::gpu_cache::Cache;
use crate::prepare::GpuOutline;
use crate::types::ColorMode;

use wgpu::{
    BindGroup, DepthStencilState, Device, MultisampleState, Queue, RenderPipeline, TextureFormat,
};

/// Curve texture width. Fixed like the band texture — rows wrap at this boundary.

/// An atlas containing cached glyph curve and band data for GPU rendering.
/// Initial buffer capacity in vec4<i32> elements (16 bytes each).
const INITIAL_BUFFER_CAPACITY: u32 = 8192;

pub struct TextAtlas {
    pub(crate) cache: Cache,
    device: Device,
    pub(crate) glyph_buffer: wgpu::Buffer,
    pub(crate) bind_group: BindGroup,
    pub(crate) format: TextureFormat,
    #[allow(dead_code)] // API contract — read by iced integration
    pub(crate) color_mode: ColorMode,

    // Buffer state
    buffer_capacity: u32, // in vec4<i32> elements
    buffer_cursor: u32,   // append cursor in elements
    buffer_data: Vec<[i32; 4]>, // CPU-side copy for re-upload on growth

    // Scratch buffers reused across upload_glyph() calls
    scratch_curve_texels: Vec<[i32; 4]>,
    scratch_curve_locations: Vec<CurveLocation>,
    scratch_band_entries: Vec<i16>,

    // Glyph cache
    pub(crate) glyphs: GlyphMap,
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
        color_mode: ColorMode,
    ) -> Self {
        let glyph_buffer = create_glyph_buffer(device, INITIAL_BUFFER_CAPACITY);
        let bind_group = cache.create_atlas_bind_group(device, &glyph_buffer);

        Self {
            cache: cache.clone(),
            device: device.clone(),
            glyph_buffer,
            bind_group,
            format,
            color_mode,
            buffer_capacity: INITIAL_BUFFER_CAPACITY,
            buffer_cursor: 0,
            buffer_data: Vec::new(),
            scratch_curve_texels: Vec::new(),
            scratch_curve_locations: Vec::new(),
            scratch_band_entries: Vec::new(),
            glyphs: GlyphMap::new(),
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

    /// Buffer elements used (in vec4<i32> units).
    pub fn buffer_elements_used(&self) -> u32 {
        self.buffer_cursor
    }

    /// End-of-frame cache management.
    ///
    /// Clears per-frame usage tracking. When the buffer has grown beyond
    /// its initial size AND fewer than a quarter of cached glyphs are in use,
    /// performs a full reset: recreates buffer at initial size (reclaiming
    /// GPU memory immediately), clears the glyph cache. The next prepare()
    /// re-extracts only the visible glyphs.
    ///
    /// This is pressure-based: a stable document with many glyphs will not
    /// trigger reset as long as the textures haven't grown. Only when GPU
    /// memory has expanded (texture growth happened) AND the working set
    /// has shifted does eviction fire.
    pub fn trim(&mut self) {
        // Only consider reset when buffer has grown substantially.
        let substantial_growth = self.buffer_capacity > INITIAL_BUFFER_CAPACITY * 4;

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
        self.buffer_cursor = 0;
        self.buffer_data.clear();

        self.buffer_capacity = INITIAL_BUFFER_CAPACITY;
        self.glyph_buffer = create_glyph_buffer(&self.device, self.buffer_capacity);
        self.bind_group = self
            .cache
            .create_atlas_bind_group(&self.device, &self.glyph_buffer);
    }

    /// Upload a glyph's GPU-prepared outline and band data into the textures.
    /// Returns the GlyphEntry for vertex packing.
    #[hotpath::measure]
    pub(crate) fn upload_glyph(
        &mut self,
        device: &Device,
        queue: &Queue,
        gpu_outline: &GpuOutline,
        band_count_x: u32,
        band_count_y: u32,
    ) -> Result<GlyphEntry, crate::types::PrepareError> {
        let num_curves = gpu_outline.curves.len() as u32;

        // Reject glyphs with coordinates that would overflow i16 quantization.
        // i16 range ±32767 at 4 units/em → ±8191.75 font units.
        let [bmin_x, bmin_y, bmax_x, bmax_y] = gpu_outline.bounds;
        let max_coord = bmin_x.abs().max(bmin_y.abs()).max(bmax_x.abs()).max(bmax_y.abs());
        if max_coord * 4.0 > 32767.0 {
            return Ok(crate::glyph_cache::NON_VECTOR_GLYPH);
        }

        // Build curve texels with implicit p1 sharing within contours.
        // No row-boundary padding needed — storage buffer is 1D.
        self.scratch_curve_texels.clear();
        self.scratch_curve_texels.reserve(num_curves as usize * 2);
        self.scratch_curve_locations.clear();
        self.scratch_curve_locations.reserve(num_curves as usize);

        // Quantize f32 em-space coordinate to i32 at 4 units/em.
        let q = |v: f32| -> i32 { (v * 4.0).round() as i32 };

        for (i, curve) in gpu_outline.curves.iter().enumerate() {
            let is_continuation = i > 0
                && curve.p1 == gpu_outline.curves[i - 1].p3;

            if is_continuation {
                // Overwrite previous p3 texel's .zw with our p2
                let last = self.scratch_curve_texels.last_mut().expect("continuation curve must have preceding texel");
                last[2] = q(curve.p2[0]);
                last[3] = q(curve.p2[1]);
            } else {
                // New contour: emit fresh p12 texel
                self.scratch_curve_texels.push([
                    q(curve.p1[0]), q(curve.p1[1]), q(curve.p2[0]), q(curve.p2[1]),
                ]);
            }

            // Record curve location (0-based within curve data region)
            let curve_linear = self.scratch_curve_texels.len() as u32 - 1;
            self.scratch_curve_locations.push(CurveLocation {
                offset: curve_linear,
            });

            // Emit p3 texel
            self.scratch_curve_texels.push([q(curve.p3[0]), q(curve.p3[1]), 0, 0]);
        }
        let curve_element_count = self.scratch_curve_texels.len() as u32;

        // Build band data. Curve locations are 0-based within the curve region;
        // build_bands produces glyph-relative offsets for band headers.
        // We'll fixup curve ref offsets after we know the band data size.
        let band_data = crate::band::build_bands(
            gpu_outline,
            &self.scratch_curve_locations,
            band_count_x,
            band_count_y,
            std::mem::take(&mut self.scratch_band_entries),
        );
        let bd_count_x = band_data.band_count_x;
        let bd_count_y = band_data.band_count_y;
        let bd_transform = band_data.band_transform;
        let band_element_count = (band_data.entries.len() / 4) as u32;

        // Fixup curve ref offsets: add band_element_count so they point into
        // the curve region of the blob (which comes after band data).
        // Band entries layout: first num_headers * 4 i16 values are headers,
        // rest are curve refs (4 i16 values each, first is the offset).
        let num_headers = (bd_count_x + bd_count_y) as usize;
        let mut band_entries = band_data.entries;
        for ref_idx in num_headers..(band_element_count as usize) {
            let i = ref_idx * 4; // offset i16 is at position 0 of each texel
            // Decode biased offset, add band size, re-encode
            let raw_offset = band_entries[i] as i32 + 32768;
            let adjusted = raw_offset as u32 + band_element_count;
            band_entries[i] = (adjusted as i32 - 32768) as i16;
        }

        // Assemble glyph blob: [band_data] [curve_data]
        let blob_size = band_element_count + curve_element_count;

        // Overflow check
        if blob_size > 65535 {
            self.scratch_band_entries = band_entries;
            return Err(crate::types::PrepareError::AtlasFull);
        }

        // Check buffer capacity and grow if needed
        let glyph_offset = self.buffer_cursor;
        let new_end = glyph_offset + blob_size;
        if new_end > self.buffer_capacity {
            self.grow_buffer(device, queue, new_end);
        }

        // Widen band entries from i16 to i32 and append to CPU copy
        let band_i32: Vec<[i32; 4]> = band_entries
            .chunks_exact(4)
            .map(|c| [c[0] as i32, c[1] as i32, c[2] as i32, c[3] as i32])
            .collect();
        self.buffer_data.extend_from_slice(&band_i32);
        self.buffer_data.extend_from_slice(&self.scratch_curve_texels);

        // Upload blob to GPU
        let byte_offset = glyph_offset as u64 * 16; // 16 bytes per vec4<i32>
        let blob_bytes: &[u8] = bytemuck::cast_slice(
            &self.buffer_data[glyph_offset as usize..new_end as usize],
        );
        queue.write_buffer(&self.glyph_buffer, byte_offset, blob_bytes);

        // Reclaim scratch
        self.scratch_band_entries = band_entries;
        self.buffer_cursor = new_end;

        Ok(GlyphEntry {
            band_offset: glyph_offset,
            band_max_x: bd_count_x.saturating_sub(1),
            band_max_y: bd_count_y.saturating_sub(1),
            band_transform: bd_transform,
            bounds: gpu_outline.bounds,
            last_used_epoch: 0,
        })
    }

    fn grow_buffer(&mut self, device: &Device, queue: &Queue, min_capacity: u32) {
        let mut new_cap = self.buffer_capacity;
        while new_cap < min_capacity {
            new_cap *= 2;
        }
        // Use max to avoid multiple growths in one frame
        new_cap = new_cap.max(min_capacity);

        log::debug!(
            "Growing glyph buffer: {} → {} elements (cursor: {})",
            self.buffer_capacity,
            new_cap,
            self.buffer_cursor,
        );

        self.glyph_buffer = create_glyph_buffer(device, new_cap);
        self.buffer_capacity = new_cap;

        // Re-upload existing data
        if !self.buffer_data.is_empty() {
            queue.write_buffer(
                &self.glyph_buffer,
                0,
                bytemuck::cast_slice(&self.buffer_data),
            );
        }

        self.bind_group = self
            .cache
            .create_atlas_bind_group(device, &self.glyph_buffer);
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

fn create_glyph_buffer(device: &Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sluggrs glyph buffer"),
        size: capacity as u64 * 16, // 16 bytes per vec4<i32>
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
