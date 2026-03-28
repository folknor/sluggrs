use crate::band::CurveLocation;
use crate::glyph_cache::{GlyphEntry, GlyphMap};
use crate::gpu_cache::Cache;
use crate::prepare::GpuOutline;
use crate::types::ColorMode;
use crate::BAND_TEXTURE_WIDTH;

use wgpu::{
    BindGroup, DepthStencilState, Device, Extent3d, MultisampleState, Queue, RenderPipeline,
    TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

/// Curve texture width. Fixed like the band texture — rows wrap at this boundary.
const CURVE_TEXTURE_WIDTH: u32 = 4096;
const INITIAL_CURVE_HEIGHT: u32 = 1;
const INITIAL_BAND_HEIGHT: u32 = 1;

/// An atlas containing cached glyph curve and band data for GPU rendering.
pub struct TextAtlas {
    pub(crate) cache: Cache,
    device: Device,
    pub(crate) curve_texture: wgpu::Texture,
    pub(crate) curve_view: TextureView,
    pub(crate) band_texture: wgpu::Texture,
    pub(crate) band_view: TextureView,
    pub(crate) bind_group: BindGroup,
    pub(crate) format: TextureFormat,
    #[allow(dead_code)] // API contract — read by iced integration
    pub(crate) color_mode: ColorMode,

    // Texture dimensions (height grows, width is fixed)
    curve_height: u32,
    band_height: u32,

    // Write cursors (linear texel offsets, append-only)
    curve_cursor: u32,
    band_cursor: u32,

    // CPU-side copies for re-upload on texture growth
    curve_data: Vec<[f32; 4]>,
    band_data: Vec<[u32; 4]>,

    // Scratch buffers reused across upload_glyph() calls
    scratch_curve_texels: Vec<[f32; 4]>,
    scratch_curve_locations: Vec<CurveLocation>,
    scratch_band_entries: Vec<u32>,

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
        let curve_height = INITIAL_CURVE_HEIGHT;
        let band_height = INITIAL_BAND_HEIGHT;

        let curve_texture = create_curve_texture(device, curve_height);
        let band_texture = create_band_texture(device, band_height);

        let curve_view = curve_texture.create_view(&TextureViewDescriptor::default());
        let band_view = band_texture.create_view(&TextureViewDescriptor::default());
        let bind_group = cache.create_atlas_bind_group(device, &curve_view, &band_view);

        Self {
            cache: cache.clone(),
            device: device.clone(),
            curve_texture,
            curve_view,
            band_texture,
            band_view,
            bind_group,
            format,
            color_mode,
            curve_height,
            band_height,
            curve_cursor: 0,
            band_cursor: 0,
            curve_data: Vec::new(),
            band_data: Vec::new(),
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

    /// Linear texel offset into the curve texture (append-only cursor).
    pub fn curve_texels_used(&self) -> u32 {
        self.curve_cursor
    }

    /// Linear texel offset into the band texture (append-only cursor).
    pub fn band_texels_used(&self) -> u32 {
        self.band_cursor
    }

    /// End-of-frame cache management.
    ///
    /// Clears per-frame usage tracking. When the textures have grown beyond
    /// their initial size AND fewer than a quarter of cached glyphs are in use,
    /// performs a full reset: recreates textures at initial size (reclaiming
    /// GPU memory immediately), clears the glyph cache. The next prepare()
    /// re-extracts only the visible glyphs.
    ///
    /// This is pressure-based: a stable document with many glyphs will not
    /// trigger reset as long as the textures haven't grown. Only when GPU
    /// memory has expanded (texture growth happened) AND the working set
    /// has shifted does eviction fire.
    pub fn trim(&mut self) {
        // Only consider reset when textures have grown substantially.
        // Minimum 16 rows means ~1MB+ of GPU memory before we bother
        // with eviction — avoids churn for modest working sets.
        let substantial_growth = self.curve_height >= 16 || self.band_height >= 16;

        if substantial_growth {
            let cached = self.glyphs.len();
            let in_use = self.glyphs.in_use_count();

            if cached > 0 && in_use < cached / 4 {
                self.reset_atlas();
            } else {
                log::trace!(
                    "trim: retained ({in_use}/{cached} glyphs in use, \
                     curve={}x{} band={}x{})",
                    CURVE_TEXTURE_WIDTH, self.curve_height,
                    BAND_TEXTURE_WIDTH, self.band_height,
                );
            }
        }

        self.glyphs.next_frame();
    }

    /// Full atlas reset: recreate textures at initial size, clear all caches.
    /// GPU memory is reclaimed immediately (old textures are dropped).
    fn reset_atlas(&mut self) {
        log::debug!(
            "trim: resetting atlas ({}/{} glyphs in use, \
             curve={}x{} band={}x{} texels)",
            self.glyphs.in_use_count(),
            self.glyphs.len(),
            CURVE_TEXTURE_WIDTH,
            self.curve_height,
            BAND_TEXTURE_WIDTH,
            self.band_height,
        );

        self.glyphs.clear();
        self.curve_cursor = 0;
        self.band_cursor = 0;
        self.curve_data.clear();
        self.band_data.clear();

        // Recreate GPU textures at initial size — old textures are dropped,
        // freeing their GPU allocations immediately.
        self.curve_height = INITIAL_CURVE_HEIGHT;
        self.band_height = INITIAL_BAND_HEIGHT;
        self.curve_texture = create_curve_texture(&self.device, self.curve_height);
        self.band_texture = create_band_texture(&self.device, self.band_height);
        self.curve_view = self.curve_texture.create_view(&TextureViewDescriptor::default());
        self.band_view = self.band_texture.create_view(&TextureViewDescriptor::default());
        self.bind_group =
            self.cache
                .create_atlas_bind_group(&self.device, &self.curve_view, &self.band_view);
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

        // Build curve texels (2 texels per curve) — reuse scratch buffer
        let curve_start = self.curve_cursor;
        self.scratch_curve_texels.clear();
        self.scratch_curve_texels.reserve(num_curves as usize * 2);
        for curve in &gpu_outline.curves {
            self.scratch_curve_texels.push([curve.p1[0], curve.p1[1], curve.p2[0], curve.p2[1]]);
            self.scratch_curve_texels.push([curve.p3[0], curve.p3[1], 0.0, 0.0]);
        }
        let curve_texel_count = self.scratch_curve_texels.len() as u32;

        // Build curve locations as 2D coordinates — reuse scratch buffer
        self.scratch_curve_locations.clear();
        self.scratch_curve_locations.reserve(num_curves as usize);
        for i in 0..num_curves {
            let linear = curve_start + i * 2;
            self.scratch_curve_locations.push(CurveLocation {
                x: linear % CURVE_TEXTURE_WIDTH,
                y: linear / CURVE_TEXTURE_WIDTH,
            });
        }

        // Build band data with absolute 2D curve locations
        let band_start = self.band_cursor;
        let band_data =
            crate::band::build_bands(gpu_outline, &self.scratch_curve_locations, band_count_x, band_count_y, std::mem::take(&mut self.scratch_band_entries));
        let bd_count_x = band_data.band_count_x;
        let bd_count_y = band_data.band_count_y;
        let bd_transform = band_data.band_transform;
        let band_texel_count = (band_data.entries.len() / 4) as u32;

        // Check device limits before any mutation
        let max_dim = device.limits().max_texture_dimension_2d;
        let new_curve_end = self.curve_cursor + curve_texel_count;
        let required_curve_height = new_curve_end.div_ceil(CURVE_TEXTURE_WIDTH);
        let new_band_end = self.band_cursor + band_texel_count;
        let required_band_height = new_band_end.div_ceil(BAND_TEXTURE_WIDTH);

        if required_curve_height > max_dim || required_band_height > max_dim {
            // Reclaim the allocation before returning
            self.scratch_band_entries = band_data.entries;
            return Err(crate::types::PrepareError::AtlasFull);
        }

        // Grow textures if needed
        if required_curve_height > self.curve_height {
            self.grow_curve_texture(device, queue, required_curve_height);
        }
        if required_band_height > self.band_height {
            self.grow_band_texture(device, queue, required_band_height);
        }

        // Access band texels from the returned BandData
        let band_texels: &[[u32; 4]] = bytemuck::cast_slice(&band_data.entries);

        // Append to CPU-side copies
        self.curve_data.extend_from_slice(&self.scratch_curve_texels);
        self.band_data.extend_from_slice(band_texels);

        // Upload curve texels (handling wrapping across rows)
        if !self.scratch_curve_texels.is_empty() {
            self.upload_wrapped_texels_f32(
                queue,
                &self.curve_texture,
                &self.scratch_curve_texels,
                self.curve_cursor,
                CURVE_TEXTURE_WIDTH,
            );
        }

        // Upload band texels (handling wrapping across rows)
        if !band_texels.is_empty() {
            upload_wrapped_texels_u32(
                queue,
                &self.band_texture,
                band_texels,
                self.band_cursor,
                BAND_TEXTURE_WIDTH,
            );
        }

        // Reclaim the scratch allocation for reuse (after all borrows of band_data.entries)
        self.scratch_band_entries = band_data.entries;

        // Advance cursors
        self.curve_cursor = new_curve_end;
        self.band_cursor = new_band_end;

        Ok(GlyphEntry {
            band_offset: band_start,
            band_max_x: bd_count_x.saturating_sub(1),
            band_max_y: bd_count_y.saturating_sub(1),
            band_transform: bd_transform,
            bounds: gpu_outline.bounds,
            last_used_epoch: 0,
        })
    }


    /// Upload f32 texels at a linear offset, handling row wrapping.
    fn upload_wrapped_texels_f32(
        &self,
        queue: &Queue,
        texture: &wgpu::Texture,
        texels: &[[f32; 4]],
        linear_offset: u32,
        tex_width: u32,
    ) {
        let mut remaining = texels;
        let mut offset = linear_offset;

        while !remaining.is_empty() {
            let x = offset % tex_width;
            let y = offset / tex_width;
            let row_remaining = (tex_width - x) as usize;
            let count = remaining.len().min(row_remaining);

            queue.write_texture(
                TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: TextureAspect::All,
                },
                bytemuck::cast_slice(&remaining[..count]),
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(count as u32 * 16),
                    rows_per_image: None,
                },
                Extent3d {
                    width: count as u32,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );

            remaining = &remaining[count..];
            offset += count as u32;
        }
    }

    fn grow_curve_texture(&mut self, device: &Device, queue: &Queue, min_height: u32) {
        let mut new_height = self.curve_height;
        while new_height < min_height {
            new_height *= 2;
        }

        log::debug!(
            "Growing curve texture: {}x{} → {}x{} (cursor: {})",
            CURVE_TEXTURE_WIDTH,
            self.curve_height,
            CURVE_TEXTURE_WIDTH,
            new_height,
            self.curve_cursor
        );

        self.curve_texture = create_curve_texture(device, new_height);
        self.curve_view = self
            .curve_texture
            .create_view(&TextureViewDescriptor::default());
        self.curve_height = new_height;

        // Re-upload existing data
        if !self.curve_data.is_empty() {
            self.upload_wrapped_texels_f32(
                queue,
                &self.curve_texture,
                &self.curve_data,
                0,
                CURVE_TEXTURE_WIDTH,
            );
        }

        self.rebind(device);
    }

    fn grow_band_texture(&mut self, device: &Device, queue: &Queue, min_height: u32) {
        let mut new_height = self.band_height;
        while new_height < min_height {
            new_height *= 2;
        }

        log::debug!(
            "Growing band texture: {}x{} → {}x{} (cursor: {})",
            BAND_TEXTURE_WIDTH,
            self.band_height,
            BAND_TEXTURE_WIDTH,
            new_height,
            self.band_cursor
        );

        self.band_texture = create_band_texture(device, new_height);
        self.band_view = self
            .band_texture
            .create_view(&TextureViewDescriptor::default());
        self.band_height = new_height;

        // Re-upload existing data
        if !self.band_data.is_empty() {
            upload_wrapped_texels_u32(
                queue,
                &self.band_texture,
                &self.band_data,
                0,
                BAND_TEXTURE_WIDTH,
            );
        }

        self.rebind(device);
    }

    fn rebind(&mut self, device: &Device) {
        self.bind_group =
            self.cache
                .create_atlas_bind_group(device, &self.curve_view, &self.band_view);
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

fn upload_wrapped_texels_u32(
    queue: &Queue,
    texture: &wgpu::Texture,
    texels: &[[u32; 4]],
    linear_offset: u32,
    tex_width: u32,
) {
    let mut remaining = texels;
    let mut offset = linear_offset;

    while !remaining.is_empty() {
        let x = offset % tex_width;
        let y = offset / tex_width;
        let row_remaining = (tex_width - x) as usize;
        let count = remaining.len().min(row_remaining);

        queue.write_texture(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(&remaining[..count]),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(count as u32 * 16),
                rows_per_image: None,
            },
            Extent3d {
                width: count as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        remaining = &remaining[count..];
        offset += count as u32;
    }
}

fn create_curve_texture(device: &Device, height: u32) -> wgpu::Texture {
    device.create_texture(&TextureDescriptor {
        label: Some("sluggrs curve texture"),
        size: Extent3d {
            width: CURVE_TEXTURE_WIDTH,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn create_band_texture(device: &Device, height: u32) -> wgpu::Texture {
    device.create_texture(&TextureDescriptor {
        label: Some("sluggrs band texture"),
        size: Extent3d {
            width: BAND_TEXTURE_WIDTH,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Uint,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    })
}
