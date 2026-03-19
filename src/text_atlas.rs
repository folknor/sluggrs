use crate::band::CurveLocation;
use crate::glyph_cache::{GlyphEntry, GlyphKey, GlyphMap, NON_VECTOR_GLYPH};
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
    pub(crate) curve_texture: wgpu::Texture,
    pub(crate) curve_view: TextureView,
    pub(crate) band_texture: wgpu::Texture,
    pub(crate) band_view: TextureView,
    pub(crate) bind_group: BindGroup,
    pub(crate) format: TextureFormat,
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
            glyphs: GlyphMap::new(),
        }
    }

    pub fn trim(&mut self) {
        // Match cryoglyph semantics: retain cached data.
        // For now this is a no-op since we don't track per-frame usage yet.
    }

    /// Upload a glyph's GPU-prepared outline and band data into the textures.
    /// Returns the GlyphEntry for vertex packing.
    pub(crate) fn upload_glyph(
        &mut self,
        device: &Device,
        queue: &Queue,
        gpu_outline: &GpuOutline,
        band_count_x: u32,
        band_count_y: u32,
    ) -> GlyphEntry {
        let num_curves = gpu_outline.curves.len() as u32;

        // Build curve texels (2 texels per curve)
        let curve_start = self.curve_cursor;
        let mut curve_texels = Vec::with_capacity(num_curves as usize * 2);
        for curve in &gpu_outline.curves {
            curve_texels.push([curve.p1[0], curve.p1[1], curve.p2[0], curve.p2[1]]);
            curve_texels.push([curve.p3[0], curve.p3[1], 0.0, 0.0]);
        }
        let curve_texel_count = curve_texels.len() as u32;

        // Build curve locations as 2D coordinates (wrapping at CURVE_TEXTURE_WIDTH)
        let curve_locations: Vec<CurveLocation> = (0..num_curves)
            .map(|i| {
                let linear = curve_start + i * 2;
                CurveLocation {
                    x: linear % CURVE_TEXTURE_WIDTH,
                    y: linear / CURVE_TEXTURE_WIDTH,
                }
            })
            .collect();

        // Build band data with absolute 2D curve locations
        let band_start = self.band_cursor;
        let band_data =
            crate::band::build_bands(gpu_outline, &curve_locations, band_count_x, band_count_y);
        let mut band_texels = Vec::new();
        for chunk in band_data.entries.chunks(4) {
            let mut texel = [0u32; 4];
            for (i, &val) in chunk.iter().enumerate() {
                texel[i] = val;
            }
            band_texels.push(texel);
        }
        let band_texel_count = band_texels.len() as u32;

        // Grow textures if needed
        let new_curve_end = self.curve_cursor + curve_texel_count;
        let required_curve_height = (new_curve_end + CURVE_TEXTURE_WIDTH - 1) / CURVE_TEXTURE_WIDTH;
        if required_curve_height > self.curve_height {
            self.grow_curve_texture(device, queue, required_curve_height);
        }

        let new_band_end = self.band_cursor + band_texel_count;
        let required_band_height = (new_band_end + BAND_TEXTURE_WIDTH - 1) / BAND_TEXTURE_WIDTH;
        if required_band_height > self.band_height {
            self.grow_band_texture(device, queue, required_band_height);
        }

        // Append to CPU-side copies
        self.curve_data.extend_from_slice(&curve_texels);
        self.band_data.extend_from_slice(&band_texels);

        // Upload curve texels (handling wrapping across rows)
        if !curve_texels.is_empty() {
            self.upload_wrapped_texels_f32(
                queue,
                &self.curve_texture,
                &curve_texels,
                self.curve_cursor,
                CURVE_TEXTURE_WIDTH,
            );
        }

        // Upload band texels (handling wrapping across rows)
        if !band_texels.is_empty() {
            upload_wrapped_texels_u32(
                queue,
                &self.band_texture,
                &band_texels,
                self.band_cursor,
                BAND_TEXTURE_WIDTH,
            );
        }

        // Advance cursors
        self.curve_cursor = new_curve_end;
        self.band_cursor = new_band_end;

        GlyphEntry {
            band_offset: band_start,
            band_max_x: band_data.band_count_x.saturating_sub(1),
            band_max_y: band_data.band_count_y.saturating_sub(1),
            band_transform: band_data.band_transform,
            bounds: gpu_outline.bounds,
        }
    }

    /// Mark a glyph as non-vector (no outline available).
    pub(crate) fn mark_non_vector(&mut self, key: GlyphKey) {
        self.glyphs.insert(key, NON_VECTOR_GLYPH);
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
