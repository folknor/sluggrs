use crate::gpu_cache::Cache;
use crate::types::ColorMode;
use crate::BAND_TEXTURE_WIDTH;

use wgpu::{
    BindGroup, DepthStencilState, Device, Extent3d, MultisampleState, Queue, RenderPipeline,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

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

    // Texture dimensions
    pub(crate) curve_width: u32,
    pub(crate) band_height: u32,
}

const INITIAL_CURVE_WIDTH: u32 = 1024;
const INITIAL_BAND_HEIGHT: u32 = 1;

impl TextAtlas {
    pub fn new(
        device: &Device,
        queue: &Queue,
        cache: &Cache,
        format: TextureFormat,
    ) -> Self {
        Self::with_color_mode(device, queue, cache, format, ColorMode::Accurate)
    }

    pub fn with_color_mode(
        device: &Device,
        _queue: &Queue,
        cache: &Cache,
        format: TextureFormat,
        color_mode: ColorMode,
    ) -> Self {
        let curve_width = INITIAL_CURVE_WIDTH;
        let band_height = INITIAL_BAND_HEIGHT;

        let curve_texture = device.create_texture(&TextureDescriptor {
            label: Some("sluggrs curve texture"),
            size: Extent3d {
                width: curve_width,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let band_texture = device.create_texture(&TextureDescriptor {
            label: Some("sluggrs band texture"),
            size: Extent3d {
                width: BAND_TEXTURE_WIDTH,
                height: band_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Uint,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

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
            curve_width,
            band_height,
        }
    }

    pub fn trim(&mut self) {
        // TODO: clear glyphs_in_use set, retain cache.
        // For Phase A stub, this is a no-op.
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
