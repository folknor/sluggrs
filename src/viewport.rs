use crate::gpu_cache::{Cache, Params};
use crate::types::Resolution;

use wgpu::{BindGroup, Buffer, BufferDescriptor, BufferUsages, Device, Queue};

use std::mem;

#[derive(Debug)]
pub struct Viewport {
    params: Params,
    params_buffer: Buffer,
    pub(crate) bind_group: BindGroup,
}

impl Viewport {
    pub fn new(device: &Device, cache: &Cache) -> Self {
        let params = Params {
            screen_size: [0.0, 0.0],
            scroll_offset: [0.0, 0.0],
            flags: 1, // MSAA+stem darkening on by default
            _pad: 0,
        };

        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sluggrs params"),
            size: mem::size_of::<Params>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = cache.create_uniforms_bind_group(device, &params_buffer);

        Self {
            params,
            params_buffer,
            bind_group,
        }
    }

    pub fn update(&mut self, queue: &Queue, resolution: Resolution) {
        let new_size = [resolution.width as f32, resolution.height as f32];
        if self.params.screen_size != new_size {
            self.params.screen_size = new_size;
            queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        }
    }

    /// The bind group for the viewport/params uniform buffer.
    /// Shared with the raster fallback pipeline.
    pub fn bind_group(&self) -> &BindGroup {
        &self.bind_group
    }

    pub fn resolution(&self) -> Resolution {
        Resolution {
            width: self.params.screen_size[0] as u32,
            height: self.params.screen_size[1] as u32,
        }
    }
}
