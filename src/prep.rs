//! Pure CPU-side glyph blob preparation, separated from atlas commit.
//!
//! The mono / COLRv0 paths split into two phases:
//! 1. **Prepare** (this module): outline → packed i32 blob + metadata.
//!    Pure, scratch-buffer-driven, parallelizable across glyphs.
//! 2. **Commit** (`text_atlas`): blob → atlas buffer + cursor advance.
//!    Serial; touches GPU-tracked state and may grow the storage buffer.

use crate::band::{BandScratch, CurveLocation, build_bands};
use crate::outline::GlyphOutline;

/// Per-worker scratch reused across `prepare_mono` calls. Hold one per
/// rayon worker (or one total when serial). All buffers are `clear()`-ed
/// at the start of each call; capacity is retained.
#[derive(Default)]
pub struct PrepScratch {
    curve_locations: Vec<CurveLocation>,
    band_entries: Vec<i16>,
    band_scratch: BandScratch,
}

/// CPU-prepared mono glyph blob, ready for `TextAtlas::commit_mono`.
pub struct PreparedMono {
    pub bounds: [f32; 4],
    pub units_per_em: f32,
    /// Packed blob: `[band texels...] [curve texels...]`. 2 i32 per texel
    /// (each i32 holds an i16 pair).
    pub blob_data: Vec<i32>,
    /// Total texel count (band_element_count + curve_element_count).
    pub blob_size: u32,
    pub band_count_x: u32,
    pub band_count_y: u32,
    pub band_transform: [f32; 4],
}

/// Pack two i16 values into one i32 (low = a, high = b). Mirrors the
/// shader's `unpack_lo/unpack_hi`.
#[inline]
pub fn pack_i16_pair(a: i16, b: i16) -> i32 {
    (a as u16 as u32 | ((b as u16 as u32) << 16)) as i32
}

#[inline]
fn quantize(v: f32) -> i32 {
    (v * 4.0).round() as i32
}

/// Prepare a mono glyph blob from an outline. Returns `None` if any
/// quantized coordinate would overflow i16 (caller maps to NON_VECTOR_GLYPH).
pub fn prepare_mono(
    outline: &GlyphOutline,
    band_count_x: u32,
    band_count_y: u32,
    units_per_em: f32,
    scratch: &mut PrepScratch,
) -> Option<PreparedMono> {
    let [bmin_x, bmin_y, bmax_x, bmax_y] = outline.bounds;
    let max_coord = bmin_x
        .abs()
        .max(bmin_y.abs())
        .max(bmax_x.abs())
        .max(bmax_y.abs());
    if max_coord * 4.0 > 32767.0 {
        return None;
    }

    let num_curves = outline.curves.len();
    scratch.curve_locations.clear();
    scratch.curve_locations.reserve(num_curves);

    let mut curve_texels: Vec<[i32; 4]> = Vec::with_capacity(num_curves * 2);
    for (i, curve) in outline.curves.iter().enumerate() {
        let is_continuation = i > 0 && curve.p1 == outline.curves[i - 1].p3;
        if is_continuation {
            let last_idx = curve_texels.len() - 1;
            curve_texels[last_idx][2] = quantize(curve.p2[0]);
            curve_texels[last_idx][3] = quantize(curve.p2[1]);
        } else {
            curve_texels.push([
                quantize(curve.p1[0]),
                quantize(curve.p1[1]),
                quantize(curve.p2[0]),
                quantize(curve.p2[1]),
            ]);
        }
        let curve_linear = curve_texels.len() as u32 - 1;
        scratch.curve_locations.push(CurveLocation { offset: curve_linear });
        curve_texels.push([quantize(curve.p3[0]), quantize(curve.p3[1]), 0, 0]);
    }
    let curve_element_count = curve_texels.len() as u32;

    let band_data = build_bands(
        outline,
        &scratch.curve_locations,
        band_count_x,
        band_count_y,
        std::mem::take(&mut scratch.band_entries),
        &mut scratch.band_scratch,
    );
    let band_entries = band_data.entries;
    let band_element_count = (band_entries.len() / 4) as u32;
    let blob_size = band_element_count + curve_element_count;

    let mut blob_data: Vec<i32> = Vec::with_capacity(blob_size as usize * 2);
    for c in band_entries.chunks_exact(4) {
        blob_data.push(pack_i16_pair(c[0], c[1]));
        blob_data.push(pack_i16_pair(c[2], c[3]));
    }
    for v in &curve_texels {
        blob_data.push(pack_i16_pair(v[0] as i16, v[1] as i16));
        blob_data.push(pack_i16_pair(v[2] as i16, v[3] as i16));
    }

    // Reclaim the band_entries allocation for the next call.
    let mut reclaimed = band_entries;
    reclaimed.clear();
    scratch.band_entries = reclaimed;

    Some(PreparedMono {
        bounds: outline.bounds,
        units_per_em,
        blob_data,
        blob_size,
        band_count_x: band_data.band_count_x,
        band_count_y: band_data.band_count_y,
        band_transform: band_data.band_transform,
    })
}
