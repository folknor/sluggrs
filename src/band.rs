/// Band acceleration structure builder for Slug rendering.
///
/// Divides the glyph bounding box into horizontal and vertical bands,
/// recording which curves intersect each band. This lets the fragment
/// shader skip curves that can't affect the current pixel.
use crate::outline::GlyphOutline;

/// Reusable scratch buffers for `build_bands()`.
/// Avoids 14 per-glyph allocations by clearing and reusing across calls.
/// Per-curve metadata cached in Phase 1 to avoid redundant recomputation.
#[derive(Clone, Copy)]
struct CurveMeta {
    hband_min: usize,
    hband_max: usize,
    vband_min: usize,
    vband_max: usize,
    is_horizontal: bool,
    is_vertical: bool,
}

#[derive(Default)]
pub struct BandScratch {
    curve_meta: Vec<CurveMeta>,
    max_x_keys: Vec<f32>,
    max_y_keys: Vec<f32>,
    min_x_keys: Vec<f32>,
    min_y_keys: Vec<f32>,
    hband_counts: Vec<u32>,
    vband_counts: Vec<u32>,
    hband_offsets: Vec<u32>,
    vband_offsets: Vec<u32>,
    desc_indices: Vec<usize>,
    asc_indices: Vec<usize>,
    hband_fill: Vec<u32>,
    vband_fill: Vec<u32>,
    hband_splits: Vec<f32>,
    vband_splits: Vec<f32>,
}

/// Band data ready for GPU upload.
pub struct BandData {
    /// Entries in the band texture as i16 texels: headers + curve indices.
    /// Pass this Vec back via `build_bands` to reuse its allocation.
    pub entries: Vec<i16>,
    /// Number of horizontal bands
    pub band_count_x: u32,
    /// Number of vertical bands
    pub band_count_y: u32,
    /// Transform from em-space to band index: band_idx = coord * scale + offset
    pub band_transform: [f32; 4], // scale_x, scale_y, offset_x, offset_y
}

/// Curve location as a linear offset in the glyph blob.
#[derive(Debug, Clone, Copy)]
pub struct CurveLocation {
    pub offset: u32,
}

/// Encode a glyph-relative offset as i16.
/// Values up to 65535 are stored by reinterpreting the low 16 bits as i16.
/// The shader recovers the original value with a mask: `u32(v) & 0xFFFF`.
fn encode_offset(offset: u32) -> i16 {
    offset as u16 as i16
}

/// Find the optimal split coordinate for a band's dual sorted lists.
/// Walks the descending list, tracking a monotone pointer into the ascending
/// list, and picks the split that minimizes max(left_count, right_count).
fn find_split(
    desc_indices: &[usize],
    asc_indices: &[usize],
    max_keys: &[f32],
    min_keys: &[f32],
    bounds_min: f32,
    bounds_max: f32,
) -> f32 {
    let n = desc_indices.len();
    if n == 0 {
        return (bounds_min + bounds_max) * 0.5;
    }

    let mut best_worst = n;
    let mut best_split = (bounds_min + bounds_max) * 0.5;
    let mut left_ptr = n;

    for ci in 0..n {
        let split = max_keys[desc_indices[ci]];
        let right_count = ci + 1;

        // Shrink left_ptr: remove curves from the end of asc list that have min > split
        while left_ptr > 0 && min_keys[asc_indices[left_ptr - 1]] > split {
            left_ptr -= 1;
        }
        let left_count = left_ptr;

        let worst = right_count.max(left_count);
        if worst < best_worst {
            best_worst = worst;
            best_split = split;
        }
    }

    best_split
}

/// Build the band acceleration structure for a glyph.
///
/// Produces dual sorted curve lists (descending by max, ascending by min)
/// with a split point per band for direction-aware shader early exit.
/// `curve_locations` maps each curve index to its (x, y) position in the curve texture.
#[hotpath::measure]
pub fn build_bands(
    outline: &GlyphOutline,
    curve_locations: &[CurveLocation],
    band_count_x: u32,
    band_count_y: u32,
    mut scratch_entries: Vec<i16>,
    scratch: &mut BandScratch,
) -> BandData {
    let [min_x, min_y, max_x, max_y] = outline.bounds;
    let width = max_x - min_x;
    let height = max_y - min_y;

    // Avoid division by zero for degenerate glyphs
    let safe_width = if width < 1e-6 { 1.0 } else { width };
    let safe_height = if height < 1e-6 { 1.0 } else { height };

    let scale_x = band_count_x as f32 / safe_width;
    let scale_y = band_count_y as f32 / safe_height;
    let offset_x = -min_x * scale_x;
    let offset_y = -min_y * scale_y;

    const BAND_EPSILON: f32 = 1e-5;
    let hcount = band_count_y as usize;
    let vcount = band_count_x as usize;

    // Clear and reuse scratch buffers
    let num_curves = outline.curves.len();
    scratch.curve_meta.clear();
    scratch.curve_meta.reserve(num_curves);
    scratch.max_x_keys.clear();
    scratch.max_y_keys.clear();
    scratch.min_x_keys.clear();
    scratch.min_y_keys.clear();
    scratch.max_x_keys.reserve(num_curves);
    scratch.max_y_keys.reserve(num_curves);
    scratch.min_x_keys.reserve(num_curves);
    scratch.min_y_keys.reserve(num_curves);

    // Phase 1: compute per-curve metadata and count curves per band.
    // CurveMeta caches min/max band indices and axis-alignment flags so
    // Phase 2 can skip recomputing them.
    scratch.hband_counts.clear();
    scratch.hband_counts.resize(hcount, 0);
    scratch.vband_counts.clear();
    scratch.vband_counts.resize(vcount, 0);

    for curve in &outline.curves {
        let curve_min_y = curve.p1[1].min(curve.p2[1]).min(curve.p3[1]);
        let curve_max_y = curve.p1[1].max(curve.p2[1]).max(curve.p3[1]);
        let curve_min_x = curve.p1[0].min(curve.p2[0]).min(curve.p3[0]);
        let curve_max_x = curve.p1[0].max(curve.p2[0]).max(curve.p3[0]);

        scratch.max_x_keys.push(curve_max_x);
        scratch.max_y_keys.push(curve_max_y);
        scratch.min_x_keys.push(curve_min_x);
        scratch.min_y_keys.push(curve_min_y);

        // Axis-aligned curve filtering (matching harfbuzz hb-gpu-draw.cc:361-391):
        // A horizontal curve (all y equal) never crosses a horizontal ray → skip hbands.
        // A vertical curve (all x equal) never crosses a vertical ray → skip vbands.
        let is_horizontal = curve_min_y == curve_max_y;
        let is_vertical = curve_min_x == curve_max_x;

        let mut hband_min = 0;
        let mut hband_max = 0;
        if !is_horizontal {
            hband_min = (curve_min_y * scale_y + offset_y)
                .floor()
                .clamp(0.0, hcount as f32 - 1.0) as usize;
            hband_max = ((curve_max_y * scale_y + offset_y - BAND_EPSILON).floor())
                .clamp(0.0, hcount as f32 - 1.0) as usize;
            for count in &mut scratch.hband_counts[hband_min..=hband_max] {
                *count += 1;
            }
        }

        let mut vband_min = 0;
        let mut vband_max = 0;
        if !is_vertical {
            vband_min = (curve_min_x * scale_x + offset_x)
                .floor()
                .clamp(0.0, vcount as f32 - 1.0) as usize;
            vband_max = ((curve_max_x * scale_x + offset_x - BAND_EPSILON).floor())
                .clamp(0.0, vcount as f32 - 1.0) as usize;
            for count in &mut scratch.vband_counts[vband_min..=vband_max] {
                *count += 1;
            }
        }

        scratch.curve_meta.push(CurveMeta {
            hband_min,
            hband_max,
            vband_min,
            vband_max,
            is_horizontal,
            is_vertical,
        });
    }

    // Phase 2: build dual sorted lists (desc by max, asc by min) for each band.
    // Both lists contain the same curves, just in different order.
    let htotal: u32 = scratch.hband_counts.iter().sum();
    let vtotal: u32 = scratch.vband_counts.iter().sum();
    let total_refs = (htotal + vtotal) as usize;

    scratch.hband_offsets.clear();
    let mut offset = 0u32;
    for &count in &scratch.hband_counts {
        scratch.hband_offsets.push(offset);
        offset += count;
    }
    scratch.vband_offsets.clear();
    for &count in &scratch.vband_counts {
        scratch.vband_offsets.push(offset);
        offset += count;
    }

    // Two flat arrays: desc (sorted by descending max) and asc (sorted by ascending min)
    scratch.desc_indices.clear();
    scratch.desc_indices.resize(total_refs, 0);
    scratch.asc_indices.clear();
    scratch.asc_indices.resize(total_refs, 0);
    scratch.hband_fill.clear();
    scratch.hband_fill.extend_from_slice(&scratch.hband_offsets);
    scratch.vband_fill.clear();
    scratch.vband_fill.extend_from_slice(&scratch.vband_offsets);

    for (i, meta) in scratch.curve_meta.iter().enumerate() {
        if !meta.is_horizontal {
            for fill in &mut scratch.hband_fill[meta.hband_min..=meta.hband_max] {
                let idx = *fill as usize;
                scratch.desc_indices[idx] = i;
                scratch.asc_indices[idx] = i;
                *fill += 1;
            }
        }

        if !meta.is_vertical {
            for fill in &mut scratch.vband_fill[meta.vband_min..=meta.vband_max] {
                let idx = *fill as usize;
                scratch.desc_indices[idx] = i;
                scratch.asc_indices[idx] = i;
                *fill += 1;
            }
        }
    }

    // Sort desc by descending max, asc by ascending min
    for b in 0..hcount {
        let start = scratch.hband_offsets[b] as usize;
        let end = start + scratch.hband_counts[b] as usize;
        scratch.desc_indices[start..end].sort_unstable_by(|&a, &b| {
            scratch.max_x_keys[b]
                .partial_cmp(&scratch.max_x_keys[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scratch.asc_indices[start..end].sort_unstable_by(|&a, &b| {
            scratch.min_x_keys[a]
                .partial_cmp(&scratch.min_x_keys[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    for b in 0..vcount {
        let start = scratch.vband_offsets[b] as usize;
        let end = start + scratch.vband_counts[b] as usize;
        scratch.desc_indices[start..end].sort_unstable_by(|&a, &b| {
            scratch.max_y_keys[b]
                .partial_cmp(&scratch.max_y_keys[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scratch.asc_indices[start..end].sort_unstable_by(|&a, &b| {
            scratch.min_y_keys[a]
                .partial_cmp(&scratch.min_y_keys[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Find optimal split per band. The split minimizes max(left_count, right_count)
    // where left/right are determined by fragment position relative to the split.
    scratch.hband_splits.clear();
    for b in 0..hcount {
        let start = scratch.hband_offsets[b] as usize;
        let end = start + scratch.hband_counts[b] as usize;
        scratch.hband_splits.push(find_split(
            &scratch.desc_indices[start..end],
            &scratch.asc_indices[start..end],
            &scratch.max_x_keys,
            &scratch.min_x_keys,
            outline.bounds[0],
            outline.bounds[2],
        ));
    }
    scratch.vband_splits.clear();
    for b in 0..vcount {
        let start = scratch.vband_offsets[b] as usize;
        let end = start + scratch.vband_counts[b] as usize;
        scratch.vband_splits.push(find_split(
            &scratch.desc_indices[start..end],
            &scratch.asc_indices[start..end],
            &scratch.max_y_keys,
            &scratch.min_y_keys,
            outline.bounds[1],
            outline.bounds[3],
        ));
    }

    // Build GPU texture data.
    // Layout: [headers...] [band0_desc, band0_asc, band1_desc, band1_asc, ...]
    // Header: (count, desc_offset, asc_offset, split_bits)
    let num_headers = band_count_y + band_count_x;
    let curve_lists_start = num_headers;
    // Pre-compute band_element_count so curve refs can be written with final offsets
    let band_element_count = num_headers + (total_refs * 2) as u32;

    scratch_entries.clear();
    scratch_entries.reserve((band_element_count as usize) * 4);

    // Write horizontal band headers (biased offsets, quantized split)
    let mut texel_offset = curve_lists_start;
    for b in 0..hcount {
        let count = scratch.hband_counts[b];
        scratch_entries.push(count as i16);
        scratch_entries.push(encode_offset(texel_offset)); // desc_offset
        scratch_entries.push(encode_offset(texel_offset + count)); // asc_offset
        scratch_entries.push((scratch.hband_splits[b] * 4.0).round() as i16);
        texel_offset += count * 2; // desc + asc
    }
    // Write vertical band headers
    for b in 0..vcount {
        let count = scratch.vband_counts[b];
        scratch_entries.push(count as i16);
        scratch_entries.push(encode_offset(texel_offset)); // desc_offset
        scratch_entries.push(encode_offset(texel_offset + count)); // asc_offset
        scratch_entries.push((scratch.vband_splits[b] * 4.0).round() as i16);
        texel_offset += count * 2;
    }

    // Write curve references: desc then asc for each band
    // Curve offsets are biased and include the curve_data region's base.
    for b in 0..hcount {
        let start = scratch.hband_offsets[b] as usize;
        let end = start + scratch.hband_counts[b] as usize;
        for &curve_idx in &scratch.desc_indices[start..end] {
            let loc = curve_locations[curve_idx];
            scratch_entries.push(encode_offset(loc.offset + band_element_count));
            scratch_entries.push(0);
            scratch_entries.push(0);
            scratch_entries.push(0);
        }
        for &curve_idx in &scratch.asc_indices[start..end] {
            let loc = curve_locations[curve_idx];
            scratch_entries.push(encode_offset(loc.offset + band_element_count));
            scratch_entries.push(0);
            scratch_entries.push(0);
            scratch_entries.push(0);
        }
    }
    for b in 0..vcount {
        let start = scratch.vband_offsets[b] as usize;
        let end = start + scratch.vband_counts[b] as usize;
        for &curve_idx in &scratch.desc_indices[start..end] {
            let loc = curve_locations[curve_idx];
            scratch_entries.push(encode_offset(loc.offset + band_element_count));
            scratch_entries.push(0);
            scratch_entries.push(0);
            scratch_entries.push(0);
        }
        for &curve_idx in &scratch.asc_indices[start..end] {
            let loc = curve_locations[curve_idx];
            scratch_entries.push(encode_offset(loc.offset + band_element_count));
            scratch_entries.push(0);
            scratch_entries.push(0);
            scratch_entries.push(0);
        }
    }

    BandData {
        entries: scratch_entries,
        band_count_x,
        band_count_y,
        band_transform: [scale_x, scale_y, offset_x, offset_y],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outline::{GlyphOutline, QuadCurve};

    fn make_outline(curves: Vec<QuadCurve>) -> GlyphOutline {
        let mut min = [f32::MAX; 2];
        let mut max = [f32::MIN; 2];
        for c in &curves {
            for p in [c.p1, c.p2, c.p3] {
                min[0] = min[0].min(p[0]);
                min[1] = min[1].min(p[1]);
                max[0] = max[0].max(p[0]);
                max[1] = max[1].max(p[1]);
            }
        }
        GlyphOutline {
            curves,
            bounds: [min[0], min[1], max[0], max[1]],
        }
    }

    fn sequential_locations(n: usize) -> Vec<CurveLocation> {
        (0..n)
            .map(|i| CurveLocation {
                offset: i as u32 * 2,
            })
            .collect()
    }

    #[test]
    fn degenerate_glyph_zero_size() {
        // A glyph with zero width/height (all points at same location)
        let outline = make_outline(vec![QuadCurve {
            p1: [50.0, 50.0],
            p2: [50.0, 50.0],
            p3: [50.0, 50.0],
        }]);
        let locs = sequential_locations(1);
        let data = build_bands(&outline, &locs, 4, 4, Vec::new(), &mut BandScratch::default());

        // Should not panic, should produce valid band data
        assert!(!data.entries.is_empty());
        assert!(data.band_transform[0].is_finite());
        assert!(data.band_transform[1].is_finite());
    }

    #[test]
    fn single_curve_appears_in_all_1x1_bands() {
        let outline = make_outline(vec![QuadCurve {
            p1: [0.0, 0.0],
            p2: [50.0, 100.0],
            p3: [100.0, 0.0],
        }]);
        let locs = sequential_locations(1);
        let data = build_bands(&outline, &locs, 1, 1, Vec::new(), &mut BandScratch::default());

        // 2 band headers (h + v) * 4 u32 each = 8
        // 1 curve * 2 bands * 2 lists (desc+asc) * 4 u32 each = 16
        assert_eq!(data.entries.len(), 24);
        // h-band should have count=1
        assert_eq!(data.entries[0], 1);
        // v-band should have count=1
        assert_eq!(data.entries[4], 1);
    }

    #[test]
    fn curve_in_correct_horizontal_band() {
        // Two curves: one in bottom half, one in top half
        let outline = make_outline(vec![
            QuadCurve {
                p1: [0.0, 0.0],
                p2: [50.0, 20.0],
                p3: [100.0, 0.0],
            },
            QuadCurve {
                p1: [0.0, 80.0],
                p2: [50.0, 100.0],
                p3: [100.0, 80.0],
            },
        ]);
        let locs = sequential_locations(2);
        let data = build_bands(&outline, &locs, 1, 2, Vec::new(), &mut BandScratch::default());

        // 3 band headers (2 hbands + 1 vband) * 4 u32 = 12
        // h-band 0 (bottom) should contain curve 0
        // h-band 1 (top) should contain curve 1
        let hband0_count = data.entries[0];
        let hband1_count = data.entries[4];
        assert!(
            hband0_count >= 1,
            "bottom h-band should contain bottom curve"
        );
        assert!(hband1_count >= 1, "top h-band should contain top curve");
    }

    #[test]
    fn band_transform_maps_bounds_to_band_range() {
        let outline = make_outline(vec![QuadCurve {
            p1: [10.0, 20.0],
            p2: [50.0, 80.0],
            p3: [90.0, 20.0],
        }]);
        let locs = sequential_locations(1);
        let data = build_bands(&outline, &locs, 4, 4, Vec::new(), &mut BandScratch::default());

        let [scale_x, scale_y, offset_x, offset_y] = data.band_transform;

        // min_x mapped should give 0, max_x mapped should give band_count_x
        let mapped_min_x = outline.bounds[0] * scale_x + offset_x;
        let mapped_max_x = outline.bounds[2] * scale_x + offset_x;
        assert!(
            (mapped_min_x).abs() < 1e-4,
            "min_x should map to ~0, got {mapped_min_x}"
        );
        assert!(
            (mapped_max_x - 4.0).abs() < 1e-4,
            "max_x should map to ~4, got {mapped_max_x}"
        );

        let mapped_min_y = outline.bounds[1] * scale_y + offset_y;
        let mapped_max_y = outline.bounds[3] * scale_y + offset_y;
        assert!(
            (mapped_min_y).abs() < 1e-4,
            "min_y should map to ~0, got {mapped_min_y}"
        );
        assert!(
            (mapped_max_y - 4.0).abs() < 1e-4,
            "max_y should map to ~4, got {mapped_max_y}"
        );
    }

    #[test]
    fn wide_curve_spans_multiple_vertical_bands() {
        // A curve spanning the full width should appear in all vertical bands
        let outline = make_outline(vec![QuadCurve {
            p1: [0.0, 0.0],
            p2: [50.0, 50.0],
            p3: [100.0, 0.0],
        }]);
        let locs = sequential_locations(1);
        let data = build_bands(&outline, &locs, 4, 1, Vec::new(), &mut BandScratch::default());

        // 1 h-band header + 4 v-band headers = 5 * 4 = 20 u32s
        // The curve appears in hband (1 ref * 2 lists) + all 4 vbands (4 refs * 2 lists)
        // = 10 refs * 4 u32 = 40
        let total_expected = 20 + 40;
        assert_eq!(data.entries.len(), total_expected);
    }

    #[test]
    fn entries_are_aligned_to_uint4() {
        // Every entry in the band data should be a multiple of 4 u32s
        let outline = make_outline(vec![
            QuadCurve {
                p1: [0.0, 0.0],
                p2: [25.0, 50.0],
                p3: [50.0, 0.0],
            },
            QuadCurve {
                p1: [50.0, 50.0],
                p2: [75.0, 100.0],
                p3: [100.0, 50.0],
            },
        ]);
        let locs = sequential_locations(2);
        let data = build_bands(&outline, &locs, 3, 3, Vec::new(), &mut BandScratch::default());
        assert_eq!(data.entries.len() % 4, 0, "entries must be uint4-aligned");
    }
}
