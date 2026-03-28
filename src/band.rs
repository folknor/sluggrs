/// Band acceleration structure builder for Slug rendering.
///
/// Divides the glyph bounding box into horizontal and vertical bands,
/// recording which curves intersect each band. This lets the fragment
/// shader skip curves that can't affect the current pixel.
use crate::prepare::GpuOutline;

/// Band data ready for GPU upload.
pub struct BandData {
    /// Entries in the band texture: (curve_count, offset) pairs followed by curve indices.
    /// Pass this Vec back via `build_bands` to reuse its allocation.
    pub entries: Vec<u32>,
    /// Number of horizontal bands
    pub band_count_x: u32,
    /// Number of vertical bands
    pub band_count_y: u32,
    /// Transform from em-space to band index: band_idx = coord * scale + offset
    pub band_transform: [f32; 4], // scale_x, scale_y, offset_x, offset_y
}

/// Curve location in the curve texture (x, y).
#[derive(Debug, Clone, Copy)]
pub struct CurveLocation {
    pub x: u32,
    pub y: u32,
}

/// Build the band acceleration structure for a glyph.
///
/// Operates on GPU-prepared geometry (with perturbed line segments).
/// `curve_locations` maps each curve index to its (x, y) position in the curve texture.
#[hotpath::measure]
pub fn build_bands(
    outline: &GpuOutline,
    curve_locations: &[CurveLocation],
    band_count_x: u32,
    band_count_y: u32,
    mut scratch_entries: Vec<u32>,
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

    // Pre-compute sort keys and band ranges per curve (single pass over curves)
    let num_curves = outline.curves.len();
    let mut max_x_keys = Vec::with_capacity(num_curves);
    let mut max_y_keys = Vec::with_capacity(num_curves);

    // Phase 1: count curves per band (no allocations per band)
    let mut hband_counts = vec![0u32; hcount];
    let mut vband_counts = vec![0u32; vcount];

    for curve in &outline.curves {
        let curve_min_y = curve.p1[1].min(curve.p2[1]).min(curve.p3[1]);
        let curve_max_y = curve.p1[1].max(curve.p2[1]).max(curve.p3[1]);
        let curve_min_x = curve.p1[0].min(curve.p2[0]).min(curve.p3[0]);
        let curve_max_x = curve.p1[0].max(curve.p2[0]).max(curve.p3[0]);

        max_x_keys.push(curve_max_x);
        max_y_keys.push(curve_max_y);

        let hband_min = (curve_min_y * scale_y + offset_y).floor().clamp(0.0, hcount as f32 - 1.0) as usize;
        let hband_max = ((curve_max_y * scale_y + offset_y - BAND_EPSILON).floor()).clamp(0.0, hcount as f32 - 1.0) as usize;
        for b in hband_min..=hband_max {
            hband_counts[b] += 1;
        }

        let vband_min = (curve_min_x * scale_x + offset_x).floor().clamp(0.0, vcount as f32 - 1.0) as usize;
        let vband_max = ((curve_max_x * scale_x + offset_x - BAND_EPSILON).floor()).clamp(0.0, vcount as f32 - 1.0) as usize;
        for b in vband_min..=vband_max {
            vband_counts[b] += 1;
        }
    }

    // Phase 2: compute offsets into flat array, then fill
    let htotal: u32 = hband_counts.iter().sum();
    let vtotal: u32 = vband_counts.iter().sum();

    let mut hband_offsets = Vec::with_capacity(hcount);
    let mut offset = 0u32;
    for &count in &hband_counts {
        hband_offsets.push(offset);
        offset += count;
    }
    let mut vband_offsets = Vec::with_capacity(vcount);
    for &count in &vband_counts {
        vband_offsets.push(offset);
        offset += count;
    }

    // Single flat array for all curve indices
    let mut flat_indices = vec![0usize; (htotal + vtotal) as usize];
    let mut hband_fill = hband_offsets.clone();
    let mut vband_fill = vband_offsets.clone();

    for (i, curve) in outline.curves.iter().enumerate() {
        let curve_min_y = curve.p1[1].min(curve.p2[1]).min(curve.p3[1]);
        let curve_max_y = curve.p1[1].max(curve.p2[1]).max(curve.p3[1]);
        let curve_min_x = curve.p1[0].min(curve.p2[0]).min(curve.p3[0]);
        let curve_max_x = curve.p1[0].max(curve.p2[0]).max(curve.p3[0]);

        let hband_min = (curve_min_y * scale_y + offset_y).floor().clamp(0.0, hcount as f32 - 1.0) as usize;
        let hband_max = ((curve_max_y * scale_y + offset_y - BAND_EPSILON).floor()).clamp(0.0, hcount as f32 - 1.0) as usize;
        for b in hband_min..=hband_max {
            flat_indices[hband_fill[b] as usize] = i;
            hband_fill[b] += 1;
        }

        let vband_min = (curve_min_x * scale_x + offset_x).floor().clamp(0.0, vcount as f32 - 1.0) as usize;
        let vband_max = ((curve_max_x * scale_x + offset_x - BAND_EPSILON).floor()).clamp(0.0, vcount as f32 - 1.0) as usize;
        for b in vband_min..=vband_max {
            flat_indices[vband_fill[b] as usize] = i;
            vband_fill[b] += 1;
        }
    }

    // Sort each band's slice by descending max coordinate (for shader early exit)
    for b in 0..hcount {
        let start = hband_offsets[b] as usize;
        let end = start + hband_counts[b] as usize;
        flat_indices[start..end].sort_unstable_by(|&a, &b| {
            max_x_keys[b].partial_cmp(&max_x_keys[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    for b in 0..vcount {
        let start = vband_offsets[b] as usize;
        let end = start + vband_counts[b] as usize;
        flat_indices[start..end].sort_unstable_by(|&a, &b| {
            max_y_keys[b].partial_cmp(&max_y_keys[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Build the band texture data.
    // Layout:
    //   Row 0: [hband_0_header, hband_1_header, ..., vband_0_header, vband_1_header, ...]
    //   After headers: curve index lists
    //
    // Each header is (count, offset_from_glyph_start).
    // Each curve index entry is (curve_loc.x, curve_loc.y, 0, 0) but in the Slug format
    // it's just (curve_loc.x, curve_loc.y) packed as uint2.

    // Build GPU texture data from flat arrays
    let num_headers = band_count_y + band_count_x;
    let curve_lists_start = num_headers;

    // Reuse scratch buffer for entries
    let total_refs = (htotal + vtotal) as usize;
    scratch_entries.clear();
    scratch_entries.reserve((num_headers as usize + total_refs) * 4);

    // Write horizontal band headers
    let mut texel_offset = curve_lists_start;
    for b in 0..hcount {
        scratch_entries.push(hband_counts[b]);
        scratch_entries.push(texel_offset);
        scratch_entries.push(0);
        scratch_entries.push(0);
        texel_offset += hband_counts[b];
    }
    // Write vertical band headers
    for b in 0..vcount {
        scratch_entries.push(vband_counts[b]);
        scratch_entries.push(texel_offset);
        scratch_entries.push(0);
        scratch_entries.push(0);
        texel_offset += vband_counts[b];
    }

    // Write curve references from flat array
    for &curve_idx in &flat_indices {
        let loc = curve_locations[curve_idx];
        scratch_entries.push(loc.x);
        scratch_entries.push(loc.y);
        scratch_entries.push(0);
        scratch_entries.push(0);
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
    use crate::outline::QuadCurve;

    fn make_outline(curves: Vec<QuadCurve>) -> GpuOutline {
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
        GpuOutline {
            curves,
            bounds: [min[0], min[1], max[0], max[1]],
        }
    }

    fn sequential_locations(n: usize) -> Vec<CurveLocation> {
        (0..n).map(|i| CurveLocation { x: i as u32 * 2, y: 0 }).collect()
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
        let data = build_bands(&outline, &locs, 4, 4, Vec::new());

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
        let data = build_bands(&outline, &locs, 1, 1, Vec::new());

        // 2 band headers (h + v) * 4 u32 each = 8
        // 1 curve * 2 bands * 4 u32 each = 8
        assert_eq!(data.entries.len(), 16);
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
        let data = build_bands(&outline, &locs, 1, 2, Vec::new());

        // 3 band headers (2 hbands + 1 vband) * 4 u32 = 12
        // h-band 0 (bottom) should contain curve 0
        // h-band 1 (top) should contain curve 1
        let hband0_count = data.entries[0];
        let hband1_count = data.entries[4];
        assert!(hband0_count >= 1, "bottom h-band should contain bottom curve");
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
        let data = build_bands(&outline, &locs, 4, 4, Vec::new());

        let [scale_x, scale_y, offset_x, offset_y] = data.band_transform;

        // min_x mapped should give 0, max_x mapped should give band_count_x
        let mapped_min_x = outline.bounds[0] * scale_x + offset_x;
        let mapped_max_x = outline.bounds[2] * scale_x + offset_x;
        assert!((mapped_min_x).abs() < 1e-4, "min_x should map to ~0, got {mapped_min_x}");
        assert!((mapped_max_x - 4.0).abs() < 1e-4, "max_x should map to ~4, got {mapped_max_x}");

        let mapped_min_y = outline.bounds[1] * scale_y + offset_y;
        let mapped_max_y = outline.bounds[3] * scale_y + offset_y;
        assert!((mapped_min_y).abs() < 1e-4, "min_y should map to ~0, got {mapped_min_y}");
        assert!((mapped_max_y - 4.0).abs() < 1e-4, "max_y should map to ~4, got {mapped_max_y}");
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
        let data = build_bands(&outline, &locs, 4, 1, Vec::new());

        // 1 h-band header + 4 v-band headers = 5 * 4 = 20 u32s
        // The curve should appear in the h-band (1 ref) and all 4 v-bands (4 refs)
        // = 5 refs * 4 u32 = 20
        let total_expected = 20 + 20;
        assert_eq!(data.entries.len(), total_expected);
    }

    #[test]
    fn entries_are_aligned_to_uint4() {
        // Every entry in the band data should be a multiple of 4 u32s
        let outline = make_outline(vec![
            QuadCurve { p1: [0.0, 0.0], p2: [25.0, 50.0], p3: [50.0, 0.0] },
            QuadCurve { p1: [50.0, 50.0], p2: [75.0, 100.0], p3: [100.0, 50.0] },
        ]);
        let locs = sequential_locations(2);
        let data = build_bands(&outline, &locs, 3, 3, Vec::new());
        assert_eq!(data.entries.len() % 4, 0, "entries must be uint4-aligned");
    }
}
