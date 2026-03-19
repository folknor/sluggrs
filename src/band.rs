/// Band acceleration structure builder for Slug rendering.
///
/// Divides the glyph bounding box into horizontal and vertical bands,
/// recording which curves intersect each band. This lets the fragment
/// shader skip curves that can't affect the current pixel.
use crate::prepare::GpuOutline;

/// Band data ready for GPU upload.
pub struct BandData {
    /// Entries in the band texture: (curve_count, offset) pairs followed by curve indices.
    /// Layout: [hband_0_count, hband_0_offset, hband_1_count, hband_1_offset, ...,
    ///          vband_0_count, vband_0_offset, ...,
    ///          curve_indices...]
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
pub fn build_bands(
    outline: &GpuOutline,
    curve_locations: &[CurveLocation],
    band_count_x: u32,
    band_count_y: u32,
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

    // For each horizontal band (indexed by y), collect curves that overlap it
    let mut hband_curves: Vec<Vec<usize>> = vec![Vec::new(); band_count_y as usize];
    // For each vertical band (indexed by x), collect curves that overlap it
    let mut vband_curves: Vec<Vec<usize>> = vec![Vec::new(); band_count_x as usize];

    for (i, curve) in outline.curves.iter().enumerate() {
        let curve_min_y = curve.p1[1].min(curve.p2[1]).min(curve.p3[1]);
        let curve_max_y = curve.p1[1].max(curve.p2[1]).max(curve.p3[1]);
        let curve_min_x = curve.p1[0].min(curve.p2[0]).min(curve.p3[0]);
        let curve_max_x = curve.p1[0].max(curve.p2[0]).max(curve.p3[0]);

        // Horizontal bands: inclusive lower bound, exclusive upper bound (biased by epsilon)
        const BAND_EPSILON: f32 = 1e-5;

        let hmin = curve_min_y * scale_y + offset_y;
        let hmax = curve_max_y * scale_y + offset_y;
        let hband_min = hmin.floor().clamp(0.0, band_count_y as f32 - 1.0) as u32;
        let hband_max = (hmax - BAND_EPSILON)
            .floor()
            .clamp(0.0, band_count_y as f32 - 1.0) as u32;

        for b in hband_min..=hband_max {
            hband_curves[b as usize].push(i);
        }

        // Vertical bands: inclusive lower bound, exclusive upper bound (biased by epsilon)
        let vmin = curve_min_x * scale_x + offset_x;
        let vmax = curve_max_x * scale_x + offset_x;
        let vband_min = vmin.floor().clamp(0.0, band_count_x as f32 - 1.0) as u32;
        let vband_max = (vmax - BAND_EPSILON)
            .floor()
            .clamp(0.0, band_count_x as f32 - 1.0) as u32;

        for b in vband_min..=vband_max {
            vband_curves[b as usize].push(i);
        }
    }

    // Sort horizontal band curves by descending max x (for early exit in shader)
    for band in &mut hband_curves {
        band.sort_by(|&a, &b| {
            let max_x_a = outline.curves[a]
                .p1[0]
                .max(outline.curves[a].p2[0])
                .max(outline.curves[a].p3[0]);
            let max_x_b = outline.curves[b]
                .p1[0]
                .max(outline.curves[b].p2[0])
                .max(outline.curves[b].p3[0]);
            max_x_b.partial_cmp(&max_x_a).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Sort vertical band curves by descending max y
    for band in &mut vband_curves {
        band.sort_by(|&a, &b| {
            let max_y_a = outline.curves[a]
                .p1[1]
                .max(outline.curves[a].p2[1])
                .max(outline.curves[a].p3[1]);
            let max_y_b = outline.curves[b]
                .p1[1]
                .max(outline.curves[b].p2[1])
                .max(outline.curves[b].p3[1]);
            max_y_b.partial_cmp(&max_y_a).unwrap_or(std::cmp::Ordering::Equal)
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

    let num_headers = band_count_y + band_count_x;
    let mut entries: Vec<u32> = Vec::new();

    // Reserve space for headers (2 u32s per band: count, offset)
    // But in the Slug format, each band header is stored as a single texel with
    // count in .x and offset in .y
    // We'll pack: each header = 2 values, each curve ref = 2 values
    // Headers come first, then curve index lists

    // Phase 1: calculate offsets
    let curve_lists_start = num_headers; // offset in texels from glyph start
    let mut hband_offsets: Vec<(u32, u32)> = Vec::new(); // (count, offset)
    let mut vband_offsets: Vec<(u32, u32)> = Vec::new();

    let mut current_offset = curve_lists_start;
    for band in &hband_curves {
        hband_offsets.push((band.len() as u32, current_offset));
        current_offset += band.len() as u32;
    }
    for band in &vband_curves {
        vband_offsets.push((band.len() as u32, current_offset));
        current_offset += band.len() as u32;
    }

    // Phase 2: write headers
    // Horizontal band headers come first (band_count_y of them)
    for &(count, offset) in &hband_offsets {
        entries.push(count);
        entries.push(offset);
        entries.push(0); // padding for uint4
        entries.push(0);
    }
    // Vertical band headers
    for &(count, offset) in &vband_offsets {
        entries.push(count);
        entries.push(offset);
        entries.push(0);
        entries.push(0);
    }

    // Phase 3: write curve index lists
    for band in &hband_curves {
        for &curve_idx in band {
            let loc = curve_locations[curve_idx];
            entries.push(loc.x);
            entries.push(loc.y);
            entries.push(0);
            entries.push(0);
        }
    }
    for band in &vband_curves {
        for &curve_idx in band {
            let loc = curve_locations[curve_idx];
            entries.push(loc.x);
            entries.push(loc.y);
            entries.push(0);
            entries.push(0);
        }
    }

    BandData {
        entries,
        band_count_x,
        band_count_y,
        band_transform: [scale_x, scale_y, offset_x, offset_y],
    }
}
