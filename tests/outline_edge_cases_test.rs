//! Edge-case tests for outline extraction, italic shear, TTC face selection,
//! and the comma line-only glyph regression.
//!
//! Run with: cargo test --test outline_edge_cases_test -- --nocapture

use sluggrs::band::{build_bands, CurveLocation};
use sluggrs::outline::{char_to_glyph_id, extract_outline};
use sluggrs::prepare::{apply_italic_shear, prepare_outline};

/// Path to the bundled Inter Variable font used across these tests.
const INTER_FONT: &str = "examples/fonts/InterVariable.ttf";

fn load_inter() -> Vec<u8> {
    std::fs::read(INTER_FONT).expect("InterVariable.ttf must be present")
}

// ---------------------------------------------------------------------------
// Test 1: Fake italic changes geometry
// ---------------------------------------------------------------------------

#[test]
fn fake_italic_changes_geometry() {
    let font_data = load_inter();
    let glyph_id =
        char_to_glyph_id(&font_data, 0, 'A').expect("'A' should be mapped in Inter");

    let outline =
        extract_outline(&font_data, 0, glyph_id, &[]).expect("'A' should have an outline");
    let base = prepare_outline(&outline);
    let mut sheared = base.clone();
    apply_italic_shear(&mut sheared);

    // At least one curve's x-coordinates must differ after shearing.
    // (Points on the baseline y=0 won't shift, but points above it will.)
    let any_curve_differs = base.curves.iter().zip(sheared.curves.iter()).any(|(b, s)| {
        (b.p1[0] - s.p1[0]).abs() > 1e-6
            || (b.p2[0] - s.p2[0]).abs() > 1e-6
            || (b.p3[0] - s.p3[0]).abs() > 1e-6
    });
    assert!(
        any_curve_differs,
        "Italic shear should change at least some x-coordinates in the curves"
    );

    // The x-coordinates of the sheared outline should generally be shifted rightward
    // for points above the baseline. Compute the sum of x-shifts for points with y > 0;
    // the net shift must be positive (shear direction check).
    let mut total_x_shift = 0.0f64;
    let mut count = 0u32;
    for (b, s) in base.curves.iter().zip(sheared.curves.iter()) {
        for (po, ps) in [(b.p1, s.p1), (b.p2, s.p2), (b.p3, s.p3)] {
            if po[1].abs() > 1.0 {
                total_x_shift += (ps[0] - po[0]) as f64;
                count += 1;
            }
        }
    }
    assert!(
        count > 0,
        "'A' should have control points away from the baseline"
    );
    assert!(
        total_x_shift > 0.0,
        "Net x-shift for points with y > 0 should be positive (shear direction). \
         Got {total_x_shift:.4} over {count} points"
    );

    // Verify shear direction: for positive y values, x should increase.
    // The shear is x' = x + y * 0.2493, so for any control point with y > 0
    // the sheared x should be larger than the original.
    for (orig, shear) in base.curves.iter().zip(sheared.curves.iter()) {
        for (po, ps) in [
            (orig.p1, shear.p1),
            (orig.p2, shear.p2),
            (orig.p3, shear.p3),
        ] {
            if po[1] > 1.0 {
                assert!(
                    ps[0] > po[0],
                    "For positive y ({:.1}), sheared x ({:.4}) should be greater than original x ({:.4})",
                    po[1], ps[0], po[0]
                );
            }
            // y coordinates must be unchanged
            assert!(
                (po[1] - ps[1]).abs() < 1e-6,
                "Italic shear should not modify y-coordinates"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: TTC / face-index correctness
// ---------------------------------------------------------------------------

/// Walk font directories looking for a .ttc file that has at least 2 faces.
fn find_ttc_font() -> Option<Vec<u8>> {
    fn walk(dir: &std::path::Path) -> Option<Vec<u8>> {
        let entries = std::fs::read_dir(dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && let Some(data) = walk(&path) {
                return Some(data);
            } else if let Some(ext) = path.extension()
                && ext.eq_ignore_ascii_case("ttc")
                && let Ok(data) = std::fs::read(&path)
            {
                // Verify it actually has at least 2 faces
                if skrifa::FontRef::from_index(&data, 1).is_ok() {
                    eprintln!("Found TTC: {}", path.display());
                    return Some(data);
                }
            }
        }
        None
    }

    for dir in &["/usr/share/fonts", "/usr/local/share/fonts"] {
        let p = std::path::Path::new(dir);
        if p.is_dir() && let Some(data) = walk(p) {
            return Some(data);
        }
    }
    None
}

#[test]
fn ttc_face_index_correctness() {
    let font_data = match find_ttc_font() {
        Some(d) => d,
        None => {
            eprintln!("No TTC font found on system — skipping test");
            return;
        }
    };

    // Pick a common glyph (space or 'A') that should be present in both faces.
    // We try glyph_id 1 first (often .notdef+1 or space), then fall back to a
    // brute-force scan for any glyph that exists in both faces.
    let glyph_id = {
        let mut found = None;
        // Try common ASCII glyph IDs
        for candidate in 1u16..200 {
            let a = extract_outline(&font_data, 0, candidate, &[]);
            let b = extract_outline(&font_data, 1, candidate, &[]);
            if a.is_some() && b.is_some() {
                found = Some(candidate);
                break;
            }
        }
        match found {
            Some(id) => id,
            None => {
                eprintln!("No common glyph with outline found in both TTC faces — skipping");
                return;
            }
        }
    };

    let outline_0 =
        extract_outline(&font_data, 0, glyph_id, &[]).expect("Face 0 outline");
    let outline_1 =
        extract_outline(&font_data, 1, glyph_id, &[]).expect("Face 1 outline");

    // The two faces should produce different geometry (different typefaces in the collection).
    let bounds_differ = outline_0.bounds != outline_1.bounds;
    let curve_count_differs = outline_0.curves.len() != outline_1.curves.len();

    // If bounds and curve count are somehow identical, compare actual control points.
    let points_differ = if !bounds_differ && !curve_count_differs {
        outline_0
            .curves
            .iter()
            .zip(outline_1.curves.iter())
            .any(|(a, b)| {
                a.p1 != b.p1 || a.p2 != b.p2 || a.p3 != b.p3
            })
    } else {
        true
    };

    assert!(
        bounds_differ || curve_count_differs || points_differ,
        "Outlines from face 0 and face 1 of TTC should differ for glyph_id {glyph_id}"
    );

    eprintln!(
        "TTC face test passed: glyph_id={glyph_id}, face0 curves={}, face1 curves={}, \
         face0 bounds={:?}, face1 bounds={:?}",
        outline_0.curves.len(),
        outline_1.curves.len(),
        outline_0.bounds,
        outline_1.bounds,
    );
}

// ---------------------------------------------------------------------------
// Test 3: Line-only glyph regression (the comma)
// ---------------------------------------------------------------------------

#[test]
fn comma_line_only_glyph_regression() {
    let font_data = load_inter();
    let glyph_id =
        char_to_glyph_id(&font_data, 0, ',').expect("Comma should be mapped in Inter");

    // Step 1: Extract outline and verify it has curves.
    let outline = extract_outline(&font_data, 0, glyph_id, &[])
        .expect("Comma should have an outline");
    eprintln!(
        "Comma outline: {} curves, bounds={:?}",
        outline.curves.len(),
        outline.bounds
    );
    assert!(
        !outline.curves.is_empty(),
        "Comma should have at least one curve"
    );

    // The comma in Inter Variable is made of line segments, encoded as degenerate
    // quadratics where p2 = p1 (harfbuzz-style encoding).
    let all_linear = outline.curves.iter().all(|c| {
        (c.p2[0] - c.p1[0]).abs() < 1e-6 && (c.p2[1] - c.p1[1]).abs() < 1e-6
    });
    assert!(
        all_linear,
        "All comma curves should be degenerate quadratics (p2 = p1)"
    );
    eprintln!("Confirmed: all {} comma curves are linear (p2 = p1)", outline.curves.len());

    // Step 2: Prepare for GPU — line segments pass through unchanged,
    // the shader handles them as normal quadratics (a = p3 - p1, nonzero).
    let gpu_outline = prepare_outline(&outline);
    assert_eq!(
        gpu_outline.curves.len(),
        outline.curves.len(),
        "prepare_outline should preserve curve count"
    );

    // Lines should pass through with p2 still equal to p1.
    let all_still_linear = gpu_outline.curves.iter().all(|c| {
        (c.p2[0] - c.p1[0]).abs() < 1e-6 && (c.p2[1] - c.p1[1]).abs() < 1e-6
    });
    assert!(
        all_still_linear,
        "prepare_outline should pass line segments through unchanged"
    );
    eprintln!("Confirmed: line segments pass through unchanged after prepare_outline");

    // Step 3: Build bands with a single band (1x1) — the simplest case.
    let curve_locations: Vec<CurveLocation> = (0..gpu_outline.curves.len())
        .map(|i| CurveLocation {
            offset: (i * 3) as u32,
        })
        .collect();

    let band_data = build_bands(&gpu_outline, &curve_locations, 1, 1, Vec::new());

    // Band data should be non-empty.
    assert!(
        !band_data.entries.is_empty(),
        "Band entries should not be empty for comma"
    );
    eprintln!("Band entries: {} u32s", band_data.entries.len());

    // band_transform should have sane (non-zero, non-NaN, non-infinite) values.
    for (i, &val) in band_data.band_transform.iter().enumerate() {
        assert!(
            val.is_finite(),
            "band_transform[{i}] should be finite, got {val}"
        );
    }

    // With axis-aligned curve filtering, horizontal curves are excluded from
    // hbands and vertical curves from vbands (they can never produce ray
    // crossings on that axis). Count expected curves per band type.
    let num_curves = gpu_outline.curves.len();
    let non_horizontal = gpu_outline.curves.iter().filter(|c| {
        let min_y = c.p1[1].min(c.p2[1]).min(c.p3[1]);
        let max_y = c.p1[1].max(c.p2[1]).max(c.p3[1]);
        min_y != max_y
    }).count();
    let non_vertical = gpu_outline.curves.iter().filter(|c| {
        let min_x = c.p1[0].min(c.p2[0]).min(c.p3[0]);
        let max_x = c.p1[0].max(c.p2[0]).max(c.p3[0]);
        min_x != max_x
    }).count();

    // Headers = 2 bands * 4 u32s = 8, plus curve refs * 2 lists (desc+asc) * 4 u32s each
    let expected_total = 8 + (non_horizontal + non_vertical) * 2 * 4;
    assert_eq!(
        band_data.entries.len(),
        expected_total,
        "Expected {} h-band curves + {} v-band curves = {} entries, got {}",
        non_horizontal,
        non_vertical,
        expected_total,
        band_data.entries.len()
    );

    // Verify the h-band header reports non-horizontal curves.
    let hband_count = band_data.entries[0];
    assert_eq!(
        hband_count, non_horizontal as i16,
        "Horizontal band should contain {non_horizontal} non-horizontal curves, got {hband_count}"
    );

    // v-band header is at index 4 (after the 1 h-band header).
    let vband_count = band_data.entries[4];
    assert_eq!(
        vband_count, non_vertical as i16,
        "Vertical band should contain {non_vertical} non-vertical curves, got {vband_count}"
    );

    eprintln!(
        "Comma regression test passed: {num_curves} curves, {non_horizontal} in hband, {non_vertical} in vband"
    );
}

// ---------------------------------------------------------------------------
// Test 4: TTC units_per_em uses correct face index
// ---------------------------------------------------------------------------

#[test]
fn ttc_units_per_em_face_index() {
    let font_data = match find_ttc_font() {
        Some(d) => d,
        None => {
            eprintln!("No TTC font found on system — skipping test");
            return;
        }
    };

    use skrifa::raw::TableProvider;

    // Parse both faces and extract units_per_em
    let face_0 = skrifa::FontRef::from_index(&font_data, 0)
        .expect("Face 0 should parse");
    let face_1 = skrifa::FontRef::from_index(&font_data, 1)
        .expect("Face 1 should parse");

    let upem_0 = face_0.head().expect("face 0 head").units_per_em();
    let upem_1 = face_1.head().expect("face 1 head").units_per_em();

    eprintln!("TTC face 0 units_per_em={upem_0}, face 1 units_per_em={upem_1}");

    // Both values must be valid (non-zero, reasonable range)
    assert!(upem_0 > 0 && upem_0 <= 16384, "face 0 upem out of range: {upem_0}");
    assert!(upem_1 > 0 && upem_1 <= 16384, "face 1 upem out of range: {upem_1}");

    // Verify that FontRef::new() would have returned face 0's value,
    // demonstrating why from_index() is necessary for face 1.
    let face_default = skrifa::FontRef::new(&font_data)
        .expect("Default face should parse");
    let upem_default = face_default.head().expect("default head").units_per_em();
    assert_eq!(
        upem_default, upem_0,
        "FontRef::new() should return face 0's units_per_em"
    );

    // The bug: if we used FontRef::new() for face 1, we'd get face 0's
    // upem value. With from_index(), we get the correct one.
    let upem_1_via_index = skrifa::FontRef::from_index(&font_data, 1)
        .expect("from_index face 1")
        .head()
        .expect("face 1 head via index")
        .units_per_em();
    assert_eq!(
        upem_1_via_index, upem_1,
        "from_index(1) should return face 1's units_per_em, not face 0's"
    );

    if upem_0 != upem_1 {
        eprintln!(
            "TTC faces have DIFFERENT units_per_em ({upem_0} vs {upem_1}) — \
             using FontRef::new() for face 1 would have produced wrong scaling"
        );
    } else {
        eprintln!(
            "TTC faces have same units_per_em ({upem_0}) — bug would be \
             latent but from_index() is still correct"
        );
    }
}
