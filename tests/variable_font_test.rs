/// Tests for variable font outline extraction correctness.
///
/// Uses the Inter Variable font bundled in examples/fonts/InterVariable.ttf
/// to verify that extracting outlines at different weight settings produces
/// meaningfully different results.
use sluggrs::outline::{extract_outline, char_to_glyph_id};

const INTER_VARIABLE: &[u8] = include_bytes!("../examples/fonts/InterVariable.ttf");

#[test]
fn variable_font_outline_differs_by_weight() {
    // Map 'S' to a glyph ID — a character with curves that visibly change with weight.
    let glyph_id = char_to_glyph_id(INTER_VARIABLE, 0, 'S')
        .expect("Inter Variable should contain glyph for 'S'");

    let weight_400 = skrifa::setting::VariationSetting::new(
        skrifa::Tag::new(b"wght"),
        400.0,
    );
    let weight_700 = skrifa::setting::VariationSetting::new(
        skrifa::Tag::new(b"wght"),
        700.0,
    );

    let outline_regular = extract_outline(INTER_VARIABLE, 0, glyph_id, &[weight_400])
        .expect("Should extract outline at weight 400");
    let outline_bold = extract_outline(INTER_VARIABLE, 0, glyph_id, &[weight_700])
        .expect("Should extract outline at weight 700");

    println!("Weight 400: {} curves, bounds {:?}", outline_regular.curves.len(), outline_regular.bounds);
    println!("Weight 700: {} curves, bounds {:?}", outline_bold.curves.len(), outline_bold.bounds);

    // The outlines must differ — either in bounds or control points.
    // A bold 'S' has wider strokes, so the bounds and/or control points change.
    let bounds_differ = outline_regular.bounds != outline_bold.bounds;

    let points_differ = if outline_regular.curves.len() == outline_bold.curves.len() {
        // Same number of curves — check that at least one control point differs.
        outline_regular.curves.iter().zip(outline_bold.curves.iter()).any(|(a, b)| {
            a.p1 != b.p1 || a.p2 != b.p2 || a.p3 != b.p3
        })
    } else {
        // Different curve count already means they differ.
        true
    };

    assert!(
        bounds_differ || points_differ,
        "Outlines at weight 400 and 700 should differ for a variable font"
    );

    // Specifically check that the bolder weight is wider (x-extent grows with weight for 'S').
    let width_regular = outline_regular.bounds[2] - outline_regular.bounds[0];
    let width_bold = outline_bold.bounds[2] - outline_bold.bounds[0];
    println!("Width regular={width_regular}, bold={width_bold}");
    assert!(
        (width_regular - width_bold).abs() > 1.0,
        "Bounding box width should differ meaningfully between weights: regular={width_regular}, bold={width_bold}"
    );
}

#[test]
fn variable_font_outline_differs_for_lowercase_a() {
    // Test with 'a' as well — another character with clear weight variation.
    let glyph_id = char_to_glyph_id(INTER_VARIABLE, 0, 'a')
        .expect("Inter Variable should contain glyph for 'a'");

    let weight_400 = skrifa::setting::VariationSetting::new(
        skrifa::Tag::new(b"wght"),
        400.0,
    );
    let weight_700 = skrifa::setting::VariationSetting::new(
        skrifa::Tag::new(b"wght"),
        700.0,
    );

    let outline_regular = extract_outline(INTER_VARIABLE, 0, glyph_id, &[weight_400])
        .expect("Should extract 'a' outline at weight 400");
    let outline_bold = extract_outline(INTER_VARIABLE, 0, glyph_id, &[weight_700])
        .expect("Should extract 'a' outline at weight 700");

    println!("'a' weight 400: {} curves, bounds {:?}", outline_regular.curves.len(), outline_regular.bounds);
    println!("'a' weight 700: {} curves, bounds {:?}", outline_bold.curves.len(), outline_bold.bounds);

    // At minimum, control points must differ between regular and bold.
    let any_point_differs = outline_regular.curves.iter()
        .flat_map(|c| [c.p1, c.p2, c.p3])
        .zip(outline_bold.curves.iter().flat_map(|c| [c.p1, c.p2, c.p3]))
        .any(|(a, b)| a != b);

    // If curve counts differ, that alone proves they're different.
    let curves_differ = outline_regular.curves.len() != outline_bold.curves.len();

    assert!(
        curves_differ || any_point_differs,
        "Outlines for 'a' at weight 400 and 700 should differ"
    );
}

#[test]
fn same_weight_produces_identical_outlines() {
    // Sanity check: same weight should produce identical outlines.
    let glyph_id = char_to_glyph_id(INTER_VARIABLE, 0, 'S')
        .expect("Inter Variable should contain glyph for 'S'");

    let weight_400 = skrifa::setting::VariationSetting::new(
        skrifa::Tag::new(b"wght"),
        400.0,
    );

    let outline_a = extract_outline(INTER_VARIABLE, 0, glyph_id, &[weight_400])
        .expect("Should extract outline");
    let outline_b = extract_outline(INTER_VARIABLE, 0, glyph_id, &[weight_400])
        .expect("Should extract outline");

    assert_eq!(outline_a.curves.len(), outline_b.curves.len(), "Same weight should produce same curve count");
    assert_eq!(outline_a.bounds, outline_b.bounds, "Same weight should produce same bounds");

    for (i, (a, b)) in outline_a.curves.iter().zip(outline_b.curves.iter()).enumerate() {
        assert_eq!(a.p1, b.p1, "Curve {i} p1 should match");
        assert_eq!(a.p2, b.p2, "Curve {i} p2 should match");
        assert_eq!(a.p3, b.p3, "Curve {i} p3 should match");
    }
}
