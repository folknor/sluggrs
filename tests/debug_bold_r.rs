use skrifa::setting::VariationSetting;
use sluggrs::band::{CurveLocation, build_bands};
use sluggrs::outline::extract_outline;
use sluggrs::prepare::prepare_outline;

#[test]
fn dump_bold_r_geometry() {
    let font_data = std::fs::read("examples/fonts/InterVariable.ttf")
        .expect("InterVariable.ttf must be present");
    let glyph_id =
        sluggrs::outline::char_to_glyph_id(&font_data, 0, 'r').expect("'r' should be mapped");

    let wght = skrifa::Tag::new(b"wght");
    let location = [VariationSetting::new(wght, 700.0)];
    let outline = extract_outline(&font_data, 0, glyph_id, &location).expect("should have outline");
    let gpu = prepare_outline(&outline);

    println!("=== ORIGINAL curves 2..5 ===");
    for i in 2..=5.min(outline.curves.len() - 1) {
        let c = &outline.curves[i];
        println!(
            "  orig {:2}: p1=({:9.4},{:9.4}) p2=({:9.4},{:9.4}) p3=({:9.4},{:9.4})",
            i, c.p1[0], c.p1[1], c.p2[0], c.p2[1], c.p3[0], c.p3[1]
        );
    }

    println!("\n=== GPU-PREPARED curves 2..5 ===");
    for i in 2..=5.min(gpu.curves.len() - 1) {
        let c = &gpu.curves[i];
        let a_x = c.p1[0] - 2.0 * c.p2[0] + c.p3[0];
        let a_y = c.p1[1] - 2.0 * c.p2[1] + c.p3[1];
        println!(
            "  gpu  {:2}: p1=({:9.4},{:9.4}) p2=({:9.4},{:9.4}) p3=({:9.4},{:9.4})  a=({:12.8},{:12.8})",
            i, c.p1[0], c.p1[1], c.p2[0], c.p2[1], c.p3[0], c.p3[1], a_x, a_y
        );
    }

    println!("\n=== ALL GPU-PREPARED curves ===");
    for (i, c) in gpu.curves.iter().enumerate() {
        let mid_x = (c.p1[0] + c.p3[0]) * 0.5;
        let mid_y = (c.p1[1] + c.p3[1]) * 0.5;
        let was_linear = (c.p2[0] - mid_x).abs() < 0.02 && (c.p2[1] - mid_y).abs() < 0.02;
        let a_x = c.p1[0] - 2.0 * c.p2[0] + c.p3[0];
        let a_y = c.p1[1] - 2.0 * c.p2[1] + c.p3[1];
        println!(
            "  {:2}: p1=({:9.4},{:9.4}) p2=({:9.4},{:9.4}) p3=({:9.4},{:9.4})  a=({:12.8},{:12.8}) perturbed={}",
            i, c.p1[0], c.p1[1], c.p2[0], c.p2[1], c.p3[0], c.p3[1], a_x, a_y, was_linear
        );
    }

    // Build bands
    let num_curves = gpu.curves.len();
    let band_count = if num_curves < 10 {
        4
    } else if num_curves < 30 {
        8
    } else {
        12
    };
    let curve_locs: Vec<CurveLocation> = (0..num_curves)
        .map(|i| CurveLocation {
            offset: (i as u32) * 2,
        })
        .collect();
    let band_data = build_bands(&gpu, &curve_locs, band_count, band_count, Vec::new());

    // Dump which bands contain curves 2..5
    let hcount = band_data.band_count_y as usize;
    let vcount = band_data.band_count_x as usize;

    println!("\n=== BAND MEMBERSHIP for curves 2..5 ===");
    for target_curve in 2u32..=5 {
        let mut h_bands = Vec::new();
        let mut v_bands = Vec::new();

        for i in 0..hcount {
            let base = i * 4;
            let count = band_data.entries[base] as usize;
            let offset = (band_data.entries[base + 1] as i32 + 32768) as usize;
            for ci in 0..count {
                let ref_base = (offset + ci) * 4;
                let curve_offset = (band_data.entries[ref_base] as i32 + 32768) as i16;
                if curve_offset / 2 == target_curve as i16 {
                    h_bands.push(i);
                }
            }
        }

        for i in 0..vcount {
            let base = (hcount + i) * 4;
            let count = band_data.entries[base] as usize;
            let offset = (band_data.entries[base + 1] as i32 + 32768) as usize;
            for ci in 0..count {
                let ref_base = (offset + ci) * 4;
                let curve_offset = (band_data.entries[ref_base] as i32 + 32768) as i16;
                if curve_offset / 2 == target_curve as i16 {
                    v_bands.push(i);
                }
            }
        }

        println!("  curve {target_curve}: h_bands={h_bands:?}, v_bands={v_bands:?}");
    }

    // What are the band y-ranges?
    let [_, min_y, _, max_y] = gpu.bounds;
    let height = max_y - min_y;
    let band_h = height / band_count as f32;
    println!("\n=== BAND Y-RANGES (horizontal bands) ===");
    for i in 0..hcount {
        let lo = min_y + i as f32 * band_h;
        let hi = lo + band_h;
        println!("  h{i}: y=[{lo:.1}, {hi:.1}]");
    }

    let [min_x, _, max_x, _] = gpu.bounds;
    let width = max_x - min_x;
    let band_w = width / band_count as f32;
    println!("\n=== BAND X-RANGES (vertical bands) ===");
    for i in 0..vcount {
        let lo = min_x + i as f32 * band_w;
        let hi = lo + band_w;
        println!("  v{i}: x=[{lo:.1}, {hi:.1}]");
    }
}
