# GPU Vector Text Rendering (Slug + Loop-Blinn)

GPU-accelerated vector text rendering via direct bezier curve evaluation in fragment shaders. Resolution-independent, perfect at any zoom/scale, no glyph atlas needed.

## Why This Matters

The current iced text rendering stack (`cosmic-text` + `cryoglyph`) is atlas-based and produces slightly off results. Slug is a fundamentally different approach — curves are evaluated per-pixel on the GPU, which means:

- No rasterization artifacts at any scale factor
- No atlas memory overhead
- Clean rendering at fractional scale factors (relevant for our DPI auto-detection)
- Correct subpixel positioning without hinting hacks

## Status

Two key patents covering GPU-based curve rendering have become/are becoming freely available within days of each other:

### Slug — public domain (March 17, 2026)

The Slug algorithm patent (US #10,373,352) was **dedicated to the public domain** on March 17, 2026 by Eric Lengyel. Slug evaluates quadratic/cubic bezier curves per-pixel in the fragment shader for font glyph rendering. Reference shaders are available under MIT license.

- Blog post: https://terathon.com/blog/decade-slug.html
- Reference implementation (vertex + pixel shaders, MIT): https://github.com/EricLengyel/Slug
- Original JCGT paper: "GPU-Centered Font Rendering Directly from Glyph Outlines" (2017)

### Loop-Blinn — patent expires March 25, 2026

Microsoft's Loop-Blinn patent ([US #7,564,459](https://patents.google.com/patent/US7564459B2/en)) covers resolution-independent rendering of cubic bezier curves on programmable GPU hardware. The technique classifies cubic curves by inflection points, projects them into a canonical texture space, and uses pixel shaders for point-in-shape testing. Inventors: Charles Loop, Jim Blinn. Assignee: Microsoft Technology Licensing LLC.

- Filed: October 31, 2005 — **Expires: March 25, 2026**
- Original paper: "Resolution Independent Curve Rendering using Programmable Graphics Hardware" (Loop & Blinn, SIGGRAPH 2005)

### Combined significance

Together, these two developments remove all major patent barriers to GPU-based vector text and curve rendering. The entire design space for atlas-free, resolution-independent text pipelines is now open.

## Actionability

Not actionable now. This would be a change to the iced rendering layer, not our app code. Possible paths:

1. **Upstream iced adoption** — if iced replaces its text renderer with a Slug-based approach, we get it for free
2. **Custom iced fork patch** — if text rendering quality becomes a blocker, we could prototype a Slug-based text renderer for iced's wgpu backend
3. **Standalone investigation** — evaluate the reference shaders against our use case (email client = lots of text at various sizes, mixed fonts)

Revisit when text rendering quality becomes a priority or when someone in the iced/wgpu ecosystem picks this up.
