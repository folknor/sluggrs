use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

/// Cache key for a glyph's outline geometry.
///
/// Captures everything that affects outline shape. Deliberately excludes
/// size, position, and subpixel offset — outlines are resolution-independent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlyphKey {
    pub font_id: cosmic_text::fontdb::ID,
    pub glyph_id: u16,
    pub font_weight: u16,
    pub cache_key_flags: cosmic_text::CacheKeyFlags,
}

impl Hash for GlyphKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.font_id.hash(state);
        self.glyph_id.hash(state);
        self.font_weight.hash(state);
        self.cache_key_flags.bits().hash(state);
    }
}

impl GlyphKey {
    pub fn from_layout_glyph(glyph: &cosmic_text::LayoutGlyph) -> Self {
        Self {
            font_id: glyph.font_id,
            glyph_id: glyph.glyph_id,
            font_weight: glyph.font_weight.0,
            cache_key_flags: glyph.cache_key_flags,
        }
    }
}

/// Cached location and metadata for a glyph in the GPU textures.
#[derive(Clone, Copy, Debug)]
pub struct GlyphEntry {
    /// Texel offset of this glyph's band headers in the band texture.
    /// Linear texel space — the shader's calc_band_loc() wraps at
    /// BAND_TEXTURE_WIDTH.
    pub band_offset: u32,
    /// Number of vertical bands minus 1 (band_max.x in shader).
    pub band_max_x: u32,
    /// Number of horizontal bands minus 1 (band_max.y in shader).
    pub band_max_y: u32,
    /// Transform from em-space to band index.
    pub band_transform: [f32; 4],
    /// Glyph bounding box in em-space.
    pub bounds: [f32; 4],
}

/// Sentinel for glyphs that have no vector outline (emoji, bitmap fonts).
pub const NON_VECTOR_GLYPH: GlyphEntry = GlyphEntry {
    band_offset: u32::MAX,
    band_max_x: 0,
    band_max_y: 0,
    band_transform: [0.0; 4],
    bounds: [0.0; 4],
};

impl GlyphEntry {
    pub fn is_non_vector(&self) -> bool {
        self.band_offset == u32::MAX
    }
}

/// The glyph cache maps GlyphKey → GlyphEntry, with per-frame usage tracking.
#[derive(Default)]
pub struct GlyphMap {
    map: HashMap<GlyphKey, GlyphEntry>,
    in_use: HashSet<GlyphKey>,
}

impl GlyphMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &GlyphKey) -> Option<&GlyphEntry> {
        self.map.get(key)
    }

    pub fn insert(&mut self, key: GlyphKey, entry: GlyphEntry) {
        self.map.insert(key, entry);
    }

    pub fn contains_key(&self, key: &GlyphKey) -> bool {
        self.map.contains_key(key)
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.in_use.clear();
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Mark a glyph as used this frame.
    pub fn mark_used(&mut self, key: GlyphKey) {
        self.in_use.insert(key);
    }

    /// Clear the per-frame usage set (called from trim).
    pub fn clear_usage(&mut self) {
        self.in_use.clear();
    }

    /// Number of glyphs used this frame.
    pub fn in_use_count(&self) -> usize {
        self.in_use.len()
    }
}
