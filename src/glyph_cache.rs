use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};

/// Cache key for a glyph's outline geometry.
///
/// Captures everything that affects outline shape. Deliberately excludes
/// size, position, and subpixel offset — outlines are resolution-independent.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
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
    /// Font units per em — avoids per-glyph font re-parse on warm path.
    pub units_per_em: f32,
    /// Frame epoch when this glyph was last used (for trim heuristic).
    pub(crate) last_used_epoch: u32,
}

/// Sentinel for glyphs that have no vector outline (emoji, bitmap fonts).
pub const NON_VECTOR_GLYPH: GlyphEntry = GlyphEntry {
    band_offset: u32::MAX,
    band_max_x: 0,
    band_max_y: 0,
    band_transform: [0.0; 4],
    bounds: [0.0; 4],
    units_per_em: 1000.0,
    last_used_epoch: 0,
};

/// Sentinel for COLRv0 color glyphs — look up `color_glyphs` map for layers.
pub const COLOR_VECTOR_GLYPH: GlyphEntry = GlyphEntry {
    band_offset: u32::MAX - 1,
    band_max_x: 0,
    band_max_y: 0,
    band_transform: [0.0; 4],
    bounds: [0.0; 4],
    units_per_em: 1000.0,
    last_used_epoch: 0,
};

/// A single layer in a COLRv0 color glyph.
pub struct ColorGlyphLayer {
    pub entry: GlyphEntry,
    pub color: [f32; 4],
    /// True if this layer uses the foreground text color (palette_index 0xFFFF).
    pub use_foreground: bool,
}

/// A COLRv0 color glyph: multiple solid-colored layers composited back-to-front.
pub struct ColorGlyphEntry {
    pub layers: Vec<ColorGlyphLayer>,
    pub units_per_em: f32,
}

impl GlyphEntry {
    /// Construct a new GlyphEntry. Sets `last_used_epoch` to 0 (not yet used).
    pub fn new(
        band_offset: u32,
        band_max_x: u32,
        band_max_y: u32,
        band_transform: [f32; 4],
        bounds: [f32; 4],
        units_per_em: f32,
    ) -> Self {
        Self {
            band_offset,
            band_max_x,
            band_max_y,
            band_transform,
            bounds,
            units_per_em,
            last_used_epoch: 0,
        }
    }

    pub fn is_non_vector(&self) -> bool {
        self.band_offset == u32::MAX
    }

    pub fn is_color_vector(&self) -> bool {
        self.band_offset == u32::MAX - 1
    }
}

/// The glyph cache maps GlyphKey → GlyphEntry, with per-frame usage tracking
/// via epoch counter (no HashSet).
pub struct GlyphMap {
    map: FxHashMap<GlyphKey, GlyphEntry>,
    current_epoch: u32,
    frame_used: usize,
}

impl Default for GlyphMap {
    fn default() -> Self {
        Self {
            map: FxHashMap::default(),
            current_epoch: 1,
            frame_used: 0,
        }
    }
}

impl GlyphMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a glyph and mark it used this frame. Single hash.
    /// Returns a copy of the entry (GlyphEntry is Copy).
    pub fn get_and_mark_used(&mut self, key: &GlyphKey) -> Option<GlyphEntry> {
        let entry = self.map.get_mut(key)?;
        if entry.last_used_epoch != self.current_epoch {
            entry.last_used_epoch = self.current_epoch;
            self.frame_used += 1;
        }
        Some(*entry)
    }

    /// Insert a new entry and mark it used. Single hash.
    pub fn insert_and_mark_used(&mut self, key: GlyphKey, mut entry: GlyphEntry) -> GlyphEntry {
        entry.last_used_epoch = self.current_epoch;
        self.map.insert(key, entry);
        self.frame_used += 1;
        entry
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.frame_used = 0;
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Read-only lookup. Does NOT mark the glyph as used this frame.
    pub fn get(&self, key: &GlyphKey) -> Option<GlyphEntry> {
        self.map.get(key).copied()
    }

    pub fn contains_key(&self, key: &GlyphKey) -> bool {
        self.map.contains_key(key)
    }

    /// Advance to the next frame. Resets per-frame usage counter.
    pub fn next_frame(&mut self) {
        self.frame_used = 0;
        self.current_epoch = self.current_epoch.wrapping_add(1);
        // Epoch 0 is the sentinel "never used" value
        if self.current_epoch == 0 {
            self.current_epoch = 1;
            for entry in self.map.values_mut() {
                entry.last_used_epoch = 0;
            }
        }
    }

    /// Number of glyphs used this frame.
    pub fn in_use_count(&self) -> usize {
        self.frame_used
    }
}
