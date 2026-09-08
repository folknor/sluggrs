use crate::glyph_cache::GlyphKey;
use rustc_hash::FxHashMap;

#[derive(Clone, Debug)]
pub(crate) struct ResidentBlob {
    pub start_texel: u32,
    pub texel_len: u32,
    pub kind: BlobKind,
    pub border: Option<ResidentBorderBlob>,
}

#[derive(Clone, Debug)]
pub(crate) struct ResidentBorderBlob {
    pub start_texel: u32,
    pub texel_len: u32,
    /// The blob's two INDEPENDENT capacities: `ppem_ceiling` bounds the
    /// boundary approximation's error, `grid_radius_units` bounds the
    /// distance query. Font units, so one capacity serves every ppem the
    /// glyph is drawn at; a pixel radius would not.
    pub ppem_ceiling: f32,
    pub grid_radius_units: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct CachedBlob {
    pub data: Box<[i32]>,
    pub kind: BlobKind,
    pub border: Option<CachedBorderBlob>,
    pub last_used_epoch: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct CachedBorderBlob {
    pub relative_offset: u32,
    pub texel_len: u32,
    pub ppem_ceiling: f32,
    pub grid_radius_units: f32,
}

#[derive(Clone, Debug)]
pub(crate) enum BlobKind {
    Mono {
        bounds: [f32; 4],
        units_per_em: f32,
    },
    ColorV0 {
        units_per_em: f32,
        layers: Box<[CachedColorLayer]>,
    },
    ColorV1 {
        bounds: [f32; 4],
        units_per_em: f32,
        cmd_texel_count: u32,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct CachedColorLayer {
    pub offset: Option<u32>,
    pub bounds: [f32; 4],
    pub color: [f32; 4],
    pub use_foreground: bool,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct BlobCacheStats {
    pub hits: u64,
    pub budget_evictions: u64,
    pub oversized_drops: u64,
    pub repopulated_bytes: u64,
    pub bytes: usize,
}

pub(crate) struct GlyphBlobCache {
    entries: FxHashMap<GlyphKey, CachedBlob>,
    bytes: usize,
    budget: usize,
    stats: BlobCacheStats,
}

impl GlyphBlobCache {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            entries: FxHashMap::default(),
            bytes: 0,
            budget,
            stats: BlobCacheStats::default(),
        }
    }
    pub(crate) fn take(&mut self, key: &GlyphKey) -> Option<CachedBlob> {
        let entry = self.entries.remove(key)?;
        self.bytes -= entry.data.len() * std::mem::size_of::<i32>();
        self.stats.bytes = self.bytes;
        self.stats.hits += 1;
        Some(entry)
    }
    pub(crate) fn texel_len(&self, key: &GlyphKey) -> Option<u32> {
        self.entries
            .get(key)
            .and_then(|entry| u32::try_from(entry.data.len() / 2).ok())
    }
    pub(crate) fn add_repopulated_bytes(&mut self, bytes: u64) {
        self.stats.repopulated_bytes += bytes;
    }
    pub(crate) fn budget(&self) -> usize {
        self.budget
    }
    /// Install a pre-selected entry set (see `select_within_budget`); the
    /// caller guarantees it fits the budget. Counters accumulate.
    pub(crate) fn replace_entries(
        &mut self,
        entries: Vec<(GlyphKey, CachedBlob)>,
        budget_evictions: u64,
        oversized_drops: u64,
    ) {
        self.entries = entries.into_iter().collect();
        self.bytes = self
            .entries
            .values()
            .map(|entry| entry.data.len() * std::mem::size_of::<i32>())
            .sum();
        self.stats.bytes = self.bytes;
        self.stats.budget_evictions += budget_evictions;
        self.stats.oversized_drops += oversized_drops;
    }
    pub(crate) fn stats(&self) -> BlobCacheStats {
        self.stats
    }
    pub(crate) fn drain(&mut self) -> Vec<(GlyphKey, CachedBlob)> {
        self.bytes = 0;
        self.stats.bytes = 0;
        self.entries.drain().collect()
    }
}

/// Newest-first budget selection over `(last_used_epoch, byte_len)`
/// candidates. Returns the selected candidate indices plus
/// (budget_evictions, oversized_drops) counts for the losers. Pure so the
/// retention policy is unit-testable without a GPU: under pressure the
/// NEWEST candidates must survive.
pub(crate) fn select_within_budget(
    candidates: &[(u64, usize)],
    budget: usize,
) -> (Vec<usize>, u64, u64) {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_unstable_by_key(|&i| (std::cmp::Reverse(candidates[i].0), i));

    let mut selected = Vec::new();
    let mut used = 0usize;
    let mut budget_evictions = 0u64;
    let mut oversized_drops = 0u64;
    for i in order {
        let (_, bytes) = candidates[i];
        if bytes > budget {
            oversized_drops += 1;
        } else if used + bytes <= budget {
            used += bytes;
            selected.push(i);
        } else {
            budget_evictions += 1;
        }
    }
    (selected, budget_evictions, oversized_drops)
}

#[cfg(test)]
mod tests {
    use super::select_within_budget;

    #[test]
    fn newest_candidates_survive_budget_pressure() {
        // One-blob budget, epochs 10 and 5: the NEWEST (epoch 10) must win,
        // regardless of candidate order.
        let (selected, evicted, oversized) = select_within_budget(&[(10, 100), (5, 100)], 100);
        assert_eq!(selected, vec![0]);
        assert_eq!(evicted, 1);
        assert_eq!(oversized, 0);

        let (selected, evicted, oversized) = select_within_budget(&[(5, 100), (10, 100)], 100);
        assert_eq!(selected, vec![1]);
        assert_eq!(evicted, 1);
        assert_eq!(oversized, 0);
    }

    #[test]
    fn byte_accounting_is_exact_at_the_boundary() {
        // 60 + 40 fills the budget exactly; the third-newest is evicted.
        let (selected, evicted, oversized) = select_within_budget(&[(3, 60), (2, 40), (1, 1)], 100);
        assert_eq!(selected, vec![0, 1]);
        assert_eq!(evicted, 1);
        assert_eq!(oversized, 0);
    }

    #[test]
    fn oversized_blobs_are_counted_separately_and_do_not_block_others() {
        let (selected, evicted, oversized) =
            select_within_budget(&[(9, 1000), (5, 50), (4, 50)], 100);
        assert_eq!(selected, vec![1, 2]);
        assert_eq!(evicted, 0);
        assert_eq!(oversized, 1);
    }

    #[test]
    fn a_skipped_larger_blob_does_not_starve_smaller_older_ones() {
        // Newest (80) fits; next (50) does not; the oldest small one still
        // fits the remainder.
        let (selected, evicted, oversized) =
            select_within_budget(&[(9, 80), (8, 50), (7, 20)], 100);
        assert_eq!(selected, vec![0, 2]);
        assert_eq!(evicted, 1);
        assert_eq!(oversized, 0);
    }

    #[test]
    fn equal_epochs_select_in_stable_index_order() {
        let (selected, evicted, oversized) =
            select_within_budget(&[(5, 60), (5, 60), (5, 60)], 120);
        assert_eq!(selected, vec![0, 1]);
        assert_eq!(evicted, 1);
        assert_eq!(oversized, 0);
    }
}
