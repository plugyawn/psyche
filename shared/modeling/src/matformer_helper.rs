//! Stochastic suffix sampling for MatFormer helper mode.
//!
//! This module implements "helper mode" where smaller-tier clients contribute
//! gradients not just for their own prefix neurons, but also for a stochastically
//! sampled subset of suffix neurons that only larger tiers normally train.
//!
//! Key properties:
//! - Deterministic: Same inputs → same outputs across all nodes
//! - Rotation epochs: Indices fixed for H rounds, then rotate for full coverage
//! - No communication: Indices derived from round number, not transmitted

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Configuration for MatFormer helper mode.
#[derive(Debug, Clone, Copy)]
pub struct HelperConfig {
    /// Fraction of prefix size to sample from suffix (e.g., 0.25 = 25% extra neurons)
    pub helper_fraction: f32,
    /// How many rounds before rotating to new suffix sample
    pub rotation_interval: u64,
}

impl Default for HelperConfig {
    fn default() -> Self {
        Self {
            helper_fraction: 0.0,
            rotation_interval: 16,
        }
    }
}

impl HelperConfig {
    /// Create a new helper config.
    pub fn new(helper_fraction: f32, rotation_interval: u64) -> Self {
        Self {
            helper_fraction,
            rotation_interval,
        }
    }

    /// Check if helper mode is enabled.
    pub fn is_enabled(&self) -> bool {
        self.helper_fraction > 0.0
    }
}

/// Generate deterministic helper indices for a given rotation epoch.
///
/// Indices stay FIXED for `rotation_interval` rounds, then rotate.
/// This allows DisTrO's delta accumulation to build meaningful signal.
///
/// # Arguments
/// * `prefix_end` - Where this tier's prefix ends (e.g., 768 for tier 2)
/// * `full_size` - Full FFN intermediate width (e.g., 3072)
/// * `helper_count` - How many helper neurons to sample
/// * `round` - Current training round number
/// * `rotation_interval` - Rounds before rotating (e.g., 16)
/// * `layer_idx` - Layer index (different layers get different samples)
///
/// # Returns
/// Sorted vector of suffix indices to include as helpers.
pub fn generate_helper_indices(
    prefix_end: usize,
    full_size: usize,
    helper_count: usize,
    round: u64,
    rotation_interval: u64,
    layer_idx: usize,
) -> Vec<i64> {
    // Compute rotation epoch - indices are fixed within each epoch
    let rotation_epoch = round / rotation_interval.max(1);

    // Create deterministic seed from epoch + layer (NOT round!)
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    rotation_epoch.hash(&mut hasher);
    layer_idx.hash(&mut hasher);
    let seed = hasher.finish();

    // Simple LCG PRNG for deterministic sampling
    let mut rng_state = seed;
    let lcg_next = |state: &mut u64| -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    };

    // Sample without replacement from [prefix_end, full_size)
    let suffix_size = full_size.saturating_sub(prefix_end);
    if suffix_size == 0 {
        return Vec::new();
    }

    if helper_count >= suffix_size {
        // If requesting more than available, return all suffix indices
        return (prefix_end as i64..full_size as i64).collect();
    }

    let mut selected: HashSet<usize> = HashSet::with_capacity(helper_count);
    while selected.len() < helper_count {
        let idx = prefix_end + (lcg_next(&mut rng_state) as usize % suffix_size);
        selected.insert(idx);
    }

    let mut indices: Vec<i64> = selected.into_iter().map(|x| x as i64).collect();
    indices.sort(); // Sort for potentially better memory access patterns
    indices
}

/// Combine prefix indices with helper indices for MatFormer forward pass.
///
/// # Arguments
/// * `prefix_size` - Size of this tier's prefix (e.g., 768)
/// * `full_size` - Full FFN intermediate width (e.g., 3072)
/// * `config` - Helper configuration
/// * `round` - Current training round
/// * `layer_idx` - Layer index
///
/// # Returns
/// Vector of all indices to use: [0..prefix_size) + sampled suffix indices
pub fn get_matformer_indices(
    prefix_size: usize,
    full_size: usize,
    config: &HelperConfig,
    round: u64,
    layer_idx: usize,
) -> Vec<i64> {
    // Prefix indices (always included)
    let mut indices: Vec<i64> = (0..prefix_size as i64).collect();

    if config.helper_fraction > 0.0 && prefix_size < full_size {
        let helper_count = ((prefix_size as f32) * config.helper_fraction) as usize;
        if helper_count > 0 {
            let helper_indices = generate_helper_indices(
                prefix_size,
                full_size,
                helper_count,
                round,
                config.rotation_interval,
                layer_idx,
            );
            indices.extend(helper_indices);
        }
    }

    indices
}

/// Calculate how many rotation epochs needed for expected full suffix coverage.
///
/// This is a statistical estimate - actual coverage depends on random sampling.
pub fn epochs_for_full_coverage(suffix_size: usize, helper_count: usize) -> usize {
    if helper_count == 0 {
        return usize::MAX;
    }
    // ceil division
    (suffix_size + helper_count - 1) / helper_count
}

/// Calculate total rounds needed for expected full suffix coverage.
pub fn rounds_for_full_coverage(
    suffix_size: usize,
    helper_count: usize,
    rotation_interval: u64,
) -> u64 {
    epochs_for_full_coverage(suffix_size, helper_count) as u64 * rotation_interval
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helper_indices_deterministic() {
        let indices1 = generate_helper_indices(768, 3072, 256, 42, 16, 5);
        let indices2 = generate_helper_indices(768, 3072, 256, 42, 16, 5);
        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_helper_indices_different_rounds_same_epoch() {
        // Rounds 0-15 should all produce the same indices (epoch 0)
        let indices_r0 = generate_helper_indices(768, 3072, 256, 0, 16, 5);
        let indices_r5 = generate_helper_indices(768, 3072, 256, 5, 16, 5);
        let indices_r15 = generate_helper_indices(768, 3072, 256, 15, 16, 5);
        assert_eq!(indices_r0, indices_r5);
        assert_eq!(indices_r0, indices_r15);
    }

    #[test]
    fn test_helper_indices_different_epochs() {
        // Round 0 (epoch 0) vs round 16 (epoch 1) should differ
        let indices_epoch0 = generate_helper_indices(768, 3072, 256, 0, 16, 5);
        let indices_epoch1 = generate_helper_indices(768, 3072, 256, 16, 16, 5);
        assert_ne!(indices_epoch0, indices_epoch1);
    }

    #[test]
    fn test_helper_indices_in_suffix_range() {
        let indices = generate_helper_indices(768, 3072, 256, 42, 16, 5);
        assert_eq!(indices.len(), 256);
        for &idx in &indices {
            assert!(
                idx >= 768 && idx < 3072,
                "Index {} out of suffix range",
                idx
            );
        }
    }

    #[test]
    fn test_helper_indices_sorted() {
        let indices = generate_helper_indices(768, 3072, 256, 42, 16, 5);
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(indices, sorted);
    }

    #[test]
    fn test_helper_indices_no_duplicates() {
        let indices = generate_helper_indices(768, 3072, 256, 42, 16, 5);
        let set: HashSet<_> = indices.iter().collect();
        assert_eq!(indices.len(), set.len());
    }

    #[test]
    fn test_helper_indices_different_layers() {
        let indices_l0 = generate_helper_indices(768, 3072, 256, 42, 16, 0);
        let indices_l5 = generate_helper_indices(768, 3072, 256, 42, 16, 5);
        assert_ne!(indices_l0, indices_l5);
    }

    #[test]
    fn test_helper_indices_oversized_request() {
        // Request more than available suffix size
        let indices = generate_helper_indices(768, 1024, 500, 42, 16, 5);
        // Should return all suffix indices
        let expected: Vec<i64> = (768..1024).collect();
        assert_eq!(indices, expected);
    }

    #[test]
    fn test_helper_indices_no_suffix() {
        // prefix_end == full_size means no suffix
        let indices = generate_helper_indices(3072, 3072, 256, 42, 16, 5);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_get_matformer_indices_no_helper() {
        let config = HelperConfig::new(0.0, 16);
        let indices = get_matformer_indices(768, 3072, &config, 42, 5);
        let expected: Vec<i64> = (0..768).collect();
        assert_eq!(indices, expected);
    }

    #[test]
    fn test_get_matformer_indices_with_helper() {
        let config = HelperConfig::new(0.25, 16);
        let indices = get_matformer_indices(768, 3072, &config, 42, 5);
        // Should have prefix (768) + helper (768 * 0.25 = 192)
        assert_eq!(indices.len(), 768 + 192);
        // First 768 should be prefix
        for i in 0..768 {
            assert_eq!(indices[i], i as i64);
        }
        // Rest should be in suffix range
        for &idx in &indices[768..] {
            assert!(idx >= 768 && idx < 3072);
        }
    }

    #[test]
    fn test_epochs_for_full_coverage() {
        // 2304 suffix neurons, 192 helpers per epoch
        let epochs = epochs_for_full_coverage(2304, 192);
        assert_eq!(epochs, 12); // 2304 / 192 = 12
    }

    #[test]
    fn test_rounds_for_full_coverage() {
        let rounds = rounds_for_full_coverage(2304, 192, 16);
        assert_eq!(rounds, 192); // 12 epochs * 16 rounds/epoch
    }
}
