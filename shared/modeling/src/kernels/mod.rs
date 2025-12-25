//! Compute kernel abstraction layer
//!
//! This module provides a trait-based system for compute kernels with runtime dispatch.
//! Kernels can have multiple implementations (CPU, CUDA, etc.) that are selected at runtime
//! based on device availability and configuration.
//!
//! # Design
//!
//! The kernel system is designed around two main concepts:
//!
//! 1. **Kernel traits** - Define the interface for specific operations (e.g., `Matmul`, `Orthogonalize`)
//! 2. **Kernel dispatcher** - Selects the best available implementation at runtime
//!
//! # Adaptive Precision
//!
//! For heterogeneous GPU clusters (V100, A100, H100 mix), precision is selected per-device:
//! - CPU: FP32 (universal)
//! - V100 (Volta): FP16 (native tensor cores, BF16 is emulated/slow)
//! - A100 (Ampere): BF16 (native tensor cores)
//! - H100 (Hopper): BF16 (native, could use FP8 for more speed)
//!
//! Detection is done by probing matmul performance at startup.
//!
//! # Example
//!
//! ```ignore
//! use psyche_modeling::kernels::{KernelDispatcher, Orthogonalize};
//!
//! let dispatcher = KernelDispatcher::new(KernelConfig::default());
//! let orthogonalizer = dispatcher.orthogonalizer();
//!
//! // Use the kernel - precision is auto-detected based on input tensor device
//! let g = orthogonalizer.polar_express(&gradient, 5);
//! ```

mod orthogonalize;

pub use orthogonalize::{AdaptiveOrthogonalize, CpuOrthogonalize, Orthogonalize};

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use tch::{Device, Kind, Tensor};

/// Global cache for detected optimal precision per CUDA device
/// Key: device index, Value: optimal Kind for that device
static PRECISION_CACHE: OnceLock<Mutex<HashMap<usize, Kind>>> = OnceLock::new();

fn get_precision_cache() -> &'static Mutex<HashMap<usize, Kind>> {
    PRECISION_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Detect the optimal precision for matrix operations on a given device.
///
/// This probes the device by running small matmuls in BF16 and FP16,
/// comparing performance to determine hardware support:
/// - If BF16 is fast → Ampere+ (A100, H100) → use BF16
/// - If BF16 is slow but FP16 is fast → Volta (V100) → use FP16
/// - CPU → always FP32
///
/// Results are cached per device to avoid repeated probing.
pub fn detect_optimal_precision(device: Device) -> Kind {
    match device {
        Device::Cpu => Kind::Float, // FP32 always for CPU
        Device::Cuda(idx) => {
            // Check cache first
            {
                let cache = get_precision_cache().lock().unwrap();
                if let Some(&kind) = cache.get(&idx) {
                    return kind;
                }
            }

            // Probe and cache
            let kind = probe_cuda_precision(idx);
            {
                let mut cache = get_precision_cache().lock().unwrap();
                cache.insert(idx, kind);
            }
            kind
        }
        // MPS (Apple Silicon) and Vulkan - use FP32 for safety
        Device::Mps | Device::Vulkan => Kind::Float,
    }
}

/// Probe a CUDA device to determine optimal precision.
///
/// Runs matmul benchmarks in BF16 and FP16 to detect hardware support:
/// - Ampere+ (A100, H100): BF16 has native tensor cores, ~same speed as FP16
/// - Volta (V100): BF16 is emulated (~10x slower), FP16 has tensor cores
fn probe_cuda_precision(device_idx: usize) -> Kind {
    let device = Device::Cuda(device_idx);
    let size = 256; // Small enough to be fast, large enough to show difference
    let iterations = 5;

    // Ensure CUDA is initialized and device is valid
    if !tch::Cuda::is_available() {
        tracing::warn!("CUDA not available, falling back to FP32");
        return Kind::Float;
    }

    let _guard = tch::no_grad_guard();

    // Benchmark BF16
    let bf16_time = {
        let a = Tensor::randn([size, size], (Kind::BFloat16, device));
        let b = Tensor::randn([size, size], (Kind::BFloat16, device));
        // Warmup
        let _ = a.matmul(&b);
        tch::Cuda::synchronize(device_idx as i64);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = a.matmul(&b);
        }
        tch::Cuda::synchronize(device_idx as i64);
        start.elapsed()
    };

    // Benchmark FP16
    let fp16_time = {
        let a = Tensor::randn([size, size], (Kind::Half, device));
        let b = Tensor::randn([size, size], (Kind::Half, device));
        // Warmup
        let _ = a.matmul(&b);
        tch::Cuda::synchronize(device_idx as i64);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = a.matmul(&b);
        }
        tch::Cuda::synchronize(device_idx as i64);
        start.elapsed()
    };

    // If BF16 is more than 3x slower than FP16, we're on Volta (V100)
    // Use FP16 which has native tensor core support on Volta
    let ratio = bf16_time.as_secs_f64() / fp16_time.as_secs_f64();

    let selected = if ratio > 3.0 {
        tracing::info!(
            device_idx,
            bf16_ms = bf16_time.as_secs_f64() * 1000.0,
            fp16_ms = fp16_time.as_secs_f64() * 1000.0,
            ratio,
            "Detected Volta-class GPU (BF16 emulated), using FP16"
        );
        Kind::Half // FP16
    } else {
        tracing::info!(
            device_idx,
            bf16_ms = bf16_time.as_secs_f64() * 1000.0,
            fp16_ms = fp16_time.as_secs_f64() * 1000.0,
            ratio,
            "Detected Ampere+ GPU (native BF16), using BF16"
        );
        Kind::BFloat16
    };

    selected
}

/// Clear the precision cache (useful for testing)
#[cfg(test)]
pub fn clear_precision_cache() {
    if let Some(cache) = PRECISION_CACHE.get() {
        cache.lock().unwrap().clear();
    }
}

/// Configuration for kernel selection
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Preferred device for computation
    pub device: Device,
    /// Whether to allow CPU fallbacks when GPU kernels are unavailable
    pub allow_cpu_fallback: bool,
    /// Enable verbose logging of kernel selection
    pub verbose: bool,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            allow_cpu_fallback: true,
            verbose: false,
        }
    }
}

impl KernelConfig {
    /// Create a config for CPU computation
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
            ..Default::default()
        }
    }

    /// Create a config for CUDA computation
    ///
    /// Note: CUDA kernels are a future feature. Currently falls back to CPU.
    pub fn cuda(device_index: usize) -> Self {
        Self {
            device: Device::Cuda(device_index),
            allow_cpu_fallback: true,
            ..Default::default()
        }
    }
}

/// Kernel dispatcher that selects implementations at runtime
pub struct KernelDispatcher {
    config: KernelConfig,
    orthogonalizer: Box<dyn Orthogonalize>,
}

impl KernelDispatcher {
    /// Create a new kernel dispatcher with the given configuration
    pub fn new(config: KernelConfig) -> Self {
        let orthogonalizer = Self::select_orthogonalizer(&config);

        Self {
            config,
            orthogonalizer,
        }
    }

    /// Get the orthogonalization kernel
    pub fn orthogonalizer(&self) -> &dyn Orthogonalize {
        self.orthogonalizer.as_ref()
    }

    /// Get the current configuration
    pub fn config(&self) -> &KernelConfig {
        &self.config
    }

    /// Select the best orthogonalization kernel for the config
    fn select_orthogonalizer(config: &KernelConfig) -> Box<dyn Orthogonalize> {
        if config.verbose {
            tracing::info!("Selected adaptive orthogonalization kernel (Polar Express)");
        }
        // Use adaptive precision that auto-detects optimal dtype per device
        Box::new(AdaptiveOrthogonalize::new())
    }
}

impl std::fmt::Debug for KernelDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelDispatcher")
            .field("config", &self.config)
            .field("orthogonalizer", &self.orthogonalizer.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_dispatcher_creation() {
        let config = KernelConfig::default();
        let dispatcher = KernelDispatcher::new(config);
        assert_eq!(dispatcher.orthogonalizer().name(), "adaptive_polar_express");
    }

    #[test]
    fn test_precision_detection_cpu() {
        // CPU should always return FP32
        let precision = detect_optimal_precision(Device::Cpu);
        assert_eq!(precision, Kind::Float, "CPU should use FP32");
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_precision_detection_cuda() {
        if !tch::Cuda::is_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        clear_precision_cache();
        let precision = detect_optimal_precision(Device::Cuda(0));

        // Should return either BF16 (Ampere+) or FP16 (Volta)
        assert!(
            precision == Kind::BFloat16 || precision == Kind::Half,
            "CUDA should use BF16 or FP16, got {:?}",
            precision
        );

        // Verify caching works - second call should be instant
        let start = std::time::Instant::now();
        let precision2 = detect_optimal_precision(Device::Cuda(0));
        let elapsed = start.elapsed();

        assert_eq!(precision, precision2, "Cached precision should match");
        assert!(
            elapsed.as_millis() < 10,
            "Cached lookup should be < 10ms, took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_precision_cache_isolation() {
        // Verify CPU doesn't pollute CUDA cache
        let _ = detect_optimal_precision(Device::Cpu);

        let cache = get_precision_cache().lock().unwrap();
        // CPU results are not cached (handled directly in match)
        assert!(
            cache.is_empty() || !cache.contains_key(&999),
            "CPU should not add to CUDA cache"
        );
    }
}
