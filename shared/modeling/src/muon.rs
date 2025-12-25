//! Muon Optimizer Implementation
//!
//! Muon is an optimizer that applies Polar Express orthogonalization to momentum.
//! It uses Nesterov-style momentum with orthogonalization for 2D (matrix) parameters,
//! and falls back to AdamW for non-2D parameters (embeddings, norms, biases).
//!
//! # Algorithm
//!
//! For 2D parameters (where Polar Express can be applied):
//! ```text
//! m_t = β * m_{t-1} + (1 - β) * g_t
//! m_ortho = PolarExpress(m_t, 5)
//! θ_t = θ_{t-1} - lr * m_ortho - wd * θ_{t-1}
//! ```
//!
//! For 1D/0D parameters (embeddings, norms, biases):
//! ```text
//! Standard AdamW update
//! ```
//!
//! # References
//!
//! - NorMuon optimizer in modded-nanogpt
//! - "Shampoo: Preconditioned Stochastic Tensor Optimization" (related work)

use crate::{
    CausalLM,
    kernels::{CpuOrthogonalize, Orthogonalize},
};
use regex::Regex;
use std::collections::HashMap;
use tch::{Kind, Tensor};

/// Configuration for a parameter group in Muon
#[derive(Debug, Clone)]
pub struct MuonParamGroupConfig {
    /// Regex pattern for matching parameter names
    pub pattern: String,
    /// Whether to use Muon (true) or AdamW fallback (false)
    pub use_muon: bool,
    /// Whether to apply Polar Express orthogonalization (only for 2D)
    pub orthogonalize: bool,
}

impl MuonParamGroupConfig {
    /// Create a Muon param group with orthogonalization
    pub fn muon(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            use_muon: true,
            orthogonalize: true,
        }
    }

    /// Create an AdamW fallback param group
    pub fn adamw(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            use_muon: false,
            orthogonalize: false,
        }
    }
}

/// Configuration for the Muon optimizer
#[derive(Debug, Clone)]
pub struct MuonConfig {
    /// Learning rate
    pub lr: f64,
    /// Momentum coefficient for Muon (typically 0.95)
    pub momentum: f64,
    /// Weight decay (decoupled, applied to weights directly)
    pub weight_decay: f64,
    /// Beta1 for AdamW fallback parameters
    pub adamw_beta1: f64,
    /// Beta2 for AdamW fallback parameters
    pub adamw_beta2: f64,
    /// Epsilon for AdamW numerical stability
    pub adamw_eps: f64,
    /// Gradient clipping norm (None = no clipping)
    pub clip_grad_norm: Option<f64>,
    /// Parameter groups (ordered, first match wins)
    pub param_groups: Vec<MuonParamGroupConfig>,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            momentum: 0.95,
            weight_decay: 0.0,
            adamw_beta1: 0.9,
            adamw_beta2: 0.999,
            adamw_eps: 1e-8,
            clip_grad_norm: Some(1.0),
            // Default param groups matching modded-nanogpt:
            // - Embeddings use AdamW
            // - Norms use AdamW
            // - lm_head uses AdamW
            // - Everything else uses Muon with orthogonalization
            param_groups: vec![
                MuonParamGroupConfig::adamw(".*embed.*"),
                MuonParamGroupConfig::adamw(".*norm.*"),
                MuonParamGroupConfig::adamw(".*lm_head.*"),
                MuonParamGroupConfig::adamw(".*bias.*"),
                MuonParamGroupConfig::muon(".*"), // Catch-all for matrices
            ],
        }
    }
}

impl MuonConfig {
    /// Create a config with custom learning rate
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Create a config with custom momentum
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Create a config with custom weight decay
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Create a config with custom gradient clipping
    pub fn with_clip_grad_norm(mut self, clip: Option<f64>) -> Self {
        self.clip_grad_norm = clip;
        self
    }
}

/// Per-parameter optimizer state
#[derive(Debug)]
struct ParamState {
    /// Which param group this parameter belongs to
    group_idx: usize,
    /// Momentum buffer for Muon
    momentum_buffer: Option<Tensor>,
    /// Exponential moving average of gradients (for AdamW)
    exp_avg: Option<Tensor>,
    /// Exponential moving average of squared gradients (for AdamW)
    exp_avg_sq: Option<Tensor>,
    /// Whether this parameter is 2D (can be orthogonalized)
    is_2d: bool,
}

/// The Muon optimizer
pub struct Muon {
    config: MuonConfig,
    states: HashMap<String, ParamState>,
    compiled_patterns: Vec<Regex>,
    orthogonalizer: Box<dyn Orthogonalize>,
    step: u64,
}

impl Muon {
    /// Create a new Muon optimizer with the given configuration
    pub fn new(config: MuonConfig) -> Self {
        // Pre-compile regex patterns
        let compiled_patterns: Vec<Regex> = config
            .param_groups
            .iter()
            .map(|pg| Regex::new(&pg.pattern).expect("Invalid param group regex pattern"))
            .collect();

        Self {
            config,
            states: HashMap::new(),
            compiled_patterns,
            orthogonalizer: Box::new(CpuOrthogonalize::new()),
            step: 0,
        }
    }

    /// Get the parameter group index for a given parameter name
    fn get_param_group(&self, name: &str) -> usize {
        for (idx, pattern) in self.compiled_patterns.iter().enumerate() {
            if pattern.is_match(name) {
                return idx;
            }
        }
        // Should not happen if there's a catch-all pattern
        self.config.param_groups.len() - 1
    }

    /// Initialize state for a parameter
    fn init_state(&mut self, name: &str, tensor: &Tensor) -> &mut ParamState {
        let group_idx = self.get_param_group(name);
        let is_2d = tensor.dim() == 2;
        let device = tensor.device();
        let kind = Kind::Float; // State always in FP32

        let group = &self.config.param_groups[group_idx];

        let state = if group.use_muon {
            // Muon: only need momentum buffer
            ParamState {
                group_idx,
                momentum_buffer: Some(Tensor::zeros(tensor.size(), (kind, device))),
                exp_avg: None,
                exp_avg_sq: None,
                is_2d,
            }
        } else {
            // AdamW: need exp_avg and exp_avg_sq
            ParamState {
                group_idx,
                momentum_buffer: None,
                exp_avg: Some(Tensor::zeros(tensor.size(), (kind, device))),
                exp_avg_sq: Some(Tensor::zeros(tensor.size(), (kind, device))),
                is_2d,
            }
        };

        self.states.insert(name.to_string(), state);
        self.states.get_mut(name).unwrap()
    }

    /// Perform a single optimization step
    ///
    /// # Arguments
    ///
    /// * `model` - The model to optimize
    /// * `lr` - Learning rate for this step (allows LR scheduling)
    pub fn step(&mut self, model: &dyn CausalLM, lr: f64) {
        self.step += 1;
        let current_step = self.step;

        let _guard = tch::no_grad_guard();

        // Copy config values to avoid borrow issues
        let momentum = self.config.momentum;
        let weight_decay = self.config.weight_decay;
        let adamw_beta1 = self.config.adamw_beta1;
        let adamw_beta2 = self.config.adamw_beta2;
        let adamw_eps = self.config.adamw_eps;

        // Optional gradient clipping
        if let Some(max_norm) = self.config.clip_grad_norm {
            model.clip_grad_norm(max_norm);
        }

        // First pass: initialize any missing states
        for var in model.variables() {
            let name = var.name();
            let tensor = var.local_tensor();
            if !self.states.contains_key(name) {
                self.init_state(name, &tensor);
            }
        }

        // Second pass: perform updates
        for var in model.variables() {
            let name = var.name();
            let mut tensor = var.local_tensor();
            let grad = tensor.grad();

            if !grad.defined() {
                continue;
            }

            let state = self.states.get_mut(name).unwrap();
            let group_idx = state.group_idx;
            let is_2d = state.is_2d;
            let use_muon = self.config.param_groups[group_idx].use_muon;
            let orthogonalize = self.config.param_groups[group_idx].orthogonalize;

            // Convert gradient to FP32 for state updates
            let grad_fp32 = grad.to_kind(Kind::Float);

            if use_muon {
                // Muon update
                let momentum_buffer = state.momentum_buffer.as_mut().unwrap();

                // m_t = β * m_{t-1} + (1 - β) * g_t
                let _ = momentum_buffer.g_mul_scalar_(momentum);
                let grad_scaled = &grad_fp32 * (1.0 - momentum);
                let _ = momentum_buffer.g_add_(&grad_scaled);

                // Orthogonalize if 2D and configured
                let update = if is_2d && orthogonalize {
                    self.orthogonalizer.polar_express(momentum_buffer, 5)
                } else {
                    momentum_buffer.shallow_clone()
                };

                // Apply update: θ = θ - lr * update
                let update = update.to_kind(tensor.kind());
                let update_scaled = &update * (-lr);
                let _ = tensor.g_add_(&update_scaled);

                // Apply decoupled weight decay: θ = θ * (1 - wd * lr)
                if weight_decay > 0.0 {
                    let decay_factor = 1.0 - weight_decay * lr;
                    let _ = tensor.g_mul_scalar_(decay_factor);
                }
            } else {
                // AdamW update
                let exp_avg = state.exp_avg.as_mut().unwrap();
                let exp_avg_sq = state.exp_avg_sq.as_mut().unwrap();

                // m_t = β1 * m_{t-1} + (1 - β1) * g_t
                let _ = exp_avg.g_mul_scalar_(adamw_beta1);
                let grad_scaled = &grad_fp32 * (1.0 - adamw_beta1);
                let _ = exp_avg.g_add_(&grad_scaled);

                // v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                let _ = exp_avg_sq.g_mul_scalar_(adamw_beta2);
                let grad_sq = &grad_fp32 * &grad_fp32;
                let grad_sq_scaled = &grad_sq * (1.0 - adamw_beta2);
                let _ = exp_avg_sq.g_add_(&grad_sq_scaled);

                // Bias correction
                let bias_correction1 = 1.0 - adamw_beta1.powi(current_step as i32);
                let bias_correction2 = 1.0 - adamw_beta2.powi(current_step as i32);

                // Compute update: m_hat / (sqrt(v_hat) + eps)
                let exp_avg_corrected = exp_avg.shallow_clone() / bias_correction1;
                let exp_avg_sq_corrected = exp_avg_sq.shallow_clone() / bias_correction2;
                let denom = exp_avg_sq_corrected.sqrt() + adamw_eps;
                let update = exp_avg_corrected / denom;

                // Apply update
                let update = update.to_kind(tensor.kind());
                let update_scaled = &update * (-lr);
                let _ = tensor.g_add_(&update_scaled);

                // Apply decoupled weight decay: θ = θ * (1 - wd * lr)
                if weight_decay > 0.0 {
                    let decay_factor = 1.0 - weight_decay * lr;
                    let _ = tensor.g_mul_scalar_(decay_factor);
                }
            }
        }
    }

    /// Zero all gradients in the model
    pub fn zero_grad(&self, model: &dyn CausalLM) {
        for var in model.variables() {
            let tensor = var.local_tensor();
            if tensor.grad().defined() {
                let _ = tensor.grad().zero_();
            }
        }
    }

    /// Get the current step count
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Get the configuration
    pub fn config(&self) -> &MuonConfig {
        &self.config
    }
}

impl std::fmt::Debug for Muon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Muon")
            .field("config", &self.config)
            .field("step", &self.step)
            .field("num_params", &self.states.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_muon_config_default() {
        let config = MuonConfig::default();
        assert_eq!(config.momentum, 0.95);
        assert_eq!(config.adamw_beta1, 0.9);
        assert!(!config.param_groups.is_empty());
    }

    #[test]
    fn test_param_group_matching() {
        let config = MuonConfig::default();
        let muon = Muon::new(config);

        // Test pattern matching
        assert_eq!(muon.get_param_group("model.embed_tokens.weight"), 0); // embed
        assert_eq!(muon.get_param_group("model.layers.0.input_layernorm.weight"), 1); // norm
        assert_eq!(muon.get_param_group("lm_head.weight"), 2); // lm_head
        assert_eq!(muon.get_param_group("model.layers.0.mlp.gate_proj.weight"), 4); // catch-all
    }
}
