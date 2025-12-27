# Optimizers

Psyche supports multiple optimizers for different training scenarios.

## DisTrO (Distributed Training)

The primary optimizer for Psyche distributed training. See [DisTrO optimizer](../explain/distro.md) for details.

```toml
[model.LLM.optimizer.Distro]
clip_grad_norm = 1.0
compression_decay = 0.999
compression_chunk = 64
compression_topk = 8
quantize_1bit = true
```

## AdamW (Local Training)

Standard AdamW optimizer, useful for local testing and development.

```toml
[model.LLM.optimizer.AdamW]
betas = [0.9, 0.95]
weight_decay = 0.1
eps = 1e-8
clip_grad_norm = 1.0
```

## Muon (Experimental)

Muon is an optimizer that orthogonalizes the momentum buffer before applying updates, inspired by modded-nanogpt. It uses the Polar Express algorithm for fast orthogonalization.

### How Muon Works

1. **Momentum accumulation**: Like AdamW, accumulates gradient momentum
2. **Orthogonalization**: Uses Polar Express to find nearest orthogonal matrix
3. **Weight update**: Applies orthogonalized momentum as update

The key insight: constraining updates to be "rotation-like" (orthogonal) empirically converges faster for transformer weights.

### Implementation

Muon is implemented in `shared/modeling/src/muon.rs` with configurable parameter groups:

```rust
pub struct MuonConfig {
    pub lr: f64,
    pub momentum: f64,        // 0.95 default
    pub weight_decay: f64,
    pub adamw_beta1: f64,     // For fallback params
    pub adamw_beta2: f64,
    pub param_groups: Vec<MuonParamGroupConfig>,
}
```

Parameter groups determine which parameters use Muon vs AdamW:
- **Muon**: 2D weight matrices (attention, MLP projections)
- **AdamW fallback**: Embeddings, layer norms, 1D parameters

### Polar Express Orthogonalization

The orthogonalization kernel in `shared/modeling/src/kernels/orthogonalize.rs` implements the Polar Express algorithm from "Polar Express: High Performance Sign Methods for Low Precision Training".

```rust
// Polar Express with 5 iterations
const POLAR_COEFFS: [(f64, f64, f64); 5] = [
    (8.157, -22.483, 15.879),
    (4.043, -2.809, 0.500),
    // ...
];

pub fn polar_express(g: &Tensor, iterations: usize) -> Tensor {
    let mut x = g / (g.norm() * 1.02 + 1e-6);  // Normalize
    for (a, b, c) in POLAR_COEFFS.iter().take(iterations) {
        let xxt = x.matmul(&x.t());
        let b_mat = xxt * b + xxt.matmul(&xxt) * c;
        x = x * a + b_mat.matmul(&x);
    }
    x
}
```

Key properties:
- 5 iterations are sufficient for good orthogonality
- Works in BF16 for speed
- Pre-computed coefficients avoid per-step optimization

### Muon + DisTrO Integration

Muon and DisTrO have a fundamental tension:
- **Muon** relies on orthogonality for its optimization properties
- **DisTrO** compression destroys orthogonality

Experimental approaches being explored in `experiment/muon-distro-composition` branch:
1. **Momentum only**: Skip orthogonalization, use Muon's momentum with DisTrO compression
2. **Server orthogonalization**: Compress, aggregate, then orthogonalize server-side
3. **Hybrid layers**: Muon for attention, DisTrO for MLP
4. **Periodic sync**: Full orthogonalization every N steps

## Adding New Optimizers

To add a new optimizer:

1. Add variant to `OptimizerDefinition` in `shared/core/src/definitions.rs`
2. Implement optimizer in `shared/modeling/src/`
3. Add integration in `shared/modeling/src/optimizer.rs`
4. Update config parsing in `shared/coordinator/src/model.rs`
