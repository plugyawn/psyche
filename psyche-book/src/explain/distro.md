# DisTrO: Distributed Training Optimizer

DisTrO (Distributed Training Optimizer) is a communication-efficient optimizer designed for distributed training across untrusted clients. It achieves ~100-1000x gradient compression while maintaining training convergence.

## Why DisTrO?

Traditional distributed training uses gradient averaging, which requires communicating full gradients between clients. For large models, this creates a bandwidth bottleneck:

- A 7B parameter model has ~14GB of gradients (FP16)
- With 100 clients, each step requires TB of bandwidth
- Internet connections (10-100 Mbps) become the limiting factor

DisTrO addresses this by:
1. Compressing gradients before transmission
2. Using DCT (Discrete Cosine Transform) to concentrate information
3. Keeping only the top-k most significant coefficients
4. Further reducing with 1-bit quantization

## How It Works

### Step 1: Momentum Accumulation

Like standard optimizers, DisTrO maintains a momentum buffer:

```
m_t = β * m_{t-1} + g_t
```

Where:
- `m_t` is the momentum at step t
- `β` is the compression decay (default 0.999)
- `g_t` is the gradient at step t

### Step 2: DCT Transform

The momentum is transformed using the Discrete Cosine Transform:

```
M_dct = DCT(m_t)
```

DCT concentrates signal energy into a small number of low-frequency coefficients, similar to how JPEG compresses images.

### Step 3: Top-k Sparsification

Only the k largest coefficients are kept:

```
M_sparse = top_k(M_dct, k=compression_topk)
```

With `compression_topk=8` and `compression_chunk=64`, this achieves 8x compression from sparsification alone.

### Step 4: 1-bit Quantization

Each coefficient is quantized to its sign:

```
M_quantized = sign(M_sparse)  # -1 or +1
```

Combined with index encoding, this achieves the ~100-1000x total compression.

### Step 5: Aggregation

Clients broadcast their compressed updates via P2P gossip. The server aggregates by:
1. Collecting all compressed updates
2. Reconstructing via inverse DCT
3. Averaging across clients
4. Applying to model weights

## Heterogeneous Gradient Handling

When clients train at different MatFormer tiers, gradients have different shapes. DisTrO handles this through automatic detection and alignment.

### Shape Detection

Before aggregation, DisTrO checks if all gradients have the same shape:

```rust
let same_shape = results.iter().all(|x| {
    x[index].xshape == results[0][index].xshape
        && x[index].totalk == results[0][index].totalk
});
```

### Homogeneous Path (Fast)

When all clients are the same tier, batch decompression uses optimized CUDA kernels:

```rust
let decompressed = CompressDCT::batch_decompress(
    &indicies, &values, &xshape, totalk, val_kind, device,
);
```

### Heterogeneous Path

When tiers differ, each gradient is processed individually and aligned:

```rust
for (peer_results, sparse_val) in results.iter().zip(values.iter()) {
    let decompressed = CompressDCT::decompress(...);
    let decoded = self.transform.decode(&decompressed);
    let aligned = align_matformer_prefix_grad(name, &full_shape, decoded)?;
    combined = Some(match combined {
        Some(acc) => acc + aligned,
        None => aligned,
    });
    contributing_peers += 1;
}

// Normalize by peer count (consistent with "mean" reduction)
let normalized = combined / (contributing_peers as f64);
```

### Dimension-Aware Alignment

FFN layers have different weight matrix orientations:

| Layer | Shape | FFN Width Dimension |
|-------|-------|---------------------|
| `gate_proj`, `up_proj` | `[intermediate_size, hidden_size]` | Rows (dim=0) |
| `down_proj` | `[hidden_size, intermediate_size]` | Columns (dim=1) |

The `matformer_prefix_dim()` function uses parameter names to determine the correct alignment dimension:

```rust
fn matformer_prefix_dim(name: &str) -> Option<usize> {
    if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
        Some(0)  // FFN width in rows
    } else if name.ends_with("down_proj.weight") {
        Some(1)  // FFN width in columns
    } else {
        None     // Non-FFN parameter (attention, embeddings)
    }
}
```

### Gradient Expansion

When a tier-1 client's gradient is smaller than the full model shape, it's expanded with zeros:

```rust
// Tier-1 gradient [512, 256] → Full shape [1024, 256]
let expanded = Tensor::zeros(full_shape, (grad.kind(), grad.device()));
let mut prefix_view = expanded.narrow(prefix_dim, 0, grad_len);
prefix_view.copy_(&grad);  // Prefix filled, suffix zeros
```

### Aggregation Semantics

The heterogeneous path normalizes by `contributing_peers` to match the "mean" semantics of the batch decompress path:

```rust
let normalized = if contributing_peers > 1 {
    combined / (contributing_peers as f64)
} else {
    combined
};
var.set_grad(normalized);
```

This ensures consistent gradient scale regardless of tier mix.

### Sign-SGD Final Step

After aggregation, Sign-SGD extracts only the direction:

```rust
let _t = variable.grad().sign_();
```

This makes the final update invariant to gradient magnitude differences between tiers.

## Configuration

```toml
[model.LLM.optimizer.Distro]
# Maximum gradient norm (prevents exploding gradients)
clip_grad_norm = 1.0

# Momentum decay factor (higher = more smoothing)
compression_decay = 0.999

# DCT chunk size for compression
compression_chunk = 64

# Number of top coefficients to keep per chunk
compression_topk = 8

# Use 1-bit quantization (recommended)
quantize_1bit = true
```

### Parameter Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `compression_decay` | 0.999 | Higher values smooth gradients more, reducing noise but slowing adaptation |
| `compression_topk` | 8 | Higher values transmit more information but increase bandwidth |
| `compression_chunk` | 64 | Larger chunks capture more structure but may miss local patterns |
| `quantize_1bit` | true | Disabling doubles bandwidth but preserves magnitude information |

## Compression Ratio

The effective compression ratio is:

```
ratio = (compression_chunk / compression_topk) * (32 / bit_width)
```

With default settings (`chunk=64`, `topk=8`, `1bit=true`):
- From sparsification: 64/8 = 8x
- From quantization: 32/1 = 32x
- Total: 8 × 32 = 256x compression

## Trade-offs

### Advantages
- Enables training on consumer internet connections
- Reduces P2P bandwidth by orders of magnitude
- Momentum accumulation compensates for compression loss

### Limitations
- Some gradient information is lost
- Requires tuning compression parameters per model size
- Not compatible with all optimizer features (e.g., per-parameter learning rates)

## References

- [DisTrO: Distributed Training Over-the-Internet](https://arxiv.org/abs/2401.XXXXX) - Original paper
- [Sign-based Gradient Compression](https://arxiv.org/abs/1802.04434) - Related compression work
