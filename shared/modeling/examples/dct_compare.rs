use psyche_modeling::{CompressDCT, DistroResult, TransformDCT};
use psyche_network::distro_results_from_reader;
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use tch::{Device, Kind, Tensor};

#[derive(Default, Debug)]
struct Stats {
    sign_matches: f64,
    total: f64,
    dot: f64,
    s_norm: f64,
    l_norm: f64,
}

impl Stats {
    fn accumulate(&mut self, s: &Tensor, l: &Tensor) {
        let s_flat = s.reshape([-1]);
        let l_flat = l.reshape([-1]);
        let s_sign = s_flat.sign();
        let l_sign = l_flat.sign();
        let matches = s_sign
            .eq_tensor(&l_sign)
            .to_kind(Kind::Int64)
            .sum(Kind::Int64)
            .double_value(&[]);
        let total = s_flat.numel() as f64;
        let dot = (&s_flat * &l_flat).sum(Kind::Float).double_value(&[]);
        let s_norm = (&s_flat * &s_flat).sum(Kind::Float).double_value(&[]);
        let l_norm = (&l_flat * &l_flat).sum(Kind::Float).double_value(&[]);

        self.sign_matches += matches;
        self.total += total;
        self.dot += dot;
        self.s_norm += s_norm;
        self.l_norm += l_norm;
    }

    fn sign_agreement(&self) -> f64 {
        if self.total == 0.0 {
            0.0
        } else {
            self.sign_matches / self.total
        }
    }

    fn cosine(&self) -> f64 {
        let denom = (self.s_norm * self.l_norm).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            self.dot / denom
        }
    }
}

fn prefix_dim(name: &str) -> Option<usize> {
    if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
        Some(0)
    } else if name.ends_with("down_proj.weight") {
        Some(1)
    } else {
        None
    }
}

fn align_overlap(name: &str, s: &Tensor, l: &Tensor) -> Option<(Tensor, Tensor)> {
    if s.size() == l.size() {
        return Some((s.shallow_clone(), l.shallow_clone()));
    }
    let dim = prefix_dim(name)?;
    let s_shape = s.size();
    let l_shape = l.size();
    if s_shape.len() != l_shape.len() {
        return None;
    }
    for (i, (s_dim, l_dim)) in s_shape.iter().zip(l_shape.iter()).enumerate() {
        if i != dim && s_dim != l_dim {
            return None;
        }
    }
    let target = s_shape[dim].min(l_shape[dim]);
    let s_aligned = s.narrow(dim as i64, 0, target);
    let l_aligned = l.narrow(dim as i64, 0, target);
    Some((s_aligned, l_aligned))
}

fn decode_result(path: &PathBuf, transform: &mut TransformDCT) -> anyhow::Result<HashMap<String, Tensor>> {
    let file = File::open(path)?;
    let mut map = HashMap::new();
    let device = Device::Cpu;
    for item in distro_results_from_reader(file) {
        let serialized = item?;
        let name = serialized.parameter_name.clone();
        if name != "model.layers.0.mlp.gate_proj.weight"
            && name != "model.layers.0.mlp.up_proj.weight"
            && name != "model.layers.0.mlp.down_proj.weight"
        {
            continue;
        }
        let result: DistroResult = (&serialized).try_into()?;
        let idx = result.sparse_idx.to_device(device);
        let val = if result.sparse_val.kind() == Kind::Bool {
            result.sparse_val.to_device(device).to_kind(Kind::Float) * -2.0 + 1.0
        } else {
            result.sparse_val.to_device(device).to_kind(Kind::Float)
        };
        let decompressed = CompressDCT::decompress(
            &idx,
            &val,
            &result.xshape,
            result.totalk,
            Kind::Float,
            device,
        );
        let decoded = transform.decode(&decompressed);
        map.insert(name, decoded);
    }
    Ok(map)
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let results_dir = PathBuf::from(args.next().expect("results dir"));
    let chunk: i64 = args
        .next()
        .map(|v| v.parse().expect("chunk"))
        .unwrap_or(64);

    let mut steps_filter: Option<Vec<u32>> = None;
    if let Some(filter) = args.next() {
        let steps = filter
            .split(',')
            .filter_map(|s| s.parse::<u32>().ok())
            .collect::<Vec<_>>();
        if !steps.is_empty() {
            steps_filter = Some(steps);
        }
    }

    let re = Regex::new(r"result-([0-9a-f]+)-step(\d+)-batch([A-Za-z0-9]+)\.vec-postcard")?;
    let mut steps: HashMap<u32, Vec<PathBuf>> = HashMap::new();

    for entry in std::fs::read_dir(&results_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = match path.file_name().and_then(|x| x.to_str()) {
            Some(name) => name,
            None => continue,
        };
        let caps = match re.captures(file_name) {
            Some(caps) => caps,
            None => continue,
        };
        let step: u32 = caps.get(2).unwrap().as_str().parse()?;
        if let Some(filter) = &steps_filter {
            if !filter.contains(&step) {
                continue;
            }
        }
        steps.entry(step).or_default().push(path);
    }

    let mut transform = TransformDCT::new(Box::new(std::iter::empty()), chunk);
    let mut stats = Stats::default();
    let mut steps_used = 0u64;

    let mut step_keys: Vec<u32> = steps.keys().copied().collect();
    step_keys.sort_unstable();

    for step in step_keys {
        let files = steps.get(&step).unwrap();
        let mut client_maps = Vec::new();
        for path in files {
            let decoded = decode_result(path, &mut transform)?;
            if !decoded.is_empty() {
                client_maps.push(decoded);
            }
        }
        if client_maps.len() < 2 {
            continue;
        }

        let mut sizes = Vec::new();
        for m in &client_maps {
            if let Some(t) = m.get("model.layers.0.mlp.gate_proj.weight") {
                sizes.push(t.numel());
            }
        }
        if sizes.is_empty() {
            continue;
        }
        let max_size = *sizes.iter().max().unwrap();

        let mut l_map: Option<HashMap<String, Tensor>> = None;
        let mut s_maps: Vec<HashMap<String, Tensor>> = Vec::new();
        for m in client_maps {
            let gate = match m.get("model.layers.0.mlp.gate_proj.weight") {
                Some(t) => t,
                None => continue,
            };
            if gate.numel() == max_size && l_map.is_none() {
                l_map = Some(m);
            } else {
                s_maps.push(m);
            }
        }
        let l_map = match l_map {
            Some(m) => m,
            None => continue,
        };
        if s_maps.is_empty() {
            continue;
        }
        steps_used += 1;

        let mut s_sum: HashMap<String, Tensor> = HashMap::new();
        for s in s_maps {
            for (name, tensor) in s {
                if let Some(existing) = s_sum.get_mut(&name) {
                    let _ = existing.f_add_(&tensor);
                } else {
                    s_sum.insert(name, tensor);
                }
            }
        }

        for (name, s_tensor) in s_sum {
            let Some(l_tensor) = l_map.get(&name) else { continue; };
            let Some((s_aligned, l_aligned)) = align_overlap(&name, &s_tensor, l_tensor) else {
                continue;
            };
            stats.accumulate(&s_aligned, &l_aligned);
        }
    }

    println!("steps_used={}", steps_used);
    println!("sign_agreement={}", stats.sign_agreement());
    println!("cosine={}", stats.cosine());
    println!("total_elems={}", stats.total as u64);

    Ok(())
}
