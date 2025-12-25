#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/psyche-env.sh
source "${ROOT}/scripts/psyche-env.sh"

OUT_DIR="${1:-${ROOT}/dist/psyche-centralized-mac}"
TAR_PATH="${2:-${ROOT}/dist/psyche-centralized-mac.tar.gz}"

mkdir -p "$(dirname "${OUT_DIR}")"
if [[ -e "${OUT_DIR}" ]]; then
  # Bundles often contain a `.venv/` that may have macOS flags/xattrs that make deletion flaky.
  # Try to clear flags + make files writable before deleting, then retry removal once.
  chflags -R nouchg,noschg "${OUT_DIR}" 2>/dev/null || true
  chmod -R u+w "${OUT_DIR}" 2>/dev/null || true
  rm -rf "${OUT_DIR}" 2>/dev/null || true
  rm -rf "${OUT_DIR}" 2>/dev/null || true
  if [[ -e "${OUT_DIR}" ]]; then
    echo "[package] Failed to remove ${OUT_DIR}. Close any processes using it and delete it manually." >&2
    exit 1
  fi
fi
mkdir -p "${OUT_DIR}/bin" "${OUT_DIR}/scripts" "${OUT_DIR}/config" "${OUT_DIR}/logs"

echo "[package] building release binaries..."
cargo build --release -p psyche-centralized-server -p psyche-centralized-client

echo "[package] copying binaries..."
cp "${CARGO_TARGET_DIR}/release/psyche-centralized-server" "${OUT_DIR}/bin/"
cp "${CARGO_TARGET_DIR}/release/psyche-centralized-client" "${OUT_DIR}/bin/"

echo "[package] copying helper scripts..."
cp "${ROOT}/scripts/psyche-env.sh" "${OUT_DIR}/scripts/"
cp "${ROOT}/scripts/psyche-centralized-spawn-clients.sh" "${OUT_DIR}/scripts/"
cp "${ROOT}/scripts/export_matformer_tiers.py" "${OUT_DIR}/scripts/"
cp "${ROOT}/scripts/bootstrap-python-runtime.sh" "${OUT_DIR}/scripts/"
cp "${ROOT}/scripts/unquarantine.sh" "${OUT_DIR}/scripts/" || true
cp "${ROOT}/scripts/prepare_tinyshakespeare_bin_dataset.py" "${OUT_DIR}/scripts/" || true

cat > "${OUT_DIR}/scripts/run-server.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
# shellcheck source=scripts/psyche-env.sh
source "${HERE}/psyche-env.sh"
exec "${ROOT}/bin/psyche-centralized-server" "$@"
SH

cat > "${OUT_DIR}/scripts/run-client.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
# shellcheck source=scripts/psyche-env.sh
source "${HERE}/psyche-env.sh"
exec "${ROOT}/bin/psyche-centralized-client" "$@"
SH

chmod +x "${OUT_DIR}/scripts/run-server.sh" "${OUT_DIR}/scripts/run-client.sh" "${OUT_DIR}/scripts/bootstrap-python-runtime.sh" "${OUT_DIR}/scripts/unquarantine.sh" || true

echo "[package] copying configs (tiny-llama)..."
if [[ -d "${ROOT}/config/test-tiny-llama" ]]; then
  mkdir -p "${OUT_DIR}/config/test-tiny-llama"
  cp "${ROOT}/config/test-tiny-llama/state.toml" "${OUT_DIR}/config/test-tiny-llama/" || true
  cp "${ROOT}/config/test-tiny-llama/README.md" "${OUT_DIR}/config/test-tiny-llama/" || true
fi

echo "[package] copying configs (tiny-llama-shakespeare)..."
if [[ -d "${ROOT}/config/test-tiny-llama-shakespeare" ]]; then
  mkdir -p "${OUT_DIR}/config/test-tiny-llama-shakespeare"
  cp "${ROOT}/config/test-tiny-llama-shakespeare/state.toml" "${OUT_DIR}/config/test-tiny-llama-shakespeare/" || true
  cp "${ROOT}/config/test-tiny-llama-shakespeare/README.md" "${OUT_DIR}/config/test-tiny-llama-shakespeare/" || true
fi

echo "[package] copying python runtime metadata..."
mkdir -p "${OUT_DIR}/python-runtime"
cp "${ROOT}/packaging/python-runtime/pyproject.toml" "${OUT_DIR}/python-runtime/" || true
cp "${ROOT}/packaging/python-runtime/requirements.lock.txt" "${OUT_DIR}/python-runtime/" || true

echo "[package] copying checkpoints (tiny-llama-local)..."
if [[ -d "${ROOT}/checkpoints/tiny-llama-local" ]]; then
  mkdir -p "${OUT_DIR}/checkpoints"
  cp -R "${ROOT}/checkpoints/tiny-llama-local" "${OUT_DIR}/checkpoints/"
  if [[ -d "${ROOT}/checkpoints/tiny-llama-local-tier1" ]]; then
    cp -R "${ROOT}/checkpoints/tiny-llama-local-tier1" "${OUT_DIR}/checkpoints/"
  fi
  if [[ -d "${ROOT}/checkpoints/tiny-llama-local-tier2" ]]; then
    cp -R "${ROOT}/checkpoints/tiny-llama-local-tier2" "${OUT_DIR}/checkpoints/"
  fi
fi

echo "[package] copying local datasets..."
if [[ -f "${ROOT}/data/tinyshakespeare-bin/train.bin" ]]; then
  mkdir -p "${OUT_DIR}/data"
  cp -R "${ROOT}/data/tinyshakespeare-bin" "${OUT_DIR}/data/"
fi

cat > "${OUT_DIR}/README.txt" <<'TXT'
Psyche Centralized (macOS) bundle
================================

Requirements on each machine:
- Python 3.12 (recommended) + PyTorch installed (torch must be importable; MPS works if you installed the Apple build).
- Recommended: `uv` (for reproducible setup). Install via `brew install uv` (or https://astral.sh/uv).

If macOS blocks the binaries ("can't be opened" / "malware" warning):
  bash scripts/unquarantine.sh
Or, equivalently:
  xattr -dr com.apple.quarantine .

Bootstrap (on each machine, in the bundle directory):
  bash scripts/bootstrap-python-runtime.sh
This creates a local `.venv/` using the pinned deps in `python-runtime/requirements.lock.txt`.

Tiny Shakespeare dataset (optional, if `data/` wasn't bundled):
  python3 scripts/prepare_tinyshakespeare_bin_dataset.py --out-dir data/tinyshakespeare-bin

Server IP / connectivity checks (LAN):
  SERVER_IP="$(ipconfig getifaddr en0 || ipconfig getifaddr en1)"
  echo "Server IP: ${SERVER_IP}"
  # on clients:
  nc -vz "${SERVER_IP}" 20000

Server (on one machine):
  RUST_LOG="warn,psyche_centralized_server=info" \
    bash scripts/run-server.sh run --state config/test-tiny-llama/state.toml --server-port 20000 --tui false --logs json --write-log logs/server.jsonl

Client (on each machine):
  RUST_LOG="warn,psyche_client=info,psyche_centralized_client=info" \
    bash scripts/run-client.sh train --run-id test-tiny-llama --server-addr <SERVER_LAN_IP>:20000 --logs json --write-log logs/client.jsonl --device mps --matformer-tier 2 --matformer-load-strategy auto --log-memory-usage --iroh-discovery local --iroh-relay disabled

Notes:
- `scripts/psyche-env.sh` auto-locates torch's `lib/` directory and sets DYLD_LIBRARY_PATH/LD_LIBRARY_PATH for tch-rs.
- `127.0.0.1:20000` only works for clients on the same machine; for other Macs use the server's LAN IP.
- On first run, macOS may prompt you to allow incoming network connections for the server binary; you must allow it.
- If you use AirDrop or download the tarball in a browser, macOS will usually quarantine it. Prefer copying via `scp`/`rsync` to avoid quarantine.
- For WAN use `--iroh-discovery n0 --iroh-relay psyche`.
- `--run-id` must match the `run_id` in the `state.toml` you're using.
- To run with local Tiny Shakespeare data, use `--state config/test-tiny-llama-shakespeare/state.toml` and `--run-id test-tiny-llama-shakespeare`.
- If you want smaller devices to only pull prefix (sliced) checkpoints, run
  `python scripts/export_matformer_tiers.py --src ./checkpoints/<model> --tiers 1 2`
  and point tiered clients at the generated `-tierN` directories (default load strategy is auto).
TXT

mkdir -p "$(dirname "${TAR_PATH}")"
rm -f "${TAR_PATH}"
tar -C "$(dirname "${OUT_DIR}")" -czf "${TAR_PATH}" "$(basename "${OUT_DIR}")"
echo "[package] wrote ${TAR_PATH}"
