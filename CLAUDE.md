# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run structure prediction
python esmfold.py datasets/2MLT.fasta -o results/

# Run with OpenMM relaxation
python esmfold.py datasets/2MLT.fasta -o results/ --relax

# Set HuggingFace credentials before first run (downloads ~1.4 GB ESM3 weights)
source vars.sh

# Build Docker image
docker build -t esmfold .

# Run via Docker (requires NVIDIA runtime)
docker run --gpus all -v /path/to/data:/data esmfold /data/sequences.fasta -o /data/results/
```

There are no tests and no linter configuration.

## Architecture

The entire application lives in a single file: `esmfold.py`. It is a CLI script that processes FASTA sequences through three sequential phases per sequence:

1. **`compute_features()`** — runs ESM3 forward pass to compute per-residue log-likelihoods and mean-pooled embeddings.
2. **`predict_structure()`** — runs ESM3 iterative structure generation (diffusion-style, configurable steps/temperature/schedule/strategy).
3. **`relax_structure()`** (optional, `--relax`) — two-stage OpenMM energy minimisation using the AMBER99SB force field in vacuum (no implicit solvent). Stage 1 uses strong Cα restraints to settle sidechains/hydrogens; stage 2 uses weak restraints to allow limited backbone movement.

**Key design decisions:**
- OpenMM is imported lazily inside `relax_structure()` — it is only loaded when `--relax` is passed, avoiding startup cost otherwise.
- OpenMM platform is auto-selected at runtime: CUDA → OpenCL → CPU.
- ESM3 runs on `cuda` if `torch.cuda.is_available()`, else `cpu`. These are independent of OpenMM's GPU selection.
- Sequences that fail are caught, logged, and recorded in `run.json`; the script exits with code `1` if any failed. Successful sequences are always written even if others fail.

**Outputs per run:**
- `{id}.pdb` — raw ESM3 structure (backbone only: N/CA/C)
- `{id}_relaxed.pdb` — all-atom structure after OpenMM relaxation (only with `--relax`)
- `{id}.json` — a sidecar JSON file for each relaxed pdb, with the ESM3 log-likelihood, embedding, pae, plddt, residue_index, energy values per each sequence.
- `run.json` — run metadata including GPU name, parameters, and failed sequence IDs
- if the output directory does not exists, create it. 

## Environment

- Python 3.12 only (`requires-python = ">=3.12,<3.13"`)
- Package manager: `uv`
- Docker base image: `nvidia/cuda:13.1.0-runtime-ubuntu24.04`
- `openmm[cuda13]` extra is used to get the CUDA 13-compatible OpenMM build
- HuggingFace token is required on first run to download ESM3 weights; `vars.sh` sets `HF_TOKEN` and `HF_HOME`
