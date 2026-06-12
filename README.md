# esmfold

[![Docker Image](https://img.shields.io/github/v/release/stracquadaniolab/esmfold?logo=docker&label=ghcr.io)](https://github.com/stracquadaniolab/esmfold/pkgs/container/esmfold)

Predict protein 3D structures from amino acid sequences using
[ESM3](https://github.com/evolutionaryscale/esm). For each sequence the script
writes a PDB file and a per-sequence `{id}.json` sidecar, plus a run-level
summary:

- `{id}.json` — per-sequence log-likelihood, mean embedding, per-residue pLDDT, PAE matrix, residue index, and (when relaxation is enabled) energy values
- `run.json` — start/end time, GPU used, run parameters, and any failed sequences

## Requirements

- Python 3.12
- PyTorch (GPU recommended)
- An [HuggingFace Account](https://huggingface.co) and API
  token to download the ESM3 weights on first run

## Installation

```bash
pip install esm pyfastx httpx
```

Then make the script executable:

```bash
chmod +x esmfold.py
```

## Usage

```
python esmfold.py <fasta_file> [options]
```

### Arguments

| Argument | Description |
|---|---|
| `fasta_file` | Input FASTA file (required) |
| `-o`, `--output-dir DIR` | Directory for output files (default: `.`) |
| `-n`, `--num-steps N` | Structure generation steps (default: `1`) |
| `-t`, `--temperature T` | Sampling temperature (default: `0.0`) |
| `-s`, `--schedule SCHEDULE` | Noise schedule, e.g. `cosine`, `linear` (default: `cosine`) |
| `--strategy STRATEGY` | Decoding strategy, e.g. `entropy`, `random` (default: `entropy`) |
| `--relax` | Run OpenMM energy minimisation after structure prediction |
| `--relax-max-iter N` | Max minimisation iterations per stage (default: `0` = until convergence) |
| `--relax-ph PH` | pH for hydrogen placement (default: `7.0`) |
| `--relax-stage1-k K` | Stage 1 Cα restraint force constant in kcal/mol/Å² (default: `10.0`) |
| `--relax-stage2-k K` | Stage 2 Cα restraint force constant in kcal/mol/Å² (default: `2.0`) |

### Examples

Basic structure prediction:

```bash
python esmfold.py sequences.fasta -o results/
```

With energy minimisation:

```bash
python esmfold.py sequences.fasta -o results/ --relax
```

This produces:

```
results/
├── SEQ1.pdb
├── SEQ1_relaxed.pdb   # only with --relax
├── SEQ1.json
├── SEQ2.pdb
├── SEQ2_relaxed.pdb   # only with --relax
├── SEQ2.json
└── run.json
```

### `{id}.json` format

One sidecar file is written per sequence:

```json
{
  "id": "SEQ1",
  "loglik": -42.3,
  "embedding": [0.12, -0.05, "..."],
  "plddt": [0.91, 0.88, "..."],
  "pae": [[0.0, 1.2, "..."], ["..."]],
  "residue_index": [1, 2, "..."],
  "energy_before_kJ_mol": -18234.1,
  "energy_after_stage1_kJ_mol": -19102.7,
  "energy_after_stage2_kJ_mol": -19308.4
}
```

`plddt` is per-residue, `pae` is the LxL predicted-aligned-error matrix, and
`residue_index` is 1-based. The three `energy_*` fields are only present when
`--relax` is used.

### `run.json` format

```json
{
  "start_time": "2026-04-16T10:23:01+00:00",
  "end_time": "2026-04-16T10:25:44+00:00",
  "gpu": "NVIDIA A100 80GB PCIe",
  "num_steps": 1,
  "temperature": 0.0,
  "schedule": "cosine",
  "strategy": "entropy",
  "relax": true,
  "relax_max_iter": 0,
  "relax_ph": 7.0,
  "relax_stage1_k": 10.0,
  "relax_stage2_k": 2.0,
  "failed_sequences": []
}
```

## Docker

Build the image:

```bash
docker build -t esmfold .
```

Run on a FASTA file, mounting a local directory for input and output:

```bash
docker run --gpus all \
  -v /path/to/data:/data \
  esmfold /data/sequences.fasta -o /data/results/
```

## Notes

- PDB files are named after the sequence identifier in the FASTA header.
  Characters that are unsafe in filenames are replaced with `_`.
- On first run ESM3 will download model weights (~1.4 GB). Set the
  `ESM_CACHE_DIR` environment variable to control where they are stored.
- Sequences that fail (e.g. due to unsupported characters) are logged and
  recorded in `run.json`. The script exits with code `1` if any sequence
  failed.
