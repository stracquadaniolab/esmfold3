#!/usr/bin/env python3
"""ESM3 protein structure prediction script.

Reads amino acid sequences from a FASTA file and predicts their 3D structures
using ESM3, writing one PDB file per sequence, a features.json with per-sequence
log-likelihoods and embeddings, and a run.json with run metadata.

Usage:
    python esmfold.py sequences.fasta -o results/
    python esmfold.py sequences.fasta -o results/ --relax
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import pyfastx
import torch
import torch.nn.functional as F
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESMProtein,
    ESMProteinError,
    GenerationConfig,
    LogitsConfig,
)
from esm.utils.constants.models import ESM3_OPEN_SMALL

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class SequenceFeatures(TypedDict, total=False):
    id: str
    loglik: float
    embedding: list[float]
    energy_before_kJ_mol: float
    energy_after_stage1_kJ_mol: float
    energy_after_stage2_kJ_mol: float


class RelaxResult(TypedDict):
    energy_before_kJ_mol: float
    energy_after_stage1_kJ_mol: float
    energy_after_stage2_kJ_mol: float


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_sequences(fasta_path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file and return a list of (id, sequence) tuples."""
    sequences = [(seq.name, seq.seq) for seq in pyfastx.Fasta(str(fasta_path))]
    if not sequences:
        raise ValueError(f"No sequences found in {fasta_path}")
    return sequences


def sanitize_id(seq_id: str) -> str:
    """Replace characters unsafe in filenames."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in seq_id)


def write_json(path: Path, data: object, **kwargs) -> None:
    """Serialise data to JSON and log the output path."""
    with path.open("w") as f:
        json.dump(data, f, **kwargs)
    log.info("Written: %s", path)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model() -> ESM3:
    """Load ESM3 from pretrained weights, using GPU if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)
    model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device(device))
    log.info("Model loaded")
    return model


# ---------------------------------------------------------------------------
# Per-sequence computations
# ---------------------------------------------------------------------------

def compute_features(model: ESM3, sequence: str) -> SequenceFeatures:
    """Return log-likelihood and mean embedding for a sequence.

    The log-likelihood is the sum of per-residue log-probabilities under the
    sequence track. The embedding is the mean-pooled per-position hidden state.
    Both exclude the BOS and EOS special tokens.
    """
    protein_tensor = model.encode(ESMProtein(sequence=sequence))
    output = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True),
    )

    seq_tokens = protein_tensor.sequence  # [L]
    seq_logits = output.logits.sequence.squeeze(0)  # [L, vocab]
    embeddings = output.embeddings.squeeze(0)  # [L, d_model]

    # Exclude BOS (position 0) and EOS (position -1)
    residue_tokens = seq_tokens[1:-1]
    log_probs = F.log_softmax(seq_logits[1:-1], dim=-1)
    loglik = log_probs[
        torch.arange(len(residue_tokens)), residue_tokens
    ].sum().item()
    embedding = embeddings[1:-1].mean(dim=0).tolist()

    return {"loglik": loglik, "embedding": embedding}


def predict_structure(
    model: ESM3,
    sequence: str,
    num_steps: int,
    temperature: float,
    schedule: str,
    strategy: str,
) -> ESMProtein:
    """Run ESM3 structure generation for a single sequence."""
    result = model.generate(
        ESMProtein(sequence=sequence),
        GenerationConfig(
            track="structure",
            num_steps=num_steps,
            temperature=temperature,
            schedule=schedule,
            strategy=strategy,
        ),
    )
    if isinstance(result, ESMProteinError):
        raise RuntimeError(
            f"ESM3 generation failed (code {result.error_code}): {result.error_msg}"
        )
    return result


def relax_structure(
    input_pdb: Path,
    output_pdb: Path,
    max_iterations: int = 0,
    ph: float = 7.0,
    stage1_k: float = 10.0,
    stage2_k: float = 2.0,
) -> RelaxResult:
    """Two-stage restrained energy minimisation with AMBER14 + GBn2 implicit solvent.

    Uses pdbfixer to reconstruct missing heavy atoms (O, sidechains) and
    hydrogens before minimisation, since ESM3 only outputs N/CA/C backbone.

    Stage 1 applies strong Cα positional restraints (stage1_k kcal/mol/Å²) so
    sidechains and hydrogens can relax without disturbing the backbone. Stage 2
    applies weak restraints (stage2_k kcal/mol/Å²) to allow limited backbone
    movement. max_iterations=0 runs each stage until convergence.
    """
    import openmm
    from openmm import CustomExternalForce, LangevinMiddleIntegrator, Platform, unit
    from openmm.app import ForceField, NoCutoff, HBonds, PDBFile, Simulation
    from pdbfixer import PDBFixer

    # kcal/mol/Å² → kJ/mol/nm²  (1 kcal = 4.184 kJ; 1 Å = 0.1 nm → 1 Å² = 0.01 nm²)
    _CONV = 4.184 / 0.01

    fixer = PDBFixer(filename=str(input_pdb))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    forcefield = ForceField("amber99sb.xml")
    system = forcefield.createSystem(
        fixer.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
    )

    # Harmonic Cα restraint: E = k * (dr)²  with k as a mutable global parameter.
    restraint = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter("k", 0.0)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    ref_pos = fixer.positions.value_in_unit(unit.nanometer)
    for atom in fixer.topology.atoms():
        if atom.name == "CA":
            x0, y0, z0 = ref_pos[atom.index]
            restraint.addParticle(atom.index, [x0, y0, z0])

    system.addForce(restraint)

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        0.002 * unit.picoseconds,
    )

    # Prefer CUDA → OpenCL → CPU
    platform = None
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            platform = Platform.getPlatformByName(name)
            break
        except openmm.OpenMMException:
            continue

    simulation = Simulation(fixer.topology, system, integrator, platform)
    log.info("  OpenMM platform: %s", simulation.context.getPlatform().getName())
    simulation.context.setPositions(fixer.positions)

    def _energy() -> float:
        return simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )

    energy_before = _energy()

    # Stage 1 — strong Cα restraints
    simulation.context.setParameter("k", stage1_k * _CONV)
    simulation.minimizeEnergy(maxIterations=max_iterations)
    energy_after_stage1 = _energy()
    log.info(
        "  Stage 1 (k=%.1f kcal/mol/Å²): %.1f → %.1f kJ/mol",
        stage1_k, energy_before, energy_after_stage1,
    )

    # Stage 2 — weak Cα restraints
    simulation.context.setParameter("k", stage2_k * _CONV)
    simulation.minimizeEnergy(maxIterations=max_iterations)
    state_final = simulation.context.getState(getEnergy=True, getPositions=True)
    energy_after_stage2 = state_final.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    log.info(
        "  Stage 2 (k=%.1f kcal/mol/Å²): %.1f → %.1f kJ/mol",
        stage2_k, energy_after_stage1, energy_after_stage2,
    )

    with output_pdb.open("w") as f:
        PDBFile.writeFile(
            simulation.topology,
            state_final.getPositions(),
            f,
            keepIds=True,
        )

    return {
        "energy_before_kJ_mol": energy_before,
        "energy_after_stage1_kJ_mol": energy_after_stage1,
        "energy_after_stage2_kJ_mol": energy_after_stage2,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict protein structures from a FASTA file using ESM3."
    )
    parser.add_argument("fasta_file", type=Path, help="Input FASTA file.")
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="Directory for output files (default: current directory).",
    )
    parser.add_argument(
        "--num-steps", "-n",
        type=int,
        default=1,
        metavar="N",
        help="Structure generation steps (default: 1).",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        metavar="T",
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--schedule", "-s",
        type=str,
        default="cosine",
        metavar="SCHEDULE",
        help="Noise schedule for structure generation, e.g. cosine, linear (default: cosine).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="entropy",
        metavar="STRATEGY",
        help="Decoding strategy, e.g. entropy, random (default: entropy).",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="Run OpenMM energy minimisation after structure prediction.",
    )
    parser.add_argument(
        "--relax-max-iter",
        type=int,
        default=0,
        metavar="N",
        help="Max iterations for energy minimisation (default: 0 = until convergence).",
    )
    parser.add_argument(
        "--relax-ph",
        type=float,
        default=7.0,
        metavar="PH",
        help="pH for hydrogen placement during relaxation (default: 7.0).",
    )
    parser.add_argument(
        "--relax-stage1-k",
        type=float,
        default=10.0,
        metavar="K",
        help="Stage 1 Cα restraint force constant in kcal/mol/Å² (default: 10.0).",
    )
    parser.add_argument(
        "--relax-stage2-k",
        type=float,
        default=2.0,
        metavar="K",
        help="Stage 2 Cα restraint force constant in kcal/mol/Å² (default: 2.0).",
    )
    parser.add_argument(
        "--relax-workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers for OpenMM relaxation (default: 1).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args(argv)

    if not args.fasta_file.is_file():
        log.error("FASTA file not found: %s", args.fasta_file)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sequences = load_sequences(args.fasta_file)
    except Exception as exc:
        log.error("Error reading FASTA file: %s", exc)
        sys.exit(1)

    log.info("Found %d sequence(s). Loading model...", len(sequences))
    try:
        model = load_model()
    except Exception as exc:
        log.error("Error loading model: %s", exc, exc_info=True)
        sys.exit(1)

    log.info(
        "Starting predictions (steps=%d, temperature=%.2f, schedule=%s, strategy=%s%s)...",
        args.num_steps,
        args.temperature,
        args.schedule,
        args.strategy,
        ", relax=on" if args.relax else "",
    )

    start_time = datetime.now(timezone.utc)
    features: list[SequenceFeatures] = []
    failed: list[str] = []

    # Phase 1: ESM3 inference (sequential — single GPU model).
    pending_relax: list[tuple[str, Path, SequenceFeatures]] = []
    for seq_id, sequence in sequences:
        log.info("[%s] length=%d", seq_id, len(sequence))
        safe_id = sanitize_id(seq_id)
        out_path = args.output_dir / f"{safe_id}.pdb"
        try:
            log.info("  Computing log-likelihood and embedding...")
            seq_features = compute_features(model, sequence)

            log.info("  Running structure prediction...")
            protein = predict_structure(
                model, sequence, args.num_steps, args.temperature,
                args.schedule, args.strategy,
            )
            protein.to_pdb(str(out_path))
            log.info("  -> %s", out_path)

            if args.relax:
                pending_relax.append((seq_id, out_path, seq_features))
            else:
                features.append({"id": seq_id, **seq_features})
        except Exception as exc:
            log.error("  [%s] FAILED: %s", seq_id, exc, exc_info=True)
            failed.append(seq_id)

    # Phase 2: OpenMM relaxation (parallel across workers).
    if pending_relax:
        log.info(
            "Running OpenMM relaxation for %d structure(s) with %d worker(s)...",
            len(pending_relax),
            args.relax_workers,
        )
        future_list = []
        with ProcessPoolExecutor(max_workers=args.relax_workers) as executor:
            for seq_id, out_path, seq_features in pending_relax:
                safe_id = sanitize_id(seq_id)
                relaxed_path = args.output_dir / f"{safe_id}_relaxed.pdb"
                future = executor.submit(
                    relax_structure,
                    out_path, relaxed_path,
                    args.relax_max_iter, args.relax_ph,
                    args.relax_stage1_k, args.relax_stage2_k,
                )
                future_list.append((seq_id, seq_features, relaxed_path, future))

        for seq_id, seq_features, relaxed_path, future in future_list:
            try:
                relax_info = future.result()
                log.info("[%s] relaxed -> %s", seq_id, relaxed_path)
                features.append({"id": seq_id, **seq_features, **relax_info})
            except Exception as exc:
                log.error("[%s] FAILED relaxation: %s", seq_id, exc, exc_info=True)
                failed.append(seq_id)

    end_time = datetime.now(timezone.utc)

    write_json(args.output_dir / "features.json", features)

    gpu_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else None
    )
    write_json(
        args.output_dir / "run.json",
        {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "gpu": gpu_name,
            "fasta_file": str(args.fasta_file),
            "num_steps": args.num_steps,
            "temperature": args.temperature,
            "schedule": args.schedule,
            "strategy": args.strategy,
            "relax": args.relax,
            "relax_max_iter": args.relax_max_iter if args.relax else None,
            "relax_ph": args.relax_ph if args.relax else None,
            "relax_stage1_k": args.relax_stage1_k if args.relax else None,
            "relax_stage2_k": args.relax_stage2_k if args.relax else None,
            "relax_workers": args.relax_workers if args.relax else None,
            "failed_sequences": failed,
        },
        indent=2,
    )

    success = len(sequences) - len(failed)
    log.info(
        "Done. %d/%d structure(s) written to %s",
        success,
        len(sequences),
        args.output_dir,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
