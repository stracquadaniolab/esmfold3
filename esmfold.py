#!/usr/bin/env python3
"""ESM3 protein structure prediction script.

Reads amino acid sequences from a FASTA file and predicts their 3D structures
using ESM3, writing one PDB file per sequence, a per-sequence {id}.json sidecar
(log-likelihood, embedding, pLDDT, PAE, residue index, and relaxation energies
when --relax is used), and a run.json with run metadata.

Usage:
    python esmfold.py sequences.fasta -o results/
    python esmfold.py sequences.fasta -o results/ --relax
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import torch
import torch.nn.functional as F
from biotite.sequence.io.fasta import FastaFile
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
    plddt: list[float]
    pae: list[list[float]]
    ptm: float
    residue_index: list[int]
    energy_before_kJ_mol: float
    energy_after_stage1_kJ_mol: float
    energy_after_stage2_kJ_mol: float
    disulfides: list[list[int]]


class RelaxResult(TypedDict):
    energy_before_kJ_mol: float
    energy_after_stage1_kJ_mol: float
    energy_after_stage2_kJ_mol: float
    disulfides: list[list[int]]
    faspr_used: bool


class OpenMMContext(NamedTuple):
    """OpenMM/pdbfixer modules and objects, imported and built once for the run."""

    forcefield: Any   # openmm.app.ForceField (AMBER99SB + OBC implicit solvent)
    platform: Any     # openmm.Platform | None
    mm: Any           # the openmm module
    app: Any          # the openmm.app module
    unit: Any         # the openmm.unit module
    pdbfixer: Any     # the pdbfixer module


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_sequences(fasta_path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file and return a list of (id, sequence) tuples.

    The id is the first whitespace-delimited token of each header (matching the
    common FASTA convention); the rest of the header line is discarded.
    """
    fasta = FastaFile.read(str(fasta_path))
    sequences = [
        (header.split()[0] if header.split() else header, seq)
        for header, seq in fasta.items()
    ]
    if not sequences:
        raise ValueError(f"No sequences found in {fasta_path}")
    return sequences


# The 20 standard amino acids. ESM3's sequence vocabulary also accepts ambiguity
# codes (X/B/U/Z/O), gaps (./-) and the chain-break token (|), and it maps unknown
# characters (e.g. embedded whitespace/newlines from a wrapped FASTA record) to
# <unk>. None of those are handled 1:1 by the structure track, so generation can
# emit a structure whose residue count disagrees with the sequence — surfacing much
# later as a cryptic length assertion inside ESM3's decoder
# (``atom37_positions.shape[0] == len(sequence)``).
CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(sequence: str) -> str:
    """Normalise a raw sequence string before it reaches ESM3.

    Strips *all* whitespace (spaces, tabs, and the embedded newlines that a wrapped
    FASTA record can leak into the sequence body) plus non-printable control bytes
    (NUL and other C0/C1 control characters that a mangled generator or a stray
    C-string terminator can inject), then upper-cases the result. This is the single
    choke point every sequence must pass through, regardless of how it was loaded, so
    malformed input can never desync ESM3's sequence and structure tracks.

    Note this only removes characters that carry no residue meaning; genuine but
    unsupported letters (gaps, ambiguity codes, '*') survive to fail loudly in
    validate_sequence() rather than being silently dropped.
    """
    return "".join(
        c for c in sequence if c.isprintable() and not c.isspace()
    ).upper()


def validate_sequence(sequence: str) -> None:
    """Raise if the sequence contains anything other than the 20 standard residues.

    Run this *after* clean_sequence(). Anything left that is not a standard amino acid
    (gaps, chain breaks, ambiguity codes, stop '*') can still desync ESM3's tracks, so
    fail here with a clear, per-sequence recorded error instead of the opaque assert.
    """
    bad = {(i + 1, c) for i, c in enumerate(sequence) if c not in CANONICAL_RESIDUES}
    if bad:
        preview = ", ".join(f"{c!r}@{i}" for i, c in sorted(bad)[:10])
        more = "" if len(bad) <= 10 else f" (+{len(bad) - 10} more)"
        raise ValueError(
            f"sequence contains {len(bad)} non-standard character(s): {preview}{more}. "
            "ESM3 structure prediction requires only the 20 standard amino acids "
            "(no gaps '-'/'.', chain breaks '|', ambiguity codes X/B/U/Z/O, or '*')."
        )


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
    with torch.no_grad():
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


def structure_confidence(protein: ESMProtein, length: int) -> SequenceFeatures:
    """Extract per-residue confidence metrics from a generated structure.

    Returns the 1-based ``residue_index``, the per-residue ``plddt`` (in [0, 1]),
    the LxL ``pae`` matrix (Å), and the scalar global ``ptm``. ESM3 returns
    ``plddt`` already trimmed to the L residues, but ``pae`` still includes the
    BOS/EOS positions, so it is sliced back to the residue block here.
    """
    result: SequenceFeatures = {"residue_index": list(range(1, length + 1))}
    if protein.plddt is not None:
        result["plddt"] = protein.plddt.squeeze().tolist()
    if protein.pae is not None:
        pae = protein.pae.squeeze(0)  # [L+2, L+2]: drop batch dim
        result["pae"] = pae[1:-1, 1:-1].tolist()  # trim BOS/EOS
    if protein.ptm is not None:
        result["ptm"] = float(protein.ptm)  # scalar global confidence, distinct from plddt/pae
    return result


def predict_structure(
    model: ESM3,
    sequence: str,
    num_steps: int,
    temperature: float,
    schedule: str,
    strategy: str,
) -> ESMProtein:
    """Run ESM3 structure generation for a single sequence."""
    with torch.no_grad():
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


def load_openmm() -> OpenMMContext:
    """Import OpenMM/pdbfixer and build the reusable objects once, at startup.

    The heavy imports, parsing the force-field XML and selecting a compute platform
    are all identical for every structure, so they are done once here rather than on
    each relax_structure() call. Everything is imported lazily (not at module top) so
    the cost is only paid when --relax is used.

    The force field is AMBER99SB with the OBC generalised-Born implicit solvent. Both
    the sidechain pre-optimisation and the production minimisation run in GB solvent:
    minimising ESM3's rebuilt sidechains in vacuum over-packs and buries polar groups,
    collapsing surface charges/salt bridges and wrecking the pKa prediction, whereas GB
    solvent penalises burial so the geometry (and PROPKA input) stays realistic. OBC via
    ``amber99_obc.xml`` uses OpenMM's native (fast) GBSAOBCForce; the newer ff14SB/GBn2
    implicit models fall back to a generic CustomGBForce that is several-fold slower.
    """
    import openmm
    import openmm.app
    import openmm.unit
    import pdbfixer

    forcefield = openmm.app.ForceField("amber99sb.xml", "amber99_obc.xml")

    # Prefer CUDA → OpenCL → CPU
    platform = None
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            platform = openmm.Platform.getPlatformByName(name)
            break
        except openmm.OpenMMException:
            continue

    log.info("OpenMM platform: %s", platform.getName() if platform else "default")
    return OpenMMContext(
        forcefield=forcefield,
        platform=platform,
        mm=openmm,
        app=openmm.app,
        unit=openmm.unit,
        pdbfixer=pdbfixer,
    )


def detect_disulfides(topology, positions_ang, cutoff: float) -> list[tuple]:
    """Infer disulphide-bonded cysteine pairs from Cβ–Cβ distances.

    ESM3 emits backbone only, so PDBFixer rebuilds Cys sidechains in a default
    rotamer — the SG positions are unreliable, but CB is rigidly fixed by the
    backbone. ``positions_ang`` is an atom-indexed sequence of coordinates in Å.
    Candidate pairs (CB–CB ≤ cutoff Å) are matched greedily nearest first, with
    each cysteine bonding at most one partner. Returns the list of
    (residue_i, residue_j) pairs.
    """
    cys_cb = []  # (residue, (x, y, z))
    for res in topology.residues():
        if res.name not in ("CYS", "CYX"):
            continue
        cb = next((a for a in res.atoms() if a.name == "CB"), None)
        if cb is not None:
            cys_cb.append((res, positions_ang[cb.index]))

    candidates = []  # (distance, i, j)
    for i in range(len(cys_cb)):
        for j in range(i + 1, len(cys_cb)):
            (_, a), (_, b) = cys_cb[i], cys_cb[j]
            d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
            if d <= cutoff:
                candidates.append((d, i, j))

    pairs = []
    used: set[int] = set()
    for _, i, j in sorted(candidates):
        if i not in used and j not in used:
            pairs.append((cys_cb[i][0], cys_cb[j][0]))
            used.update((i, j))
    return pairs


def predict_pka(pdb_path: Path, ph: float) -> dict[tuple[str, int], tuple[str, float]]:
    """Run PROPKA on a heavy-atom PDB and return predicted pKa per ionizable group.

    Returns a mapping ``(chain_id, res_num) -> (res_name, pka)``. Raises on any
    PROPKA failure; the caller is expected to fall back to pH-default protonation.
    """
    import propka.run

    # PROPKA logs a full pKa summary at INFO and benign per-atom WARNINGs (it dislikes
    # PDBFixer-reconstructed termini). We only consume the returned values, so silence
    # its loggers to ERROR to keep the pipeline output clean.
    logging.getLogger("propka").setLevel(logging.ERROR)

    mol = propka.run.single(
        str(pdb_path), optargs=["--pH", str(ph)], write_pka=False
    )
    pka: dict[tuple[str, int], tuple[str, float]] = {}
    for group in mol.conformations["AVR"].groups:
        atom = group.atom
        pka[(atom.chain_id, atom.res_num)] = (atom.res_name, group.pka_value)
    return pka


def build_variants(topology, pka, ph: float, disulfide_residues: set) -> list:
    """Map predicted pKa values to per-residue protonation variants for Modeller.

    Returns one entry per ``topology.residues()`` (positional — Modeller maps the
    list by index): a variant name, or None to use Modeller's pH default. Tyr/Arg
    have no neutral/deprotonated template in amber99sb and are left to default.
    """
    variants = []
    for res in topology.residues():
        if res in disulfide_residues:
            variants.append("CYX")
            continue

        try:
            key = (res.chain.id, int(res.id))
        except ValueError:
            variants.append(None)
            continue

        entry = pka.get(key)
        if entry is None:
            variants.append(None)
            continue
        _, value = entry

        if res.name == "ASP":
            variants.append("ASH" if ph < value else "ASP")
        elif res.name == "GLU":
            variants.append("GLH" if ph < value else "GLU")
        elif res.name == "HIS":
            variants.append("HIP" if ph < value else None)  # else let Modeller pick HID/HIE
        elif res.name == "LYS":
            variants.append("LYS" if ph < value else "LYN")
        elif res.name == "CYS":
            variants.append("CYM" if ph > value else "CYS")
        else:
            variants.append(None)  # TYR/ARG: no suitable template
    return variants


def find_faspr(explicit: str | None = None) -> str | None:
    """Locate the FASPR sidechain-packing binary, or return None if unavailable.

    Resolution order: an explicit path (CLI ``--faspr-bin``), then the ``FASPR_BIN``
    environment variable, then ``FASPR`` on ``PATH``. Returning None (rather than
    raising) lets relax_structure() fall back to optimize_sidechains(), so local runs
    without FASPR installed still work.
    """
    for candidate in (explicit, os.environ.get("FASPR_BIN")):
        if candidate:
            resolved = shutil.which(candidate) or (candidate if os.path.isfile(candidate) else None)
            if resolved:
                return resolved
    return shutil.which("FASPR")


def pack_sidechains(faspr_bin: str, in_pdb: Path, out_pdb: Path) -> None:
    """Repack sidechains onto a backbone with FASPR (combinatorial rotamer search).

    FASPR reads a heavy-atom PDB (needs N/CA/C/O) and rewrites every sidechain from a
    backbone-dependent rotamer library, replacing PDBFixer's clashing default rotamers.
    It locates its ``dun2010bbdep.bin`` rotamer library relative to its own executable,
    so the binary and data file only need to be co-located. Raises on any failure so
    the caller can fall back to a local minimisation.
    """
    subprocess.run(
        [faspr_bin, "-i", str(in_pdb), "-o", str(out_pdb)],
        check=True, capture_output=True, text=True, timeout=600,
    )
    if not out_pdb.is_file() or out_pdb.stat().st_size == 0:
        raise RuntimeError("FASPR produced no output")


def optimize_sidechains(
    ctx: OpenMMContext,
    topology,
    positions,
    ph: float,
    disulfide_residues: set,
    max_iterations: int = 0,
    backbone_k: float = 100.0,
) -> Any:
    """Locally minimise sidechains against a stiffly-restrained backbone.

    ESM3 emits backbone only, so PDBFixer rebuilds every sidechain in a default
    rotamer — frequently with steric clashes. Predicting pKa (or running the final
    minimisation) on that raw geometry is unreliable, so this de-clashes the
    sidechains first: it protonates at ``ph`` (only disulphide cysteines are pinned to
    CYX, since their SG–SG bonds are already in the topology and would otherwise clash
    with an added HG), restrains all backbone heavy atoms (N, CA, C, O) with a stiff
    harmonic force so only the sidechains and hydrogens move, and minimises **in OBC
    implicit solvent** — a vacuum minimisation instead over-packs sidechains and buries
    ionizable groups, which drives PROPKA to spurious (often non-titratable) pKa.

    This is a *local* relaxation, not a combinatorial rotamer repack — it removes
    clashes and settles rotamers into the nearest minimum, which is what PROPKA needs,
    but it cannot cross rotamer barriers. Returns the minimised heavy-atom positions
    in the same order as ``topology`` (hydrogens dropped), ready to overwrite the
    fixer positions.
    """
    mm, app, unit = ctx.mm, ctx.app, ctx.unit
    _CONV = 4.184 / 0.01  # kcal/mol/Å² → kJ/mol/nm²

    ss_variants = ["CYX" if res in disulfide_residues else None
                   for res in topology.residues()]
    modeller = app.Modeller(topology, positions)
    modeller.addHydrogens(ctx.forcefield, pH=ph, variants=ss_variants)

    system = ctx.forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    # Stiff harmonic restraint pins the backbone in place so only sidechains relax.
    restraint = mm.CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter("k", backbone_k * _CONV)
    for p in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(p)
    ref_pos = modeller.positions.value_in_unit(unit.nanometer)
    for atom in modeller.topology.atoms():
        if atom.name in ("N", "CA", "C", "O"):
            restraint.addParticle(atom.index, ref_pos[atom.index])
    system.addForce(restraint)

    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds,
    )
    simulation = app.Simulation(modeller.topology, system, integrator, ctx.platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=max_iterations)

    # addHydrogens preserves heavy-atom order and only inserts H, so the non-H atoms
    # of the minimised topology line up 1:1 with the input topology.
    coords = simulation.context.getState(getPositions=True).getPositions().value_in_unit(
        unit.nanometer
    )
    heavy = [
        coords[atom.index]
        for atom in modeller.topology.atoms()
        if atom.element is not None and atom.element.symbol != "H"
    ]
    if len(heavy) != topology.getNumAtoms():
        raise RuntimeError(
            f"sidechain optimisation atom-count mismatch: {len(heavy)} heavy atoms "
            f"vs {topology.getNumAtoms()} in topology"
        )
    return unit.Quantity(heavy, unit.nanometer)


def relax_structure(
    ctx: OpenMMContext,
    input_pdb: Path,
    output_pdb: Path,
    max_iterations: int = 0,
    ph: float = 7.0,
    stage1_k: float = 10.0,
    stage2_k: float = 2.0,
    ss_cutoff: float = 5.0,
    use_propka: bool = True,
    use_faspr: bool = True,
    faspr_bin: str | None = None,
) -> RelaxResult:
    """Two-stage restrained energy minimisation with the AMBER99SB force field in OBC
    implicit solvent.

    Uses pdbfixer to reconstruct missing heavy atoms (O, sidechains) and
    hydrogens before minimisation, since ESM3 only outputs N/CA/C backbone.

    Because ESM3 gives no sidechains, PDBFixer rebuilds them in clashing default
    rotamers. When ``use_faspr`` is set and the FASPR binary is available, sidechains
    are repacked with FASPR (a combinatorial backbone-dependent rotamer search) so
    the pKa prediction and final structure rest on realistic packing. If FASPR is not
    available (or fails), it falls back to optimize_sidechains() — a local implicit-
    solvent minimisation — which de-clashes but cannot cross rotamer barriers.

    Stage 1 applies strong Cα positional restraints (stage1_k kcal/mol/Å²) so
    sidechains and hydrogens can relax without disturbing the backbone. Stage 2
    applies weak restraints (stage2_k kcal/mol/Å²) to allow limited backbone
    movement. max_iterations=0 runs each stage until convergence. The OBC implicit
    solvent keeps surface charges and salt bridges from collapsing inward as they
    would in vacuum.

    Disulphides are inferred from Cβ–Cβ geometry and annotated as SG–SG bonds
    (so Modeller assigns CYX), and protonation states are set from PROPKA-predicted
    pKa values (on the packed sidechains) at the target pH, before hydrogens
    are added.

    All modules, the force field and the platform are taken from a shared
    OpenMMContext imported and built once by load_openmm(); only structure-specific
    objects are constructed here — nothing is imported per call.
    """
    mm, app, unit = ctx.mm, ctx.app, ctx.unit

    # kcal/mol/Å² → kJ/mol/nm²  (1 kcal = 4.184 kJ; 1 Å = 0.1 nm → 1 Å² = 0.01 nm²)
    _CONV = 4.184 / 0.01

    fixer = ctx.pdbfixer.PDBFixer(filename=str(input_pdb))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()  # supplies the O atom FASPR needs, plus placeholder sidechains

    # Repack sidechains with FASPR (combinatorial rotamer search) — the proper fix for
    # ESM3's missing sidechains. Reload the packed structure so everything downstream
    # (disulphides, PROPKA, minimisation) sees the improved geometry. Best-effort: any
    # failure falls back to optimize_sidechains() below.
    faspr_used = False
    resolved_faspr = find_faspr(faspr_bin) if use_faspr else None
    if resolved_faspr:
        try:
            with tempfile.TemporaryDirectory() as td:
                pre, packed = Path(td) / "pre.pdb", Path(td) / "packed.pdb"
                with open(pre, "w") as f:
                    app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
                pack_sidechains(resolved_faspr, pre, packed)
                fixer = ctx.pdbfixer.PDBFixer(filename=str(packed))
                fixer.findMissingResidues()
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
            faspr_used = True
            log.info("  Sidechains repacked with FASPR")
        except Exception as exc:
            log.warning("  FASPR packing failed (%s); falling back to local minimisation", exc)
    elif use_faspr:
        log.info("  FASPR not found; using local sidechain minimisation")

    # Disulphides: bond the SG atoms so Modeller auto-assigns CYX and createSystem
    # models the S–S bond (detected from backbone-fixed CB positions).
    pos_ang = fixer.positions.value_in_unit(unit.angstrom)
    pairs = detect_disulfides(fixer.topology, pos_ang, ss_cutoff)
    for ci, cj in pairs:
        sg_i = next(a for a in ci.atoms() if a.name == "SG")
        sg_j = next(a for a in cj.atoms() if a.name == "SG")
        fixer.topology.addBond(sg_i, sg_j)
    disulfide_residues = {r for pair in pairs for r in pair}
    disulfides = [[int(ci.id), int(cj.id)] for ci, cj in pairs]
    log.info("  Disulfides detected: %d %s", len(pairs), disulfides)

    # Protonation states from PROPKA-predicted pKa at the target pH (best-effort).
    # PROPKA is only meaningful on realistic sidechain geometry. FASPR already provides
    # that; without it, de-clash locally first (optimize_sidechains also seeds the final
    # minimisation below).
    variants = None
    if use_propka:
        try:
            if not faspr_used:
                fixer.positions = optimize_sidechains(
                    ctx, fixer.topology, fixer.positions, ph,
                    disulfide_residues, max_iterations,
                )
                log.info("  Sidechains optimised before pKa prediction")
            with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
                with open(tmp.name, "w") as f:
                    app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
                pka = predict_pka(Path(tmp.name), ph)
            variants = build_variants(fixer.topology, pka, ph, disulfide_residues)
        except Exception as exc:
            log.warning(
                "  Sidechain optimisation / PROPKA failed (%s); "
                "falling back to pH-default protonation", exc
            )
            variants = None

    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(ctx.forcefield, pH=ph, variants=variants)

    system = ctx.forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    # Harmonic Cα restraint: E = k * (dr)²  with k as a mutable global parameter.
    restraint = mm.CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter("k", 0.0)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    ref_pos = modeller.positions.value_in_unit(unit.nanometer)
    for atom in modeller.topology.atoms():
        if atom.name == "CA":
            x0, y0, z0 = ref_pos[atom.index]
            restraint.addParticle(atom.index, [x0, y0, z0])

    system.addForce(restraint)

    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        0.002 * unit.picoseconds,
    )

    simulation = app.Simulation(modeller.topology, system, integrator, ctx.platform)
    simulation.context.setPositions(modeller.positions)

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
        app.PDBFile.writeFile(
            simulation.topology,
            state_final.getPositions(),
            f,
            keepIds=True,
        )

    return {
        "energy_before_kJ_mol": energy_before,
        "energy_after_stage1_kJ_mol": energy_after_stage1,
        "energy_after_stage2_kJ_mol": energy_after_stage2,
        "disulfides": disulfides,
        "faspr_used": faspr_used,
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
        "--relax-ss-cutoff",
        type=float,
        default=5.0,
        metavar="D",
        help="Cβ–Cβ distance cutoff in Å for disulphide detection (default: 5.0).",
    )
    parser.add_argument(
        "--no-propka",
        action="store_true",
        help="Disable PROPKA pKa prediction; use pH-default protonation instead.",
    )
    parser.add_argument(
        "--no-faspr",
        action="store_true",
        help="Disable FASPR sidechain repacking; use local GB-solvent minimisation instead.",
    )
    parser.add_argument(
        "--faspr-bin",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the FASPR binary (default: $FASPR_BIN, else 'FASPR' on PATH).",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Include the mean-pooled ESM3 embedding in the sidecar JSON (off by default).",
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
    failed: list[str] = []

    # Phase 1: ESM3 inference (sequential — single GPU model). Each structure and
    # its feature sidecar are written to disk immediately so nothing per-sequence is
    # held in memory between phases; Phase 2 reloads only what it needs, one at a time.
    to_relax: list[tuple[str, str]] = []
    for seq_id, raw_sequence in sequences:
        safe_id = sanitize_id(seq_id)
        out_path = args.output_dir / f"{safe_id}.pdb"
        try:
            sequence = clean_sequence(raw_sequence)
            if len(sequence) != len(raw_sequence):
                log.warning(
                    "  [%s] stripped %d whitespace/control char(s) from sequence",
                    seq_id, len(raw_sequence) - len(sequence),
                )
            validate_sequence(sequence)
            log.info("[%s] length=%d", seq_id, len(sequence))

            log.info("  Computing log-likelihood and embedding...")
            seq_features = compute_features(model, sequence)

            log.info("  Running structure prediction...")
            protein = predict_structure(
                model, sequence, args.num_steps, args.temperature,
                args.schedule, args.strategy,
            )
            seq_features = {**seq_features, **structure_confidence(protein, len(sequence))}
            if not args.embedding:
                seq_features.pop("embedding", None)
            protein.to_pdb(str(out_path))
            log.info("  -> %s", out_path)

            write_json(args.output_dir / f"{safe_id}.json", {"id": seq_id, **seq_features})
            if args.relax:
                to_relax.append((seq_id, safe_id))
        except Exception as exc:
            log.error("  [%s] FAILED: %s", seq_id, exc, exc_info=True)
            failed.append(seq_id)
        finally:
            # Drop per-sequence tensors/features and reclaim GPU memory before the next one.
            protein = seq_features = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Phase 2: OpenMM relaxation (sequential). Reload each structure and its sidecar
    # from disk, relax, then update the sidecar in place with the relaxation energies.
    if to_relax:
        log.info("Running OpenMM relaxation for %d structure(s)...", len(to_relax))
        openmm_ctx = load_openmm()
        for seq_id, safe_id in to_relax:
            out_path = args.output_dir / f"{safe_id}.pdb"
            relaxed_path = args.output_dir / f"{safe_id}_relaxed.pdb"
            sidecar_path = args.output_dir / f"{safe_id}.json"
            try:
                relax_info = relax_structure(
                    openmm_ctx, out_path, relaxed_path,
                    args.relax_max_iter, args.relax_ph,
                    args.relax_stage1_k, args.relax_stage2_k,
                    args.relax_ss_cutoff, not args.no_propka,
                    not args.no_faspr, args.faspr_bin,
                )
                log.info(
                    "[%s] relaxed -> %s (%.1f → %.1f kJ/mol)",
                    seq_id, relaxed_path,
                    relax_info["energy_before_kJ_mol"],
                    relax_info["energy_after_stage2_kJ_mol"],
                )
                with sidecar_path.open() as f:
                    sidecar = json.load(f)
                sidecar.update(relax_info)
                write_json(sidecar_path, sidecar)
            except Exception as exc:
                log.error("[%s] FAILED relaxation: %s", seq_id, exc, exc_info=True)
                failed.append(seq_id)

    end_time = datetime.now(timezone.utc)

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
            "relax_ss_cutoff": args.relax_ss_cutoff if args.relax else None,
            "propka": (not args.no_propka) if args.relax else None,
            "faspr": (not args.no_faspr) if args.relax else None,
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
