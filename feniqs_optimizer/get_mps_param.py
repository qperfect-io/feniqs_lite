#!/usr/bin/env python3
"""
MIMIQ-MPS Hyperparameter Recommender (Train + Predict)
=====================================================
This tool is lightweight ML model that predicts suitable MPS execution parameters directly from the structure of an input quantum circuit in OpenQASM 2.0 format.
It contains two modes: train and predict.

Input 
--------------------
1. For train mode: CSV
Expected header: qasm_path,bond_dim,ent_dim,trunc_eps,meth,fuse,perm

Where:
- bond_dim   : integer from 2 to 4096
- ent_dim    : integer from 2 to 4096
- trunc_eps  : float   from 1e-1 to 1e-12
- meth       : categorical string label (vmpoa, dmpo)
- fuse       : boolean-like (T/F, True/False, 1/0)
- perm       : boolean-like (T/F, True/False, 1/0)

2. For predict mode: input qasm file

Output
------------------
A JSON object with: bond_dim, ent_dim, trunc_eps, meth, fuse, perm

Post-processing guarantees:
- bond_dim and ent_dim are snapped to powers of two within configured bounds
- trunc_eps is snapped onto the discrete grid: 1e-1, 1e-2, ..., 1e-12

Example of usage:
1. Train: python get_mps_param.py train --labels train/labels.csv --outdir mimiq_model

Where:
- train/labels.csv - training data in the form of a table (data is obtained from EO process)
- mimiq_model - directory where models are saved

2. Predict: python get_mps_param.py predict --modeldir mimiq_model --qasm  twolocalrandom_ucx_100.qasm 

where:
- mimiq_model - directory where models are saved
- twolocalrandom_ucx_100.qasm  - inout qasm file
"""

from __future__ import annotations

import os
import re
import json
import math
import hashlib
import argparse
from dataclasses import dataclass, asdict
from collections import Counter
from typing import Dict, List, Tuple, Iterable, Any

import numpy as np

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
except ImportError as e:
    raise SystemExit("Missing deps. Install with: pip install scikit-learn joblib numpy") from e



# Constraints 
MIN_DIM = 4
MAX_DIM = 4096

TRUNC_EPS_GRID: List[float] = [10.0 ** (-k) for k in range(1, 13)]  # 1e-1 ... 1e-12 - all suported values of trunc_eps for MIMIQ-MPS
TRUNC_EPS_TO_ID = {v: i for i, v in enumerate(TRUNC_EPS_GRID)}      # convert list to vacabulary - for convert param trunc_eps to class

# Auxillary functions
def snap_to_power_of_two(x: float, min_val: int = MIN_DIM, max_val: int = MAX_DIM) -> int:
    """
    Convert a continuous ML prediction into a valid MPS dimension.

    The MPS backend only supports bond/entanglement dimensions that are powers of two
    (4, 8, 16, 32, 64, ...). This function takes any real-valued prediction and
    maps it to the nearest valid power-of-two within allowed bounds.
    """
    # If the model produced NaN or infinity, fall back to the smallest safe value
    if not math.isfinite(x):
        return min_val
    # Round to the nearest integer
    xi = int(round(x))
    # Clamp the value to the allowed physical range
    xi = max(min_val, min(max_val, xi))
    # Convert the integer to the nearest power of two
    # Example: 130 -> log2(130)=7.02 -> round(7) -> 2^7 = 128
    return 1 << int(round(math.log2(xi)))


def snap_trunc_eps(eps: float) -> float:
    """
    Convert a continuous ML prediction into a valid MPS truncation threshold.

    MIMIQ-MPS supports a fixed grid:
        1e-1, 1e-2, ..., 1e-12

    The ML model can predict a real-valued epsilon (e.g. 3.7e-6).
    This function projects that value onto the closest valid grid point.
    """
    # If the model produced NaN, infinity, or a non-positive value, return a safe default
    if not math.isfinite(eps) or eps <= 0.0:
        return 1e-5
    # Convert epsilon to log10 scale to ensure that "closeness" is measured in orders of magnitude.
    # Example: 1e-6 and 5e-6 are very close in log-space.
    loge = math.log10(eps)
    # Find the value whose log10 is closest to the predicted log10(eps)
    # Example:
    #   eps = 3e-6  -> log10 = −5.52
    #   closest value is 1e-6 (log10 = −6)
    return min(TRUNC_EPS_GRID, key=lambda v: abs(math.log10(v) - loge))


def eps_to_class(eps: float) -> int:
    """
    Convert a truncation into a discrete class index.

    The ML model cannot regress directly on discrete values like
    [1e-1, 1e-2, ..., 1e-12], so we represent each allowed value
    as a class ID:
        1e-1  -> 0
        1e-2  -> 1
        ...
        1e-12 -> 11

    This function snaps a raw value to the nearest valid grid value
    and then returns the corresponding class index.
    """
    # Snap eps to the nearest allowed grid point
    snapped = snap_trunc_eps(float(eps))
    # Convert that grid value to its integer class label
    return TRUNC_EPS_TO_ID[snapped]


def safe_div(a: float, b: float) -> float:
    """
    Divide two numbers, but return 0 if the denominator is zero.

    This avoids NaNs and infinities in feature vectors when,
    for example, a circuit has zero CX gates.
    """
    return a / b if b else 0.0


def lg(x: float) -> float:
    """
    Logarithmic feature scaling with numerical safety.

    This computes log(1 + x), which:
    - Handles x = 0 safely
    - Compresses very large values
    - Preserves ordering for ML models

    Used to stabilize gate counts and graph statistics so that
    extremely large circuits do not dominate the ML model.
    """
    return math.log1p(max(0.0, x))


def parse_bool(value: str) -> int:
    """
    Convert CSV-style boolean strings into numerical labels.

    The training CSV may contain booleans in different formats:
        "T", "F", "True", "False", "1", "0", "yes", "no"

    ML models require numbers, not strings, so this function normalizes all valid representations into:
        True  -> 1
        False 6> 0
    """
    # Normalize the input by removing whitespace and lowercasing
    s = value.strip().lower()
    # Accepted representations for "true"
    if s in ("1", "t", "true", "yes", "y"):
        return 1
    # Accepted representations for "false"
    if s in ("0", "f", "false", "no", "n"):
        return 0
    # Reject anything ambiguous or invalid
    raise ValueError(f"Invalid boolean '{value}'.")


# QASM parsing and features
"""
 Match OpenQASM quantum register declarations like:
   qreg q[32];
   qreg b[4];
 Captures:
   group(1) -> register name ("q", "b")
   group(2) -> register size ("32", "4")
"""
QREG_RE = re.compile(r"^\s*qreg\s+([a-zA-Z_]\w*)\[(\d+)\]\s*;\s*$", re.IGNORECASE)

"""
 Match any qubit reference inside a gate line:
   q[3], data[12]
 Captures:
   group(1) -> register name
   group(2) -> qubit index within that register
"""
QOP_RE = re.compile(r"([a-zA-Z_]\w*)\[(\d+)\]")

"""
 Match the gate name at the beginning of a QASM instruction:
   cx q[0], q[1];
   rz(0.5) q[3];
 Captures:
   group(1) -> gate name ("cx", "rz", etc.)
"""
GATE_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\b", re.IGNORECASE)

# QASM keywords that do NOT represent unitary quantum evolution.
NON_UNITARY_PREFIX = ("measure", "barrier", "creg", "if(", "reset", "opaque")


def is_comment_or_empty(line: str) -> bool:
    """
    Returns True if a QASM line is empty or contains only a comment.

    - `strip()` removes whitespace from both ends.
    - If the result is an empty string, the line is empty.
    - If it starts with "//", it is a QASM comment.

    Such lines carry no semantic information and should be ignored
    when parsing circuits.
    """
    s = line.strip()
    return (not s) or s.startswith("//")


def is_non_unitary_line(line: str) -> bool:
    """
    Returns True if a QASM line represents a non-unitary operation.

    These are operations that do NOT correspond to quantum gates  for example:
      - measurements
      - barriers
      - classical registers
      - conditional execution
      - resets

    These lines are ignored when building circuit features for now
    """
    s = line.strip().lower()
    return any(s.startswith(p) for p in NON_UNITARY_PREFIX)


def build_qreg_layout(lines: Iterable[str]) -> Tuple[Dict[str, int], Dict[Tuple[str, int], int]]:
    """
    Build a mapping from (qreg_name, local_index) -> global_qubit_index.

    QASM allows multiple quantum registers, for example:
        qreg q[5];
        qreg anc[3];

    Physically this means we have 8 qubits:
        q[0..4], anc[0..2]

    This function concatenates all qregs in their declaration order to create
    a single global numbering:
        q[0]   -> 0
        q[1]   -> 1
        ...
        anc[2] -> 8

    This is essential to correctly  compute distances like |qubit_i - qubit_j|.
    """
    # Maps register name -> its size
    reg_sizes: Dict[str, int] = {}
    # Keeps registers in the order they were declared in the QASM
    reg_order: List[str] = []
    # Parse all qreg declarations
    for ln in lines:
        m = QREG_RE.match(ln)
        if not m:
            continue
        # Example: "qreg q[32];" -> name="q", size=32
        name = m.group(1)
        size = int(m.group(2))
        reg_sizes[name] = size
        reg_order.append(name)
    # Compute where each register starts in the global index space
    offsets: Dict[str, int] = {}
    cur = 0
    for r in reg_order:
        offsets[r] = cur
        cur += reg_sizes[r]
    # Build the final mapping:  (register, local index) -> global qubit index
    gmap: Dict[Tuple[str, int], int] = {}
    for r in reg_order:
        off = offsets[r]
        for i in range(reg_sizes[r]):
            gmap[(r, i)] = off + i

    return reg_sizes, gmap


def extract_features(qasm: str) -> Dict[str, float]:
    """
    Extract numerical features from an OpenQASM circuit.

    The goal of this function is to convert a quantum circuit into a fixed-size
    numerical vector that describes how hard it is for an MPS simulator to
    simulate the circuit.

    The features capture:
    - circuit size (number of qubits, number of gates),
    - how much entanglement is likely created,
    - how qubits are connected (interaction graph),
    - whether the circuit looks like a chain, a graph state, or a random circuit.

    These features are later used by Random Forest models to predict
    MIMIQ-MPS hyperparameters (bond_dim, ent_dim, trunc_eps, fuse, perm).
    """
    # Split the QASM file into lines
    lines = qasm.splitlines()
    # Build mapping from (register name, index) -> global qubit index
    # Example: q[0] -> 0, q[1] -> 1  etc.
    reg_sizes, gmap = build_qreg_layout(lines)
    # Total number of qubits in all qregs combined
    n_total = float(sum(reg_sizes.values())) if reg_sizes else 0.0
    # Size of the largest quantum register
    n_maxreg = float(max(reg_sizes.values())) if reg_sizes else 0.0
    # Number of separate quantum registers
    n_regs = float(len(reg_sizes))

    # Count how many times each gate appears (cx, u, h, t, etc.)
    counts = Counter()
    # Total number of gates, 1-qubit gates, 2-qubit (or more) gates, and non-unitary lines
    ops = ops_1q = ops_2q = non_u = 0

    # Stores how many times each pair of qubits interacted
    # Key = (u, v) qubit pair, Value = how many gates used this pair
    edge_counts = Counter()  
   
    # Loop over all QASM lines
    for ln in lines:
        # Skip empty lines and comments
        if is_comment_or_empty(ln):
            continue
        # Skip measurement, reset, barrier, etc. 
        if is_non_unitary_line(ln):
            non_u += 1
            continue
        # Extract gate name (e.g. "cx", "h", "rz", etc.)
        m_gate = GATE_RE.match(ln)
        if not m_gate:
            continue
        gname = m_gate.group(1).lower()
        # Extract all qubit operands like q[3], anc[5], etc.
        operands = [(r, int(i)) for (r, i) in QOP_RE.findall(ln)]
        if not operands:
            continue
        # Convert (reg, index) into global qubit numbers
        qbs: List[int] = []
        for r, i in operands:
            if (r, i) in gmap:
                qbs.append(gmap[(r, i)])
        if not qbs:
            continue
        # Count this gate
        counts[gname] += 1
        ops += 1
        # 1-qubit gate
        if len(qbs) == 1:
            ops_1q += 1
        # 2-qubit gate
        elif len(qbs) == 2:
            ops_2q += 1
            u, v = qbs
            if u != v:
                # Store the interaction edge in sorted order
                a, b = (u, v) if u < v else (v, u)
                edge_counts[(a, b)] += 1
        else:
            # Multi-qubit gate: treat as clique on unique operands
            ops_2q += 1
            uq = sorted(set(qbs))
            for i in range(len(uq)):
                for j in range(i + 1, len(uq)):
                    a, b = uq[i], uq[j]
                    edge_counts[(a, b)] += 1
    # Avoid division by zero later
    ops = max(ops, 1)
    # Number of CX gates
    cx = float(counts.get("cx", 0))
    # Number of T and T† gates (important for circuit type)
    t_count = float(counts.get("t", 0) + counts.get("tdg", 0))
    # Number of distinct interacting qubit pairs
    unique_edges = len(edge_counts)
    # Total number of 2-qubit interactions 
    total_edge_uses = float(sum(edge_counts.values()))

    # Compute graph statistics if there are interactions
    if unique_edges > 0:
        deg = Counter()  # degree of each qubit in the interaction graph
        spans: List[int] = []
        for (u, v) in edge_counts.keys():
            deg[u] += 1
            deg[v] += 1
            # Distance between interacting qubits
            spans.append(abs(u - v))

        deg_vals = list(deg.values())
        mu = sum(deg_vals) / len(deg_vals)
        # Variance of degrees: how unevenly connections are distributed
        degree_variance = sum((d - mu) ** 2 for d in deg_vals) / len(deg_vals)

        max_degree = float(max(deg_vals))
        avg_degree = float(mu)
        # Average and maximum distance between interacting qubits
        mean_edge_span = float(sum(spans) / len(spans))
        max_edge_span = float(max(spans))
    else:
        # No interactions at all
        degree_variance = 0.0
        max_degree = 0.0
        avg_degree = 0.0
        mean_edge_span = 0.0
        max_edge_span = 0.0

     # Number of possible qubit pairs
    n_int = int(n_total)
    denom = n_int * (n_int - 1) / 2 if n_int >= 2 else 0.0
    # How dense the interaction graph is
    edge_density = safe_div(unique_edges, denom) if denom else 0.
    # How many times, on average, each edge is reused
    edge_reuse = safe_div(total_edge_uses, float(unique_edges)) if unique_edges else 0.0
    # Final feature vector
    feats = {
        # Register structure
        "n_total": n_total,      # total number of qubits
        "n_maxreg": n_maxreg,    # size of the largest qreg
        "n_regs": n_regs,        # number of qregs

        # "Volumes"
        "ops": float(ops),       # total number of gates       
        "ops_1q": float(ops_1q), # number of 1-qubit gates
        "ops_2q": float(ops_2q), # number of 2-qubit gates
        "twoq_frac": safe_div(ops_2q, ops), # fraction of 2-qubit gates gates

        # Basic gate proxies
        "cx": cx,                # number of cx gates
        "t": t_count,            # number of T/TDG gates 
        "t_per_cx": safe_div(t_count, cx), # ratio of T to CX

        # Stabilized "volumes"
        "log_ops": lg(ops),      # log-scaled number of gates    
        "log_cx": lg(cx),        # log-scaled CX count

        # Non-unitary markers
        "non_unitary": float(non_u), # total number of non-unitary operations: measurements, resets, etc.

        # Interaction graph
        "unique_edges": float(unique_edges), # number of distinct qubit pairs
        "edge_density": float(edge_density), # how dense the interaction graph is 
        "edge_reuse": float(edge_reuse),     # how often edges are reused  
        "max_degree": float(max_degree),     # most connected qubit 
        "avg_degree": float(avg_degree),     # average connectivity  
        "degree_variance": float(degree_variance), # unevenness of connectivity

        # Chain-likeness proxies (can be useful  for perm/fuse)
        "mean_edge_span": float(mean_edge_span), # average distance between connected qubits
        "max_edge_span": float(max_edge_span),   # maximum distance 
        "log_unique_edges": lg(unique_edges),
        "log_edge_reuse": lg(edge_reuse),
        "log_mean_edge_span": lg(mean_edge_span),
        "log_max_edge_span": lg(max_edge_span),
    }
    return feats



# Exact-match cache (guarantee correct output on training set)
def normalize_qasm_for_hash(qasm: str) -> str:
    """
    Convert a QASM file into a normalized textual form.

    The purpose of this function is to make sure that two QASM files that represent the same circuit
    produce exactly the same text before hashing.

    This is needed because:
      - The ML model is not guaranteed to perfectly memorize small datasets
      - But we want a hard guarantee that training circuits always return
        their exact labeled hyperparameters

    To do this, we build a normalized representation of the circuit that:
      - Removes formatting differences
      - Removes comments
      - Preserves the semantic order of operations
    """
    out = []
    for ln in qasm.splitlines():
        s = ln.strip()
        if not s or s.startswith("//"):
            continue
        out.append(s)
    return "\n".join(out) + "\n"


def qasm_hash(qasm: str) -> str:
    """
    Compute a stable content-based hash of a QASM circuit.

    This function converts a QASM file into a "fingerprint"
    that uniquely identifies the *quantum circuit itself*, not the file.

    It is used to detect whether a circuit seen during prediction
    is exactly one of the circuits used during training.

    Why this is needed:
      - Machine learning models do NOT guarantee perfect recall
      - We must guarantee that training circuits always return
        their exact labeled hyperparameters
      - Therefore we store a hash -> label lookup table
.
    """
    norm = normalize_qasm_for_hash(qasm).encode("utf-8")
    return hashlib.sha256(norm).hexdigest()



# Schema
@dataclass(frozen=True)
class MIMIQParams:
    bond_dim: int
    ent_dim: int
    trunc_eps: float
    meth: str
    fuse: bool
    perm: bool



# CSV "utilities" - names of columns
REQUIRED_COLUMNS = ("qasm_path", "bond_dim", "ent_dim", "trunc_eps", "meth", "fuse", "perm")


def read_text(path: str) -> str:
    """
    Read the entire contents of a text file and return it as a string.

    This function is used to load QASM circuit files from disk
    both during training and during prediction.
    By this  we explicitly enforce UTF-8 encoding to avoid platform-specific
    bugs (Linux, Windows, Mac handle encodings differently)
    """
    # Open the file in text mode using UTF-8 encoding
    # UTF-8 is the standard encoding for OpenQASM files
    with open(path, "r", encoding="utf-8") as f:
         # Read the entire file into one string
        return f.read()


def load_labels_csv(path: str) -> List[Dict[str, str]]:
    """
    Read and validate the training labels CSV file.

    This function loads the CSV that defines the training dataset:
        qasm_path, bond_dim, ent_dim, trunc_eps, meth, fuse, perm

    It performs three tasks:
      1. Parses the CSV into structured rows
      2. Validates that all required columns exist
      3. Ensures data consistency before training
    """

    # Each element of rows will be a dictionary: { "qasm_path": "...", "bond_dim": "...", ... }
    rows: List[Dict[str, str]] = []
    # Will hold the CSV header once we read it
    header: List[str] | None = None
    # Open the CSV file
    with open(path, "r", encoding="utf-8") as f:
        # Process the file line by line
        for raw in f:
            # Remove surrounding whitespace and newline characters
            ln = raw.strip()
            # Skip empty lines and comment lines 
            if not ln or ln.startswith("#"):
                continue
            # Split the line on commas and strip whitespace from each field
            parts = [p.strip() for p in ln.split(",")]
            # First non-comment line is the CSV header
            if header is None:
                header = parts
                # Verify that all required columns exist
                missing = [c for c in REQUIRED_COLUMNS if c not in header]
                if missing:
                    raise ValueError(
                        f"labels CSV missing columns: {missing}\n"
                        f"Found header: {header}\n"
                        f"Expected: {list(REQUIRED_COLUMNS)}"
                    )
                # Move on to the next line 
                continue
            # Every data row must have the same number of fields as the header
            if len(parts) != len(header):
                raise ValueError(f"Bad CSV line (field mismatch): {ln}")
            # Convert ["file.qasm", "64", "4", ...] into
            # { "qasm_path": "file.qasm", "bond_dim": "64", ... }
            rows.append(dict(zip(header, parts)))
    # Training without data is meaningless -> error
    if not rows:
        raise ValueError("No valid rows found in labels CSV.")
    return rows




# Training
def train_model(labels_csv: str, outdir: str, seed: int = 0) -> None:
    """
    Train all ML models used to predict MIMIQ-MPS hyperparameters.

    This function builds a supervised learning pipeline that maps
    OpenQASM circuits -> optimal MIMIQ-MPS settings.

    Input:
    -------
    labels_csv
        CSV file containing training data with columns:
        qasm_path, bond_dim, ent_dim, trunc_eps, meth, fuse, perm

        Each row represents one quantum circuit and the hyperparameters
        that were empirically found to work best for it.

    outdir
        Directory where the trained models and metadata will be saved.
        It will contain:
            - model.joblib : all trained RandomForest models
            - meta.json   : feature ordering, method labels, trunc_eps grid,
                            power-of-two bounds, and exact-match cache

    seed
        Random seed for reproducibility of RandomForest training.

    What this function learns
    -------------------------
    It trains six independent models:
      1) clf_meth: predicts the MPS method (vmpoa, dmpo)
      2) clf_fuse: predicts whether gate fusion should be enabled
      3) clf_perm: predicts whether qubit permutation should be used
      4) reg_bond: predicts bond_dim 
      5) reg_ent : predicts ent_dim 
      6) reg_eps : predicts trunc_eps 

    All numeric quantities are learned in log-space because MPS parameters
    vary across orders of magnitude.

    How the training works
    ----------------------
    For each QASM circuit:

      1) The circuit is parsed into numerical features using extract_features().
         These features describe:
           - number of qubits
           - number of gates
           - number of 1q and 2q gates
           - structure of the interaction graph
           - degree and span statistics
           - non-unitary operations

      2) The target hyperparameters from the CSV are normalized:
           bond_dim : nearest power of two
           ent_dim  : nearest power of two
           trunc_eps: nearest value on the discrete grid (1e-1 … 1e-12)

      3) The circuit is also hashed and stored in exact_map so that if the
         same circuit appears at predict time, the exact original
         hyperparameters are returned (no ML error is allowed on training data).

    Why RandomForest is used
    ------------------------
    Random forests are chosen because they work well with small datasets 


    The forest is configured with:
      - bootstrap=False and min_samples_leaf=1 to allow exact memorization
      - many trees (1200) to reduce randomness and stabilize predictions

    The result
    ----------
    After this function finishes, the directory outdir contains
    everything needed to predict optimal MIMIQ-MPS hyperparameters
    for new QASM circuits.
    """
    # Create output directory if it does not exist.
    # This is where the trained model and metadata will be stored.
    os.makedirs(outdir, exist_ok=True)
    # Load and validate the training CSV.
    # Each row contains: qasm_path, bond_dim, ent_dim, trunc_eps, meth, fuse, perm
    rows = load_labels_csv(labels_csv)
    # Directory of the CSV file, used to resolve relative QASM paths.
    base_dir = os.path.dirname(os.path.abspath(labels_csv))
    # X_feats will store one feature dictionary per QASM circuit.
    # Each dictionary is produced by extract_features(qasm).
    X_feats: List[Dict[str, float]] = []
    # Labels for categorical and boolean outputs
    y_meth: List[str] = [] # MPS method (e.g. "vmpoa", "dmpo")
    y_fuse: List[int] = [] # 0 or 1
    y_perm: List[int] = [] # 0 or 1

    # Labels for continuous outputs, stored in log-space
    y_bond_log: List[float] = []  # log(bond_dim)
    y_ent_log: List[float] = []   # log(ent_dim)
    y_eps_log: List[float] = []   # log(trunc_eps)
    # exact_map stores a lookup table: hash(QASM) -> exact hyperparameters from CSV.
    exact_map: Dict[str, Dict[str, Any]] = {}
    # Loop over every row in the training CSV
    for r in rows:
        # Resolve QASM path (relative paths are relative to the CSV file)
        qasm_path = os.path.normpath(r["qasm_path"])
        if not os.path.isabs(qasm_path):
            qasm_path = os.path.normpath(os.path.join(base_dir, qasm_path))
        # Read QASM file
        qasm_txt = read_text(qasm_path)
        # Compute a stable hash of the QASM content.
        # This is used for the exact-match cache.
        h = qasm_hash(qasm_txt)
        # Parse and read hyperparameters from CSV
        bond = int(r["bond_dim"])
        ent = int(r["ent_dim"])
        eps = float(r["trunc_eps"])
        meth = r["meth"].strip()
        fuse = bool(parse_bool(r["fuse"]))
        perm = bool(parse_bool(r["perm"]))

        # Snap numeric values to the physically valid MIMIQ grid
        bond_n = snap_to_power_of_two(bond)
        ent_n = snap_to_power_of_two(ent)
        eps_n = snap_trunc_eps(eps)
        # Store exact parameters for this circuit in the hash map
        exact_map[h] = {
            "bond_dim": bond_n,
            "ent_dim": ent_n,
            "trunc_eps": eps_n,
            "meth": meth,
            "fuse": fuse,
            "perm": perm,
        }
        # Extract all features from the QASM
        feats = extract_features(qasm_txt)
        X_feats.append(feats)
        # Store categorical labels
        y_meth.append(meth)
        y_fuse.append(1 if fuse else 0)
        y_perm.append(1 if perm else 0)

        # Store continuous labels
        y_bond_log.append(math.log(bond_n))
        y_ent_log.append(math.log(ent_n))
        y_eps_log.append(math.log(eps_n))
    # Build a stable feature ordering so training and predicting match
    keys = sorted(X_feats[0].keys())
    # Convert feature dictionaries into a numeric matrix
    Xm = np.array([[x[k] for k in keys] for x in X_feats], dtype=float)
    # Convert method labels into integer class IDs
    meth_set = sorted(set(y_meth))
    meth_id = {m: i for i, m in enumerate(meth_set)}
    y_meth_id = np.array([meth_id[m] for m in y_meth], dtype=int)
    # Classifier for MPS method
    clf_meth = RandomForestClassifier(
        n_estimators=1200, n_jobs=-1, random_state=seed,
        class_weight="balanced", bootstrap=False, min_samples_leaf=1
    )
    # Classifier for fuse flag
    clf_fuse = RandomForestClassifier(
        n_estimators=1200, n_jobs=-1, random_state=seed,
        class_weight="balanced", bootstrap=False, min_samples_leaf=1
    )
    # Classifier for perm flag
    clf_perm = RandomForestClassifier(
        n_estimators=1200, n_jobs=-1, random_state=seed,
        class_weight="balanced", bootstrap=False, min_samples_leaf=1
    )
    # Regressor for trunc_eps
    reg_eps = RandomForestRegressor(
        n_estimators=1200, n_jobs=-1, random_state=seed,
        bootstrap=False, min_samples_leaf=1
    )
    # Regressor for bond_dim
    reg_bond = RandomForestRegressor(
        n_estimators=1200, n_jobs=-1, random_state=seed,
        bootstrap=False, min_samples_leaf=1
    )
    # Regressor for ent_dim
    reg_ent = RandomForestRegressor(
        n_estimators=1200, n_jobs=-1, random_state=seed,
        bootstrap=False, min_samples_leaf=1
    )
    # Train classifiers
    clf_meth.fit(Xm, y_meth_id)
    clf_fuse.fit(Xm, np.array(y_fuse, dtype=int))
    clf_perm.fit(Xm, np.array(y_perm, dtype=int))
    # Train regressors
    reg_bond.fit(Xm, np.array(y_bond_log, dtype=float))
    reg_ent.fit(Xm, np.array(y_ent_log, dtype=float))
    reg_eps.fit(Xm, np.array(y_eps_log, dtype=float))
    # Save trained models to disk
    joblib.dump((clf_meth, clf_fuse, clf_perm, reg_eps, reg_bond, reg_ent),
                os.path.join(outdir, "model.joblib"))
    # Save metadata needed for prediction
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({
            "keys": keys,
            "methods": meth_set,
            "trunc_eps_grid": TRUNC_EPS_GRID,
            "min_dim": MIN_DIM,
            "max_dim": MAX_DIM,
            "exact_map": exact_map,
        }, f, indent=2)


# Prediction 
def predict(modeldir: str, qasm_path: str) -> MIMIQParams:
    """
    Predict the optimal MIMIQ-MPS hyperparameters for a given QASM circuit.

    This function loads a trained ML model and uses it to infer:
        - bond_dim
        - ent_dim
        - trunc_eps
        - meth
        - fuse
        - perm

    It implements a strict two-level inference strategy:

    Level 1 — Exact match 
    ----------------------------------------------
    If the QASM file was seen during training, the function returns
    the *exact* hyperparameters stored in the training CSV.

    Level 2 — ML inference
    ------------------------------------
    If the QASM circuit is new, the function:
      1) Extracts structural features from the circuit
      2) Feeds them into six trained RandomForest models
      3) Converts the predictions back into valid MIMIQ-MPS parameters
      4) Snaps them to physically valid discrete values

    The output is guaranteed to satisfy:
      - bond_dim, ent_dim in {4, 8, 16, …, 4096}
      - trunc_eps in {1e-1, 1e-2, …, 1e-12}
    """
    # Construct path to trained model file (contains all forests)
    model_path = os.path.join(modeldir, "model.joblib")
    # Construct path to metadata (feature order, method names, exact-match cache)
    meta_path = os.path.join(modeldir, "meta.json")
    # If either file is missing, the model directory is invalid
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Model files not found in directory: {modeldir}")

    # Load the six trained models from disk
    clf_meth, clf_fuse, clf_perm, reg_eps, reg_bond, reg_ent = joblib.load(model_path)
    # Load metadata: feature ordering, method labels, etc.
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # List of feature names in the exact order used during training
    keys: List[str] = meta["keys"]
    # List of allowed method labels (["dmpo", "vmpoa"])
    methods: List[str] = meta["methods"]

    # Discrete truncation grid (1e-1 ... 1e-12)
    eps_grid: List[float] = meta["trunc_eps_grid"]
    # Dictionary: QASM hash -> true training parameters
    exact_map: Dict[str, Dict[str, Any]] = meta.get("exact_map", {})
    # Read the QASM file
    qasm_txt = read_text(qasm_path)
    # Compute its normalized hash
    h = qasm_hash(qasm_txt)

    # 1. If this circuit was seen during training, return the exact stored values
    if h in exact_map:
        d = exact_map[h]
        return MIMIQParams(
            bond_dim=int(d["bond_dim"]),
            ent_dim=int(d["ent_dim"]),
            trunc_eps=float(d["trunc_eps"]),
            meth=str(d["meth"]),
            fuse=bool(d["fuse"]),
            perm=bool(d["perm"]),
        )


    # 2. Unseen circuit -> ML prediction + snapping
    # Extract numerical features from the QASM circuit
    feats = extract_features(qasm_txt)
    # Convert the feature dictionary into a vector in the same order as training
    x = np.array([[feats[k] for k in keys]], dtype=float)
    # Predict the MPS method using class probabilities
    meth = methods[int(np.argmax(clf_meth.predict_proba(x)[0]))]
    # Predict fuse and perm by selecting the class with highest probability
    fuse = bool(clf_fuse.classes_[int(np.argmax(clf_fuse.predict_proba(x)[0]))])
    perm = bool(clf_perm.classes_[int(np.argmax(clf_perm.predict_proba(x)[0]))])
    # Predict log(trunc_eps) and convert back to linear scale
    eps_raw = math.exp(float(reg_eps.predict(x)[0]))
    # Snap trunc_eps to nearest allowed discrete grid value
    trunc_eps = snap_trunc_eps(eps_raw)
    # Predict log(bond_dim) and log(ent_dim)
    bond_raw = math.exp(float(reg_bond.predict(x)[0]))
    ent_raw = math.exp(float(reg_ent.predict(x)[0]))
    # Snap bond_dim and ent_dim to nearest power-of-two
    bond_dim = snap_to_power_of_two(bond_raw)
    ent_dim = snap_to_power_of_two(ent_raw)
    # Return the final validated parameter set
    return MIMIQParams(
        bond_dim=bond_dim,
        ent_dim=ent_dim,
        trunc_eps=trunc_eps,
        meth=meth,
        fuse=fuse,
        perm=perm,
    )



# CLI
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mimiq_mps_recommender",
        description="Train and apply an ML-based hyperparameter recommender for MIMIQ-MPS.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train a model from a labels CSV.")
    tr.add_argument("--labels", required=True, help="Path to labels CSV.")
    tr.add_argument("--outdir", required=True, help="Output directory for the trained model.")
    tr.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")

    pr = sub.add_parser("predict", help="Predict parameters for a given QASM circuit.")
    pr.add_argument("--modeldir", required=True, help="Directory produced by 'train'.")
    pr.add_argument("--qasm", required=True, help="Path to the QASM circuit.")
    pr.add_argument("--json", action="store_true", help="Print raw JSON only.")

    dbg = sub.add_parser("features", help="Print extracted features for a given QASM circuit (debug).")
    dbg.add_argument("--qasm", required=True, help="Path to the QASM circuit.")

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.cmd == "train":
        train_model(args.labels, args.outdir, seed=args.seed)
        return

    if args.cmd == "features":
        feats = extract_features(read_text(args.qasm))
        print(json.dumps(feats, indent=2))
        return

    params = predict(args.modeldir, args.qasm)

    if args.json:
        print(json.dumps(asdict(params), indent=2))
        return

    print("\nRecommended MIMIQ-MPS configuration")
    print("----------------------------------")
    print(json.dumps(asdict(params), indent=2))
    print("\nNotes:")
    print(f"- bond_dim and ent_dim snapped to powers of two within [{MIN_DIM}, {MAX_DIM}]")
    print(f"- trunc_eps is discrete on grid: 1e-1 ... 1e-12")
    print(f"- Use `features --qasm <file>` to inspect what the model sees.\n")


if __name__ == "__main__":
    main()



