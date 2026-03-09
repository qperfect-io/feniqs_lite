#
# Copyright © 2024 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""
smart_qasm_optimizer.py

Machine-learning-guided orchestrator for full-circuit quantum optimizers.

Overview
--------
This script is an orchestration layer for external quantum circuit optimizers
that operate on complete OpenQASM circuits.

It does not implement low-level quantum rewriting rules internally. Instead, it:

- extracts lightweight structural features from an input QASM circuit,
- uses a trained machine-learning model to rank full-circuit optimization
  strategies,
- evaluates candidate optimizers in probability order,
- accepts the first successful strategy that improves the circuit according to
  the selected cost metric.

Supported optimization strategies
---------------------------------
The model selects one label among the following full-circuit optimizers:

- staq_O3
- feyn_ppf
- feyn_O4
- feyn_tpar

Design constraints
------------------
- All optimizers are applied to the entire input circuit.
- No chunking is used.
- The script is therefore a full-circuit strategy selector.

Quality metric
--------------
The optimization objective is:

    cost = T_count + 10 * CX_count

where:
- T_count includes both T and TDG gates,
- CX_count includes CX gates.

This metric intentionally assigns a stronger penalty to CX gates.

Training procedure
------------------
For each training circuit:
- all available full-circuit strategies are evaluated under a timeout,
- the best improving strategy is used as the label,
- circuits for which no optimizer improves the baseline are skipped.

Inference procedure
-------------------
For a new circuit:
- features are extracted,
- the classifier ranks all strategies by predicted probability,
- strategies are tried in that order,
- the first successful strategy that improves the cost is accepted,
- otherwise the original circuit is preserved.

Dependencies
------------
Python packages:
- numpy
- scikit-learn
- joblib

External tools:
- staq
- feynopt

Notes
-----
This script is a ML-based orchestrator for different qyantum circuit optimizers.
Its purpose is to decide which external optimizer is most promising for a given
QASM circuit and to run that optimizer in a controlled and reproducible way.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
except ImportError as exc:
    raise SystemExit(
        "ERROR: install scikit-learn + joblib + numpy\n"
        "Example: pip install scikit-learn joblib numpy"
    ) from exc



# Persistent model artifacts
MODEL_FILE = "optimizer_selector.joblib"
META_FILE = "optimizer_selector_meta.json"



# Runtime defaults (seconds)
MAX_SOLVER_TIME = 30000 



# Available full-circuit strategies
FULL_SOLVERS: List[Tuple[str, List[str]]] = [
    ("staq_O3", ["staq", "{inp}", "-s", "-c", "-r", "-O3"]),
    ("feyn_ppf", ["feynopt", "-ppf", "{inp}"]),
    ("feyn_O4", ["feynopt", "-O4", "{inp}"]),
    ("feyn_tpar", ["feynopt", "-tpar", "{inp}"]),
]

ALL_LABELS = [name for name, _ in FULL_SOLVERS]



# QASM parsing helpers
QREG = re.compile(r"qreg\s+\w+\[(\d+)\]\s*;", re.IGNORECASE)
G1 = re.compile(r"^\s*([a-zA-Z_]\w*)\s+q\[(\d+)\]\s*;\s*$")
G2 = re.compile(r"^\s*([a-zA-Z_]\w*)\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]\s*;\s*$")
RZ = re.compile(r"^\s*rz\s*\(.*\)\s+q\[(\d+)\]\s*;\s*$", re.IGNORECASE)

NON_UNITARY_PREFIX = ("measure", "barrier", "creg", "if(", "reset")


def is_non_unitary_line(line: str) -> bool:
    """
    Determine whether a QASM line is non-unitary.

    Parameters
    ----------
    line : str
        Input QASM line.

    Returns
    -------
    bool
        True if the line starts with a non-unitary instruction.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("//"):
        return False
    return stripped.lower().startswith(NON_UNITARY_PREFIX)



# Feature extraction
def extract_features(qasm: str) -> Dict[str, float]:
    """
    Extract lightweight structural features from an OpenQASM circuit.

    The feature set is intentionally inexpensive and robust. It captures:
    - qubit count,
    - file size,
    - operation counts,
    - one-qubit and two-qubit balance,
    - common-gate frequencies,
    - simple density ratios,
    - non-unitary fraction,
    - a cheap gate-stream alternation signal.

    Parameters
    ----------
    qasm : str
        Full QASM text.

    Returns
    -------
    dict[str, float]
        Numeric feature dictionary.
    """
    counts = Counter()
    num_qubits = 0
    total_ops = 0
    ops_1q = 0
    ops_2q = 0
    non_unitary_count = 0
    gate_stream: List[str] = []

    for line in qasm.splitlines():
        qreg_match = QREG.search(line)
        if qreg_match:
            num_qubits = max(num_qubits, int(qreg_match.group(1)))

        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        if is_non_unitary_line(stripped):
            non_unitary_count += 1
            continue

        if RZ.match(stripped):
            gate_name = "rz"
            counts[gate_name] += 1
            total_ops += 1
            ops_1q += 1
            gate_stream.append(gate_name)
            continue

        match_2q = G2.match(stripped)
        if match_2q:
            gate_name = match_2q.group(1).lower()
            counts[gate_name] += 1
            total_ops += 1
            ops_2q += 1
            gate_stream.append(gate_name)
            continue

        match_1q = G1.match(stripped)
        if match_1q:
            gate_name = match_1q.group(1).lower()
            counts[gate_name] += 1
            total_ops += 1
            ops_1q += 1
            gate_stream.append(gate_name)

    total_ops = max(total_ops, 1)
    file_size_kb = len(qasm.encode("utf-8")) / 1024.0

    t_count = counts["t"] + counts["tdg"]
    cx_count = counts["cx"]
    h_count = counts["h"]
    rz_count = counts["rz"]

    def safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    def log1p_nonneg(x: float) -> float:
        return math.log1p(max(0.0, x))

    hx_switch = sum(
        1
        for i in range(1, len(gate_stream))
        if {"h", "cx"} == {gate_stream[i - 1], gate_stream[i]}
    )

    ops_per_kb = safe_div(total_ops, file_size_kb)

    features: Dict[str, float] = {
        "n": float(num_qubits),
        "ops": float(total_ops),
        "ops_1q": float(ops_1q),
        "ops_2q": float(ops_2q),
        "non_unitary": float(non_unitary_count),
        "t": float(t_count),
        "cx": float(cx_count),
        "h": float(h_count),
        "rz": float(rz_count),
        "t_density": safe_div(t_count, total_ops),
        "cx_density": safe_div(cx_count, total_ops),
        "h_density": safe_div(h_count, total_ops),
        "rz_density": safe_div(rz_count, total_ops),
        "t_per_cx": safe_div(t_count, cx_count),
        "twoq_frac": safe_div(ops_2q, total_ops),
        "nonu_frac": safe_div(non_unitary_count, total_ops + non_unitary_count),
        "hx_switch": float(hx_switch),
        "filesize_kb": float(file_size_kb),
        "ops_per_kb": float(ops_per_kb),
        "log_ops": log1p_nonneg(total_ops),
        "log_filesize_kb": log1p_nonneg(file_size_kb),
        "log_cx": log1p_nonneg(cx_count),
        "log_t": log1p_nonneg(t_count),
    }

    for gate_name in ("x", "y", "z", "s", "sdg", "p", "u1", "u2", "u3", "rx", "ry", "cz", "swap"):
        features[f"cnt_{gate_name}"] = float(counts.get(gate_name, 0))

    return features



# Cost metric
def cost(qasm: str) -> int:
    """
    Compute the weighted circuit cost.

    Metric
    ------
    cost = T_count + 10 * CX_count

    Parameters
    ----------
    qasm : str
        Full QASM text.

    Returns
    -------
    int
        Weighted cost value.
    """
    t_count = 0
    cx_count = 0

    for line in qasm.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("t ") or stripped.startswith("tdg "):
            t_count += 1
        if stripped.startswith("cx "):
            cx_count += 1

    return t_count + 10 * cx_count



# External solver execution
def run_solver(command_template: Sequence[str], qasm: str, timeout: int) -> Optional[str]:
    """
    Execute an external full-circuit optimizer with a hard timeout.

    The input QASM is written to a temporary file. The solver command may
    reference that file using the placeholder '{inp}'.

    Parameters
    ----------
    command_template : Sequence[str]
        External command template.
    qasm : str
        Input QASM text.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    str | None
        Solver stdout if successful and non-empty, otherwise None.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "in.qasm")

        with open(input_path, "w", encoding="utf-8") as handle:
            handle.write(qasm)

        command = [token.format(inp=input_path) for token in command_template]

        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None

        stdout = completed.stdout
        if stdout is None or not stdout.strip():
            return None

        return stdout



# Strategy evaluation
def evaluate_full_solver(name: str, qasm: str, timeout: int) -> Tuple[Optional[str], Optional[int]]:
    """
    Evaluate one full-circuit optimization strategy.

    Parameters
    ----------
    name : str
        Strategy label.
    qasm : str
        Input QASM text.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    tuple[str | None, int | None]
        Output QASM and its cost, or (None, None) on failure.
    """
    command_template = next(command for solver_name, command in FULL_SOLVERS if solver_name == name)
    output_qasm = run_solver(command_template, qasm, timeout)

    if not output_qasm:
        return None, None

    return output_qasm, cost(output_qasm)



# Training
@dataclass
class TrainingResult:
    """
    In-memory training result.
    """
    classifier: RandomForestClassifier
    feature_keys: List[str]


def train_model(train_dir: str, full_timeout: int) -> TrainingResult:
    """
    Train the full-circuit strategy selector.

    Label generation
    ----------------
    For each training circuit:
    - compute the baseline cost,
    - evaluate all full-circuit strategies under the given timeout,
    - choose the strategy that produces the best improvement,
    - skip the circuit if no strategy improves the baseline.

    Parameters
    ----------
    train_dir : str
        Directory containing training QASM files.
    full_timeout : int
        Timeout per solver call.

    Returns
    -------
    TrainingResult
        Trained classifier and feature key order.
    """
    feature_rows: List[Dict[str, float]] = []
    labels: List[int] = []

    files = [name for name in os.listdir(train_dir) if name.endswith(".qasm")]
    if not files:
        raise SystemExit(f"[TRAIN] no .qasm files found in: {train_dir}")

    print(f"[TRAIN] {len(files)} circuits", file=sys.stderr)
    print(f"        full_timeout={full_timeout}s", file=sys.stderr)

    for filename in files:
        path = os.path.join(train_dir, filename)
        with open(path, encoding="utf-8") as handle:
            qasm = handle.read()

        features = extract_features(qasm)
        base_cost = cost(qasm)
        best_label = "identity"
        best_cost = base_cost

        for label in ALL_LABELS:
            output_qasm, candidate_cost = evaluate_full_solver(label, qasm, timeout=full_timeout)
            if output_qasm and candidate_cost is not None and candidate_cost < best_cost:
                best_cost = candidate_cost
                best_label = label

        if best_label != "identity":
            feature_rows.append(features)
            labels.append(ALL_LABELS.index(best_label))
            print(
                f"  {filename} -> {best_label} "
                f"(cost {base_cost} -> {best_cost})",
                file=sys.stderr,
            )
        else:
            print(
                f"  {filename} -> (no improvement within budget)",
                file=sys.stderr,
            )

    if not feature_rows:
        raise SystemExit(
            "[TRAIN] no labeled examples produced (all identity).\n"
            "Increase timeouts or add more diverse training circuits."
        )

    feature_keys = sorted(feature_rows[0].keys())
    x_matrix = np.array(
        [[row.get(key, 0.0) for key in feature_keys] for row in feature_rows],
        dtype=float,
    )
    y_vector = np.array(labels, dtype=int)

    classifier = RandomForestClassifier(
        n_estimators=800,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=0,
    )
    classifier.fit(x_matrix, y_vector)

    joblib.dump(classifier, MODEL_FILE)

    with open(META_FILE, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "keys": feature_keys,
                "labels": ALL_LABELS,
                "full_timeout": full_timeout,
            },
            handle,
            indent=2,
        )

    print("[TRAIN] done", file=sys.stderr)
    return TrainingResult(classifier=classifier, feature_keys=feature_keys)


def load_model() -> Tuple[RandomForestClassifier, List[str], List[str], Dict[str, object]]:
    """
    Load a previously trained model and its metadata.

    Returns
    -------
    tuple
        (classifier, feature_keys, labels, metadata)
    """
    classifier = joblib.load(MODEL_FILE)

    with open(META_FILE, encoding="utf-8") as handle:
        metadata = json.load(handle)

    feature_keys = metadata["keys"]
    labels = metadata["labels"]
    return classifier, feature_keys, labels, metadata



# Inference
def rank_strategies(
    classifier: RandomForestClassifier,
    feature_keys: Sequence[str],
    labels: Sequence[str],
    qasm_text: str,
) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    """
    Rank all full-circuit strategies by predicted probability.

    Parameters
    ----------
    classifier : RandomForestClassifier
        Trained model.
    feature_keys : Sequence[str]
        Ordered model feature names.
    labels : Sequence[str]
        Label vocabulary.
    qasm_text : str
        Input QASM text.

    Returns
    -------
    tuple[list[tuple[str, float]], dict[str, float]]
        Ranked strategies and extracted features.
    """
    features = extract_features(qasm_text)
    x = np.array([[features.get(key, 0.0) for key in feature_keys]], dtype=float)
    probabilities = classifier.predict_proba(x)[0]
    order = np.argsort(-probabilities)
    ranked = [(labels[int(i)], float(probabilities[int(i)])) for i in order]
    return ranked, features


def write_output(text: str, output_path: str) -> None:
    """
    Write output QASM to a file or stdout.

    Parameters
    ----------
    text : str
        Output QASM text.
    output_path : str
        Destination file path, or '-' for stdout.
    """
    if output_path in {"", "-"}:
        print(text, end="")
        return

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)


def optimize_with_model(
    classifier: RandomForestClassifier,
    feature_keys: Sequence[str],
    labels: Sequence[str],
    qasm: str,
    output_path: str,
    full_timeout: int,
) -> None:
    """
    Run model-guided full-circuit optimization.

    Execution policy
    ----------------
    - rank strategies by predicted probability,
    - try them one by one in descending order,
    - accept the first strategy that succeeds and improves the cost,
    - if none improves the circuit, emit the original input.

    Parameters
    ----------
    classifier : RandomForestClassifier
        Trained model.
    feature_keys : Sequence[str]
        Ordered model feature names.
    labels : Sequence[str]
        Label vocabulary.
    qasm : str
        Input QASM text.
    output_path : str
        Output file path or '-' for stdout.
    full_timeout : int
        Timeout per solver call.
    """
    ranked, features = rank_strategies(
        classifier=classifier,
        feature_keys=feature_keys,
        labels=labels,
        qasm_text=qasm,
    )

    print(
        f"[INFO] filesize_kb={features['filesize_kb']:.1f} "
        f"ops≈{int(features['ops'])} "
        f"n≈{int(features['n'])}",
        file=sys.stderr,
    )
    print("[INFO] ranked strategies:", file=sys.stderr)
    for label, probability in ranked:
        print(f"   {label:16s}  p={probability:.3f}", file=sys.stderr)

    base_cost = cost(qasm)
    best_output = qasm
    best_cost = base_cost
    best_strategy = "identity"

    for label, probability in ranked:
        if probability <= 0.0:
            continue

        print(f"[TRY] {label} (timeout {full_timeout}s)", file=sys.stderr)
        output_qasm, candidate_cost = evaluate_full_solver(label, qasm, timeout=full_timeout)

        if output_qasm and candidate_cost is not None:
            print(f"[OK] {label} cost={candidate_cost}", file=sys.stderr)
            if candidate_cost < best_cost:
                best_output = output_qasm
                best_cost = candidate_cost
                best_strategy = label
                break
        print(f"[FAIL] {label}", file=sys.stderr)

    print(f"[DONE] selected={best_strategy}  cost={base_cost}->{best_cost}", file=sys.stderr)
    write_output(best_output, output_path)



# CLI
def main() -> None:
    """
    Command-line entry point.

    Behavior
    --------
    - If the model does not exist, --train_dir must be provided.
    - Otherwise, the existing model is loaded and used for inference.

    Parameters
    ----------
    qasm : str
        Input QASM file.
    -o / --out : str
        Output file, or '-' for stdout.
    --train_dir : str
        Directory of training QASM files, used only when the model is missing.
    --timeout : int
        Timeout per full-circuit solver call.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("qasm", help="Input QASM file.")
    parser.add_argument(
        "-o",
        "--out",
        default="-",
        help="Output QASM file. Default: stdout.",
    )
    parser.add_argument(
        "--train_dir",
        help="Directory with training QASM files, used only when no model exists.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=MAX_SOLVER_TIME,
        help="Timeout per full-circuit solver call in seconds.",
    )

    args = parser.parse_args()

    if not os.path.exists(MODEL_FILE) or not os.path.exists(META_FILE):
        if not args.train_dir:
            raise SystemExit("Model not found. Use --train_dir <folder_with_qasm>")

        training_result = train_model(
            train_dir=args.train_dir,
            full_timeout=args.timeout,
        )
        classifier = training_result.classifier
        feature_keys = training_result.feature_keys
        labels = ALL_LABELS
    else:
        classifier, feature_keys, labels, _ = load_model()

    with open(args.qasm, encoding="utf-8") as handle:
        qasm_text = handle.read()

    optimize_with_model(
        classifier=classifier,
        feature_keys=feature_keys,
        labels=labels,
        qasm=qasm_text,
        output_path=args.out,
        full_timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
