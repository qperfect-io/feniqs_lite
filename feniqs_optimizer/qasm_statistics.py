#!/usr/bin/env python3
"""
OpenQASM 2.0 gate statistics with CCX and SWAP decomposition and circuit export.

Behavior
--------
1. Read an OpenQASM 2.0 circuit.
2. Decompose all CCX (Toffoli) gates into a standard 1Q/2Q Clifford+T sequence.
3. Decompose all SWAP gates into 3 CX gates.
4. Save the decomposed circuit to disk (CCX- and SWAP-free).
5. Compute gate statistics exclusively on the decomposed circuit.

This workflow ensures that reported metrics are consistent with
benchmarking, cost modeling, and algebraic optimization tools.
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple

_QUBIT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\[[0-9]+\]")



# Parsing helpers
def _is_non_gate_statement(line_lower: str) -> bool:
    return (
        line_lower.startswith("openqasm")
        or line_lower.startswith("include")
        or line_lower.startswith("qreg")
        or line_lower.startswith("creg")
        or line_lower.startswith("measure")
        or line_lower.startswith("reset")
        or line_lower.startswith("barrier")
    )


def _gate_name(line: str) -> str:
    return re.split(r"\s|\(", line.strip(), maxsplit=1)[0].lower()


def _qubits(line: str, gate: str) -> List[str]:
    return _QUBIT_RE.findall(line[len(gate):])



# Gate decompositions
def _decompose_ccx(c1: str, c2: str, t: str) -> List[str]:
    # Standard Clifford+T Toffoli decomposition
    return [
        f"h {t};",
        f"cx {c2},{t};",
        f"tdg {t};",
        f"cx {c1},{t};",
        f"t {t};",
        f"cx {c2},{t};",
        f"tdg {t};",
        f"cx {c1},{t};",
        f"t {c2};",
        f"t {t};",
        f"h {t};",
        f"cx {c1},{c2};",
        f"t {c1};",
        f"tdg {c2};",
        f"cx {c1},{c2};",
    ]


def _decompose_swap(q1: str, q2: str) -> List[str]:
    return [
        f"cx {q1},{q2};",
        f"cx {q2},{q1};",
        f"cx {q1},{q2};",
    ]



# Decomposition + export
def decompose_and_export(
    input_path: Path, output_path: Path
) -> Tuple[List[str], int, int]:

    """
    Decompose CCX and SWAP gates and write a normalized OpenQASM file.

    Returns
    -------
    decomposed_gate_lines : List[str]
        Gate application lines used for statistics.
    ccx_count : int
        Number of CCX gates decomposed.
    swap_count : int
        Number of SWAP gates decomposed.
    """
    ccx_count = 0
    swap_count = 0
    decomposed_gate_lines: List[str] = []

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for raw in fin:
            stripped = raw.strip()
            low = stripped.lower()

            # Preserve headers, declarations, and non-unitary statements
            if not stripped or stripped.startswith("//") or _is_non_gate_statement(low):
                fout.write(raw)
                continue

            if not stripped.endswith(";"):
                fout.write(raw)
                continue

            gate = _gate_name(stripped)

            # Preserve gate declarations
            if gate == "gate":
                fout.write(raw)
                continue

            # SWAP decomposition 
            if gate == "swap":
                qubits = _qubits(stripped, gate)
                if len(qubits) != 2:
                    raise ValueError(f"Malformed SWAP statement: {stripped}")

                swap_count += 1
                for g in _decompose_swap(qubits[0], qubits[1]):
                    fout.write(g + "\n")
                    decomposed_gate_lines.append(g)
                continue

            # CCX decomposition 
            if gate == "ccx":
                qubits = _qubits(stripped, gate)
                if len(qubits) != 3:
                    raise ValueError(f"Malformed CCX statement: {stripped}")

                ccx_count += 1
                for g in _decompose_ccx(qubits[0], qubits[1], qubits[2]):
                    fout.write(g + "\n")
                    decomposed_gate_lines.append(g)
                continue

            # Other gates: keep as is
            fout.write(raw)
            decomposed_gate_lines.append(stripped)

    return decomposed_gate_lines, ccx_count, swap_count



# Statistics
def analyze(qasm_path: str) -> None:
    input_path = Path(qasm_path)
    output_path = input_path.with_name(input_path.stem + "_decomposed.qasm")

    gate_lines, ccx_count, swap_count = decompose_and_export(
        input_path, output_path
    )

    total = t_cnt = rz_cnt = cx_cnt = twoq_cnt = 0
    multi_qubit = False

    for line in gate_lines:
        gate = _gate_name(line)
        qs = _qubits(line, gate)

        if len(qs) > 2:
            multi_qubit = True

        total += 1
        if gate in ("t", "tdg"):
            t_cnt += 1
        if gate == "rz":
            rz_cnt += 1
        if gate in ("cx", "cnot"):
            cx_cnt += 1
        if len(qs) == 2:
            twoq_cnt += 1

    if multi_qubit:
        raise RuntimeError(
            "Multi-qubit gate detected after decomposition. "
            "Statistics are defined only for 1Q/2Q circuits."
        )

    print(f"Input circuit                 : {input_path}")
    print(f"Decomposed circuit saved as   : {output_path}")
    print(f"CCX gates decomposed          : {ccx_count}")
    print(f"SWAP gates decomposed         : {swap_count}")
    print(f"Total gates (decomposed)      : {total}")
    print(f"T / TDG gates                  : {t_cnt}")
    print(f"RZ gates                      : {rz_cnt}")
    print(f"Two-qubit gates               : {twoq_cnt}")
    print(f"CX / CNOT gates               : {cx_cnt}")



# Entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_qasm.py <circuit.qasm>")
        sys.exit(1)

    analyze(sys.argv[1])
