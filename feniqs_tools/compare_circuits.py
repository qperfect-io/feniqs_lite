# Copyright © 2025 QPerfect. All Rights Reserved.
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

import sys
import time
from typing import Optional, Tuple, Dict, Any

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

"""
This script loads one or two quantum circuits from OpenQASM files and performs
normalization and comparison using Qiskit.

Main functionality
------------------
1. If one QASM file is provided:
   - the script loads the circuit,
   - transforms it into a normalized ("canonical-like") form,
   - prints the resulting circuit representation.

2. If two QASM files are provided:
   - the script loads both circuits,
   - normalizes them,
   - checks whether they are equivalent.

Comparison strategy
-------------------
The comparison is performed in two stages:

1. Exact comparison (when affordable):
   - the script tries to simulate both circuits as statevectors,
   - then compares the resulting quantum states produced from the default
     initial state |0...0>.

2. Sampling-based comparison (fallback):
   - if exact statevector comparison is too expensive or fails,
   - the script measures both circuits using a simulator,
   - compares their output distributions,
   - computes the Total Variation Distance (TVD).

Important note
--------------
This script is intended as a practical circuit-comparison utility.
It does not guarantee full operator-level equivalence for all possible input
states. In exact mode, it compares the final states obtained from |0...0>.
In fallback mode, it compares measurement statistics in the computational basis.

Command-line parameters
-----------------------
The script accepts the following command-line arguments:

    python compare_circuits.py circuit1.qasm [circuit2.qasm]

Parameters
----------
circuit1.qasm : str
    Path to the first OpenQASM file.
    If this is the only argument, the script prints the normalized form
    of the circuit.

circuit2.qasm : str, optional
    Path to the second OpenQASM file.
    If provided, the script compares the two circuits and prints whether
    they are considered equivalent.

Outputs
-------
Depending on the mode, the script prints one of the following:
- the normalized circuit form,
- "Equal" if the circuits are considered equivalent,
- "Not equal" if they differ,
- diagnostic information such as timing, comparison method, and TVD.

Typical usage
-------------
1. Normalize a single circuit:
    python compare_circuits.py circuit1.qasm

2. Compare two circuits:
    python compare_circuits.py circuit1.qasm circuit2.qasm
"""

DEFAULT_BASIS_GATES = ["cx", "u3"]
MAX_EXACT_QUBITS = 14
DEFAULT_SHOTS = 1024
DEFAULT_TOLERANCE = 0.02


def load_qasm(path: str) -> QuantumCircuit:
    """
    Load a quantum circuit from an OpenQASM file.
    """
    return QuantumCircuit.from_qasm_file(path)


def canonical_form(
    circuit: QuantumCircuit,
    basis_gates: Optional[list[str]] = None,
    optimization_level: int = 1,
    decompose_circuit: bool = False,
    reverse_bits_order: bool = False,
) -> QuantumCircuit:
    """
    Build a normalized representation of a circuit for comparison.

    Notes
    -----
    This is a heuristic normalization, not a mathematically unique canonical form.

    Parameters
    ----------
    circuit : QuantumCircuit
        Input circuit.
    basis_gates : list[str] | None
        Basis gates used for transpilation.
    optimization_level : int
        Qiskit transpiler optimization level.
    decompose_circuit : bool
        Whether to decompose composite instructions before transpilation.
    reverse_bits_order : bool
        Whether to reverse qubit order for normalization.

    Returns
    -------
    QuantumCircuit
        Normalized circuit.
    """
    normalized = circuit.remove_final_measurements(inplace=False)

    if decompose_circuit:
        normalized = normalized.decompose()

    if reverse_bits_order:
        normalized = normalized.reverse_bits()

    normalized = transpile(
        normalized,
        basis_gates=basis_gates or DEFAULT_BASIS_GATES,
        optimization_level=optimization_level,
    )
    return normalized


def add_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Return a measured copy of the input circuit.
    """
    measured = circuit.copy()
    measured.measure_all()
    return measured


def simulate_counts(
    circuit: QuantumCircuit,
    simulator: AerSimulator,
    shots: int,
) -> dict:
    """
    Simulate a measured circuit and return counts.
    """
    compiled = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(compiled, shots=shots).result()
    return result.get_counts()


def total_variation_distance(counts1: dict, counts2: dict, shots: int) -> float:
    """
    Compute total variation distance between two empirical distributions.
    """
    keys = set(counts1) | set(counts2)
    distance = 0.0

    for key in keys:
        p = counts1.get(key, 0) / shots
        q = counts2.get(key, 0) / shots
        distance += abs(p - q)

    return 0.5 * distance


def print_sorted_counts(counts: dict, title: str) -> None:
    """
    Print counts sorted by descending frequency.
    """
    print(title)
    for bitstring, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        print(f"{bitstring}: {count}")
    print()


def is_small_enough_for_exact_check(circuit1: QuantumCircuit, circuit2: QuantumCircuit) -> bool:
    """
    Decide whether exact statevector comparison is affordable.

    This is a heuristic guard to avoid exponential blow-up.
    """
    max_qubits = max(circuit1.num_qubits, circuit2.num_qubits)
    return max_qubits <= MAX_EXACT_QUBITS


def try_exact_equivalence(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
) -> Tuple[Optional[bool], str]:
    """
    Try exact statevector comparison on the |0...0> input state.

    Returns
    -------
    tuple
        (result, message)
        - result=True   -> equivalent on |0...0>
        - result=False  -> not equivalent on |0...0>
        - result=None   -> exact check skipped or failed
    """
    if not is_small_enough_for_exact_check(circuit1, circuit2):
        return None, "Exact check skipped: circuit too large for statevector comparison."

    try:
        from qiskit.quantum_info import Statevector

        state1 = Statevector.from_instruction(circuit1)
        state2 = Statevector.from_instruction(circuit2)
        return state1.equiv(state2), "Exact statevector comparison completed."
    except Exception as exc:
        return None, f"Exact comparison failed: {exc}"


def check_equivalence(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    tolerance: float = DEFAULT_TOLERANCE,
    shots: int = DEFAULT_SHOTS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compare two circuits using exact comparison when affordable,
    otherwise use sampling with TVD.

    Returns
    -------
    dict
        {
            "status": "Identity" | "NotIdentity" | "Inconclusive",
            "tvd": float | None,
            "method": str,
            "counts1": dict | None,
            "counts2": dict | None,
            "message": str,
        }
    """
    if circuit1.num_qubits != circuit2.num_qubits:
        return {
            "status": "NotIdentity",
            "tvd": None,
            "method": "shape-check",
            "counts1": None,
            "counts2": None,
            "message": "Circuits use different numbers of qubits.",
        }

    exact_result, message = try_exact_equivalence(circuit1, circuit2)
    if verbose:
        print(message)

    if exact_result is True:
        return {
            "status": "Identity",
            "tvd": None,
            "method": "statevector",
            "counts1": None,
            "counts2": None,
            "message": message,
        }

    if exact_result is False:
        return {
            "status": "NotIdentity",
            "tvd": None,
            "method": "statevector",
            "counts1": None,
            "counts2": None,
            "message": message,
        }

    simulator = AerSimulator()

    measured1 = add_measurements(circuit1)
    measured2 = add_measurements(circuit2)

    counts1 = simulate_counts(measured1, simulator=simulator, shots=shots)
    counts2 = simulate_counts(measured2, simulator=simulator, shots=shots)

    tvd = total_variation_distance(counts1, counts2, shots)
    status = "Identity" if tvd <= tolerance else "NotIdentity"

    return {
        "status": status,
        "tvd": tvd,
        "method": "sampling",
        "counts1": counts1,
        "counts2": counts2,
        "message": "Sampling-based comparison completed.",
    }


def time_it(function, *args, **kwargs):
    """
    Measure execution time of a function call.
    """
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_single_circuit_mode(qasm_path: str) -> str:
    """
    Normalize one QASM circuit and return its textual representation.
    """
    circuit = load_qasm(qasm_path)
    normalized = canonical_form(
        circuit,
        optimization_level=1,
        decompose_circuit=False,
        reverse_bits_order=False,
    )
    return f"Canonical form:\n{normalized}"


def run_comparison_mode(qasm_path_1: str, qasm_path_2: str) -> str:
    """
    Compare two QASM circuits and return a summary.
    """
    circuit1 = load_qasm(qasm_path_1)
    circuit2 = load_qasm(qasm_path_2)

    reduced1 = canonical_form(
        circuit1,
        optimization_level=1,
        decompose_circuit=False,
        reverse_bits_order=False,
    )
    reduced2 = canonical_form(
        circuit2,
        optimization_level=1,
        decompose_circuit=False,
        reverse_bits_order=False,
    )

    result, elapsed = time_it(
        check_equivalence,
        reduced1,
        reduced2,
        tolerance=DEFAULT_TOLERANCE,
        shots=DEFAULT_SHOTS,
        verbose=True,
    )

    if result["counts1"] is not None and result["counts2"] is not None:
        print_sorted_counts(result["counts1"], "Circuit 1 counts:")
        print_sorted_counts(result["counts2"], "Circuit 2 counts:")

    status = result["status"]
    tvd = result["tvd"]
    method = result["method"]

    if status == "Identity":
        if tvd is None:
            return f"Equal (took {elapsed:.3f}s, method={method})"
        return f"Equal (took {elapsed:.3f}s, method={method}, TVD={tvd:.6f})"

    if status == "NotIdentity":
        if tvd is None:
            return f"Not equal (took {elapsed:.3f}s, method={method})"
        return f"Not equal (took {elapsed:.3f}s, method={method}, TVD={tvd:.6f})"

    return f"Inconclusive (took {elapsed:.3f}s, method={method})"


def run(args: list[str]) -> str:
    """
    Command-line entry point.
    """
    if len(args) == 1:
        return run_single_circuit_mode(args[0])

    if len(args) == 2:
        return run_comparison_mode(args[0], args[1])

    return "Usage: python compare_circuits.py circuit1.qasm [circuit2.qasm]"


if __name__ == "__main__":
    print(run(sys.argv[1:]))
