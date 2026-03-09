
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

#!/usr/bin/env python3
"""
Create a "mirror" (inverse + reversed order) quantum circuit from an input circuit.

Usage examples:
  python mirror_circuit.py --in input.qasm --out mirrored.qasm
  python mirror_circuit.py --in input.qasm --out mirrored.qasm --reverse-qubits

Requires: qiskit
  pip install qiskit
"""

import argparse
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps

def mirror_circuit(circ: QuantumCircuit, reverse_qubits: bool = False) -> QuantumCircuit:
    """
    Build a mirrored circuit:
      - iterate instructions in reverse
      - append inverse of each operation (when possible)
      - optionally reverse qubit order mapping: q -> (n-1-q)

    Notes:
      - barrier is kept (mirrored position)
      - measure is kept, but if reverse_qubits=True, measurement mapping follows qubit remap
      - reset is not unitary; we keep it as is 
    """
    n_q = circ.num_qubits
    n_c = circ.num_clbits
    out = QuantumCircuit(n_q, n_c, name=f"mirror_of_{circ.name or 'circuit'}")

    # Build qubit remap if requested
    if reverse_qubits:
        q_map = {circ.qubits[i]: out.qubits[n_q - 1 - i] for i in range(n_q)}
    else:
        q_map = {circ.qubits[i]: out.qubits[i] for i in range(n_q)}

    c_map = {circ.clbits[i]: out.clbits[i] for i in range(n_c)}

    # Traverse in reverse order
    for inst, qargs, cargs in reversed(circ.data):
        # Map qubits/clbits to output circuit
        new_qargs = [q_map[q] for q in qargs]
        new_cargs = [c_map[c] for c in cargs]

        name = inst.name.lower()

        # Keep barriers (no inverse needed)
        if name == "barrier":
            out.barrier(*new_qargs)
            continue

        # Measurements: keep them 
        if name == "measure":
            out.measure(new_qargs[0], new_cargs[0])
            continue

        # Reset: not invertible; keep as is 
        if name == "reset":
            out.reset(new_qargs[0] if len(new_qargs) == 1 else new_qargs)
            continue

        # For everything else, try inverse()
        try:
            inv = inst.inverse()
        except Exception:
            # Fallback: if inverse isn't available, try to keep gate as is
            inv = inst

        out.append(inv, new_qargs, new_cargs)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input OpenQASM 2 file")
    ap.add_argument("--out", dest="out_path", required=True, help="Output OpenQASM 2 file")
    ap.add_argument("--reverse-qubits", action="store_true", help="Reverse qubit order in the mirrored circuit")
    args = ap.parse_args()

    circ = QuantumCircuit.from_qasm_file(args.in_path)
    mirrored = mirror_circuit(circ, reverse_qubits=args.reverse_qubits)

    # Write QASM (OpenQASM 2)
    qasm_str = dumps(mirrored)
    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write(qasm_str)

    print(f"Wrote mirrored circuit to: {args.out_path}")
    print(f"Original depth: {circ.depth()}, Mirrored depth: {mirrored.depth()}")


if __name__ == "__main__":
    main()

