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

import re
import math
from functools import reduce
from operator import mul


class QasmParser():
    """
    A simple implementation of Open QASM 2.0 parser for quickly parsing input file. 
    This implementation focuses on minimizing computational overhead and quickly streamlining the parsing process.
    """
    _command_split_regex = re.compile(
        r"\s+(?=[^\)]*(?:\(|$))")  # This regex splits on the first whitespace unless it's within parentheses.
        
    # Gate mappings to minimize lookup times and maintain clarity.
    _Maped_Gates = {
        # Basic gates
        "h": "H", "x": "X", "y": "Y", "z": "Z",
        "rx": "RX", "ry": "RY", "rz": "RZ",
        "s": "S", "t": "T", "p": "P", "cx": "CNOT",
        "tdg": "TDG", "sdg": "SDG", 'sx': 'SQRTX', 'sxdg': 'SQRTXDG',
        # Parameterized and multi-qubit gates
        "u1": "U1", "u2": "U2", "u3": "U3",
        "swap": "SWAP", "cz": "CZ",
        "crx": "CRX", "cry": "CRY", "crz": "CRZ",
        "cu1": "CU1", "cu3": "CU3", "rzz": "RZZ",
        "ccx": "TOFFOLI", "id": "I"
    }

    # Set of gates that require parameters to reduce conditional checks
    _Param_Gates = {
        "rx", "ry", "rz", "u1", "u2", "u3",
        "crx", "cry", "crz", "cu1", "cu3", "rzz"
    }

    def __init__(self) -> None:
        self._qubits, self._creg, self._gate_list = {}, {}, []

    @staticmethod
    def check_if_measurements_present(qasm_code):
        """
        Checks if the OpenQASM file contains measurement operations. This is useful for
        determining whether the simulation backend should allocate memory for classical registers.
        """
        return "measure" in qasm_code

    @staticmethod
    def get_register(qasm_code):
        """
        Extracts the quantum and classical register names and sizes from the OpenQASM file.
        This is useful for initializing the simulation backend with the correct register sizes.
        """
        qreg = re.findall(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]", qasm_code)
        creg = re.findall(r"creg\s+(\w+)\s*\[\s*(\d+)\s*\]", qasm_code)
        # print(qreg, creg)
        return qreg, creg

    def parse_qasm(self, qasm_code):
        """
        Parses the provided OpenQASM file into a structured format suitable for simulation backends.
        This involves separating the code into individual operations, handling different types of
        registers, and translating QASM gates to their corresponding internal representations.
        """
        lines = self._prepare_lines(qasm_code)
        self._qubits, self._creg, self._gate_list = {}, {}, []
        for line in lines:
            command, args = self._split_command_args(line)
            # Direct routing to appropriate handlers based on command type
            if command in ['qreg', 'creg']:
                self._handle_qreg_creg(command, args)
            elif command == 'measure':
                self._handle_measure(args)
            else:
                self._handle_gate(command, args)

        return len(self._qubits), self._gate_list

    def _prepare_lines(self, qasm_code):
        """
        Prepares and cleans the OpenQASM linescregs_size for parsing. This includes removing comments,
        ensuring compliance with the QASM version, and filtering out non-instruction lines.
        """
        lines = [line.split("//")[0].strip() for line in qasm_code.split("\n")]
        lines = list(filter(None, lines))

        if lines[0] != "OPENQASM 2.0;":
            raise ValueError(
                "OpenQASM file must be started with 'OPENQASM 2.0'.")
        if "include" not in lines[1]:
            raise ValueError(
                "Standard library in not included in OpenQQASm file.")

        return lines[2:]  # Skip the QASM version and include lines

    def _split_command_args(self, line):
        """
        Splits a line of OpenQASM file into its command and argument components.
        Uses precompiled regex to split efficiently on the first whitespace that is not within parentheses.
        """
        parts = self._command_split_regex.split(line, 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    def _get_args(self, args):
        """
        Extracts qubit and/or classical register arguments from a QASM instruction.
        This function uses regex for parsing of qubit/register indices.
        """
        for name, index in re.findall(r"(\w+)\[(\d+)\]", args):
            yield name, int(index)

    def _handle_qreg_creg(self, command, args):
        """
        Handles declaration lines for quantum and classical registers, mapping each qubit
        and classical bit to a unique identifier for later reference.
        """
        for name, size in self._get_args(args):
            if command == 'qreg':
                self._qubits.update(
                    {(name, i): len(self._qubits) + i for i in range(size)})
            if command == 'creg':
                self._creg[name] = size

    def _handle_gate(self, command, args):
        """
        Parses gate operations, extracting gate names, parameters, and target qubits. This method
        supports both parameterized and non-parameterized gates, translating QASM to internal
        gate representations.
        """
        # Splitting gate name from parameters if present, using regex (for efficiency)
        match = re.match(r"(\w+)(?:\((.*?)\))?", command)
        if not match:
            raise ValueError(
                f"Wrong gate representation in the file: {command}")

        gatename, param_str = match.groups()
        params = self._parse_params(
            param_str) if gatename in self._Param_Gates else None

        # Resolving qubits and adding to the gate list
        qubit_indices = [self._qubits[qubit] for qubit in self._get_args(args)]
        self._gate_list.append(
            (self._Maped_Gates[gatename], qubit_indices, params))

    def _parse_params(self, param_str):
        """
        Parses and evaluates parameter expressions for gates, converting symbolic representations
        (e.g., pi) to numerical values and handling mathematical operations.
        """
        if param_str is None:
            return None
        # Splitting parameters and evaluating expressions involving 'pi'
        return [eval(param.replace('pi', str(math.pi))) for param in param_str.split(',')]

    def _handle_measure(self, measure_args):
        """
        Parses measurement operations, extracting quantum and classical registers.
        """
        measure_args = measure_args.split("->")
        if len(measure_args) != 2:
            raise ValueError(
                f"Measurement instruction is incorrect: {measure_args}")

        q = next(self._get_args(measure_args[0]))
        if q not in self._qubits:
            raise ValueError(
                f"Qubit {q} must be defined for Measurement operation in OpenQASM file.")

        r, i = next(self._get_args(measure_args[1]))
        if r not in self._creg:
            raise ValueError(f"Name of classical register {r} must be defined for Measurement operation"
                             "in OpenQASM file.")
        if i >= self._creg[r]:
            raise ValueError(
                f"Index {i} of element in classical register {r} is not correct")
        self._gate_list.append(("M", [self._qubits[q], i], [q[0], r]))
