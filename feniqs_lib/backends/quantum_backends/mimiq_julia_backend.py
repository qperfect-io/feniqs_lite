


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

from __future__ import annotations

import os
import threading
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional

import mimiqcircuits
from mimiqcircuits import BitString
from mimiqcircuits.qasm import load as load_qasm

from .abstract_backend import AbstractBackend
from .abstract_config import SimulatorConfig
from .abstract_simulator_backend import AbstractSimulatorBackend


_MIMIQ_IMPORTS = None
_MIMIQ_LOCK = threading.Lock()


def _get_mimiq_imports():
    """
    Lazy import of mimiqengines so the module is not imported at file load time.
    This helps keep Julia initialization deferred until the backend is actually created.
    """
    global _MIMIQ_IMPORTS

    if _MIMIQ_IMPORTS is not None:
        return _MIMIQ_IMPORTS

    with _MIMIQ_LOCK:
        if _MIMIQ_IMPORTS is None:
            from mimiqengines import MPSSimulator, StateVecSimulator
            _MIMIQ_IMPORTS = (MPSSimulator, StateVecSimulator)

    return _MIMIQ_IMPORTS


_METHOD_MAP = {
    "mpo1z": "MPO1z",
    "mpo2z": "MPO2z",
    "zipup": "ZipUp",
}

_DEVICE_MAP = {
    "mps": "mps",
    "matrix_product_state": "mps",
    "matrixproductstate": "mps",
    "statevector": "statevector",
    "state_vector": "statevector",
    "state-vector": "statevector",
    "sv": "statevector",
}


class MimiqJuliaCpuBackend(AbstractSimulatorBackend):
    def __init__(
        self,
        device_type: str = "mps",
        env: Optional[str] = None,
        bond_dimension: int = 256,
        entdim: int = 16,
        targerr: float = 1e-10,
        meth: str = "mpo1z",
        opt_level: int = 1,
        perm: bool = False,
        traversal: str = "Sequential",
        **kwargs,
    ):
        kwargs["fusion_enable"] = False

        config = SimulatorConfig(
            device="Mimiq_Julia_Cpu",
            device_type=device_type,
            package_version=self._safe_package_version("mimiqengines"),
            **kwargs,
        )
        super().__init__(config)

        self.env = env

        self.add_config_attr("bond_dimension", int(bond_dimension))
        self.add_config_attr("entdim", int(entdim))
        self.add_config_attr("targerr", float(targerr))
        self.add_config_attr("meth", str(meth))
        self.add_config_attr("opt_level", int(opt_level))
        self.add_config_attr("perm", bool(perm))
        self.add_config_attr("traversal", str(traversal))
        self.add_config_attr("runtime", "mimiqengines-python")
        self.add_config_attr("mimiqcircuits_version", self._safe_package_version("mimiqcircuits"))

        self._backend = None
        self._circuit = None
        self._has_measurements = False
        self._results = None
        self._last_samples = None
        self._last_state = None
        self._parsed_qasm_file = None

        self.separates_execution_and_sampling = False
        self.generate_backend()

    @staticmethod
    def _safe_package_version(package_name: str) -> str:
        try:
            return version(package_name)
        except PackageNotFoundError:
            return "unknown_version"

    @staticmethod
    def _normalize_device_type(device_type: str) -> str:
        key = str(device_type).strip().lower().replace(" ", "_")
        return _DEVICE_MAP.get(key, key)

    @staticmethod
    def _normalize_method(method: str) -> str:
        key = str(method).strip().lower()
        if key not in _METHOD_MAP:
            valid = ", ".join(sorted(_METHOD_MAP))
            raise ValueError(f"Unsupported MIMIQ method '{method}'. Expected one of: {valid}.")
        return _METHOD_MAP[key]

    @staticmethod
    def _get_num_qubits(circuit) -> int:
        for attr_name in ("num_qubits", "numqubits"):
            attr = getattr(circuit, attr_name, None)
            if callable(attr):
                return int(attr())
        raise AttributeError("Unable to determine the number of qubits for the MIMIQ circuit.")

    @staticmethod
    def _extract_fidelity(results: Any) -> Optional[float]:
        fidelities = getattr(results, "fidelities", None)
        if not fidelities:
            return None
        try:
            return float(fidelities[0])
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _histogram_to_counts(histogram: Dict[Any, Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for key, value in histogram.items():
            if hasattr(key, "to01"):
                bitstring = key.to01()
            else:
                bitstring = str(key)
            counts[bitstring] = int(value)
        return counts

    def _build_sampling_counts_from_state(self, state) -> Dict[str, int]:
        samples = state.sample(self.config.nb_shots, seed=self.config.seed)
        return dict(Counter(sample.to01() for sample in samples))

    @AbstractBackend._measure_time
    def generate_backend(self):
        normalized_device = self._normalize_device_type(self.config.device_type)

        MPSSimulator, StateVecSimulator = _get_mimiq_imports()

        if normalized_device == "statevector":
            self._backend = StateVecSimulator()
        elif normalized_device == "mps":
            self._backend = MPSSimulator(
                bonddim=int(self.config.bond_dimension),
                entdim=int(self.config.entdim),
                scut=float(self.config.targerr),
                method=self._normalize_method(self.config.meth),
                traversal=str(getattr(self.config, "traversal", "Sequential")),
            )
        else:
            raise ValueError(
                f"Unsupported Mimiq device_type '{self.config.device_type}'. "
                "Use 'mps' or 'statevector'."
            )

        return self._backend

    def reconfigure(self, **kwargs):
        """
        Reuse the same Python/Julia session and rebuild only the low-level engine
        when simulator parameters actually changed.
        """
        regenerate = False

        if "nb_shots" in kwargs:
            self.config.nb_shots = int(kwargs["nb_shots"])

        if "seed" in kwargs and kwargs["seed"] is not None:
            self.config.seed = int(kwargs["seed"])

        fields = {
            "device_type": str,
            "bond_dimension": int,
            "entdim": int,
            "targerr": float,
            "meth": str,
            "opt_level": int,
            "perm": bool,
            "traversal": str,
        }

        for name, cast in fields.items():
            if name not in kwargs:
                continue
            new_value = cast(kwargs[name])
            old_value = getattr(self.config, name, None)
            if old_value != new_value:
                setattr(self.config, name, new_value)
                if name in {"device_type", "bond_dimension", "entdim", "targerr", "meth", "traversal"}:
                    regenerate = True

        if regenerate:
            self.generate_backend()

        return self

    @AbstractBackend._measure_time
    def parse(self):
        if self._circuit is not None and self._parsed_qasm_file == self._qasm_file:
            return self._circuit

        includedirs = [os.path.dirname(os.path.abspath(self._qasm_file))]
        self._circuit = load_qasm(self._qasm_file, includedirs=includedirs)
        self._parsed_qasm_file = self._qasm_file
        self.config.nb_qubits = self._get_num_qubits(self._circuit)
        self._has_measurements = "measure" in self._qasm_str.lower()
        return self._circuit

    @AbstractSimulatorBackend._measure_time
    def execute_only(self):
        if self._circuit is None:
            raise ValueError("No MIMIQ circuit available. Call parse() before execute_only().")

        if self._has_measurements:
            self._results = self._backend.execute(
                self._circuit,
                self.config.nb_shots,
                seed=self.config.seed,
            )
            self.fidelity = self._extract_fidelity(self._results)
            self._last_samples = self._results.histogram()
            self._last_state = None
        else:
            state, fidelity = self._backend.evolvezerostate(
                self._circuit,
                seed=self.config.seed,
            )
            self._last_state = state
            self.fidelity = float(fidelity) if fidelity is not None else None
            self._last_samples = self._build_sampling_counts_from_state(state)
            self._results = None

        return self._last_samples

    @AbstractSimulatorBackend._measure_time
    def sample_only(self):
        if self._last_samples is None:
            raise ValueError("No samples available. Call execute_only() first.")
        return self._last_samples

    @AbstractSimulatorBackend._measure_time
    def execute_and_sample(self):
        self.execute_only()
        return self.sample_only()

    @AbstractBackend._measure_time
    def format_sample(self, samples):
        if hasattr(samples, "histogram"):
            histogram = samples.histogram()
            counts = self._histogram_to_counts(histogram)
        elif isinstance(samples, dict):
            counts = self._histogram_to_counts(samples)
        else:
            counts = dict(Counter(str(sample) for sample in samples))

        return counts

    def get_mirror_fidelity(self, qasm_file, mirror_qasm_file):
        circuit = load_qasm(qasm_file, includedirs=[os.path.dirname(os.path.abspath(qasm_file))])
        mirror_circuit = load_qasm(
            mirror_qasm_file,
            includedirs=[os.path.dirname(os.path.abspath(mirror_qasm_file))],
        )

        state, _ = self._backend.evolvezerostate(circuit, seed=self.config.seed)
        state, _ = self._backend.evolve(state, mirror_circuit, seed=self.config.seed)

        nq = self._get_num_qubits(circuit)
        zero_prob = abs(state.amplitude(BitString("0" * nq))) ** 2
        return float(zero_prob)

    def get_precision(self):
        return "double"