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

from .abstract_config import SimulatorConfig
from .abstract_julia_backend import AbstractJuliaBackend
from .abstract_config import BasicConfig
from .abstract_backend import AbstractBackend
from juliacall import newmodule
from random import randint
import time
import statistics
import numpy as np

class MimiqJuliaCpuBackend(AbstractJuliaBackend):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MimiqJuliaCpuBackend, cls).__new__(cls)
        return cls._instance

    def __init__(self, device_type='mps', env=None, bond_dimension=256, entdim=8,
                 targerr=0.005, meth="mpo1z", opt_level=1, perm=False, **kwargs):
        if not hasattr(self, 'initialized'):
            kwargs['fusion_enable'] = False

            config = SimulatorConfig(
                device='Mimiq_Julia_Cpu',
                device_type=device_type,
                package_version='MAGIC_VERSION_NUMBER',
                **kwargs
            )

            super().__init__(julia_module_name="mimiq_julia", config=config)
            self.env = "feniqs_lib/backends/plugins/venv/.mimiq_julia_cpu_env"
            self.add_config_attr("bond_dimension", bond_dimension)
            self.add_config_attr("entdim", entdim)
            self.add_config_attr("targerr", targerr)
            self.add_config_attr("meth", meth)
            self.add_config_attr("opt_level", opt_level)
            self.add_config_attr("perm", perm)

            self.import_packages()
            self.define_julia_functions()

            if self.config.device_type == 'statevector':
                self.config.package_version = self._get_julia_package_version("StateVecSim")
            elif self.config.device_type == 'mps':
                self.config.package_version = self._get_julia_package_version("MPSSim")

            self.separates_execution_and_sampling = False
            self.generate_backend()
            self.initialized = True

    def import_packages(self):
        # Import required Julia modules, including PythonCall
        self.julia_module.seval("using MimiqCircuitsBase, QASMParsers, MPSSim, Random, PythonCall")
        if self.config.device_type == 'mps':
            self.julia_module.seval("using MPSSim")
        elif self.config.device_type == 'statevector':
            self.julia_module.seval("using StateVecSim")

    def generate_backend(self):
        if self.config.device_type == 'statevector':
            self.time_value = self.julia_module.seval("@elapsed qcs = StateVecQCS()")
        elif self.config.device_type == 'mps':
            options = ''
            if self.config.bond_dimension or self.config.entdim:
                options = ';'
                if self.config.bond_dimension:
                    options += f'bonddim={self.config.bond_dimension},'
                if self.config.entdim:
                    options += f'entdim={self.config.entdim},'
                options = options[:-1]
            self.time_value = self.julia_module.seval(f"@elapsed qcs = MPSSimulator({options})")

    def define_julia_functions(self):
        # Define Julia functions and ensure the global backend is defined.
        self.julia_module.seval("""
using QASMParsers
using MimiqCircuitsBase
using MPSSim
using Random
using PythonCall

# Define default parameters and global backend
global DEFAULT_PARAMS = Dict(
    :bonddim => 256,
    :entdim  => 16,
    :nshots  => 1000,
    :targerr => 0.0000000001,
    :meth    => "mpo1z",
    :opt_level => 1,
    :iterations => nothing,
    :max_iter   => 1e10,
    :perm       => false
)
global backend = DEFAULT_PARAMS

stringinterpret(str) = QASMParsers.Interpreters.interpret(QASMParsers.Parsers.parseopenqasm(str))

function import_qasm1(qasmfile::String)
    qasmstr = read(qasmfile, String)
    return stringinterpret(qasmstr)
end

include("feniqs_lib/backends/quantum_backends/optimizer.jl")

function compile(circ::MimiqCircuitsBase.Circuit; optimize::Bool=false, opt_level, reorderqubits::Bool=false, perm::Vector{Int}=Int[], kwargs...)
    if optimize
        if opt_level == 1
            circ = compress(circ)
        end
        if reorderqubits
            if isempty(perm)
		println("ODERING")
                params = merge(backend, kwargs)
                circ, perm, dist = optimize_ordering(circ; params...)
                return circ, perm
            else
                return reorder_qubits(circ, perm), perm
            end
        end
    end
    return circ, [1:length(circ)...]
end

function execute1(bonddim, entdim, targerr, meth, circ::MimiqCircuitsBase.Circuit; state=nothing)
    sim = MPSSimulator(; bonddim=bonddim, entdim=entdim)
    println("MIMIQ")
    nqubits = MimiqCircuitsBase.numqubits(circ)
    if isnothing(state)
        state = zerostate(sim, nqubits, nqubits, 0)
    end
    ccirc = convertcircuit(sim, circ)
    _, fid = evolve!(state, ccirc; targerr=targerr, meth=meth)
    return state, fid
end

function execute_and_sample1(bonddim::Int, entdim::Int, targerr::Float64, meth::String,
                               circ::MimiqCircuitsBase.Circuit, nshots::Int; state=nothing)
    sim = MPSSimulator(; bonddim=bonddim, entdim=entdim)
    nqubits = MimiqCircuitsBase.numqubits(circ)
    println("Start compile")
    circ, perm = compile(circ; opt_level=1)
    println("End compile")
    if isnothing(state)
        state = zerostate(sim, nqubits, nqubits, 0)
    end
    ccirc = convertcircuit(sim, circ)
    _, fid = evolve!(state, ccirc; targerr=targerr, meth=Symbol(meth))

    samples = AbstractQCSs.sample(state.q, Random.GLOBAL_RNG, nshots)
    return samples, fid
end

function get_mirror_fidelity(qasm_file::String, mirror_qasm_file::String, bonddim::Int,
                             entdim::Int, targerr::Float64, meth::String)
    qc = import_qasm1(qasm_file)
    qc_mirror = import_qasm1(mirror_qasm_file)
    qc, perm = compile(qc; opt_level=1)
    qc_mirror, perm_mirror = compile(qc_mirror; opt_level=1)
    state, fid1 = execute1(bonddim, entdim, targerr, meth, qc)
    state, fid2 = execute1(bonddim, entdim, targerr, meth, qc_mirror, state=state)
    nqubits = MimiqCircuitsBase.numqubits(qc)
    zero_prob = abs2(amplitude(state.q, BitString(nqubits, 0)))
    return zero_prob
end
        """)

    def parse(self):
#        self.time_value = self.julia_module.seval(
#            "@elapsed circuit = QASMParsers.Interpreters.interpret(QASMParsers.Parsers.parseopenqasm(qasm_str))"
#        )
       start = time.time()
       # Import the QASM file and compile the circuit
       circuit = self.julia_module.import_qasm1(self._qasm_file)
       circuit, perm = self.julia_module.compile(circuit, opt_level=1, reorderqubits=False, optimize=True)
       self.compiled_circuit = circuit
       end = time.time()
       self.time_value = end - start
       
       # Optionally, return the circuit if needed:
       return circuit
   
    @AbstractBackend._measure_time
    def format_sample(self, samples):
        # Customize sample formatting as needed.
        res = {}
        return res

    def execute_and_sample(self):
        start = time.time()
        #qc = self.julia_module.import_qasm1(self._qasm_file)
        #qc, perm = self.julia_module.compile(qc, opt_level=1, reorderqubits=False, optimize=True)
        samples, fid = self.julia_module.execute_and_sample1(
            384,      # bond_dimension
            4,       # entdim
            0.0000000001,   # targerr
            "zipup",  # meth
            self.compiled_circuit, #qc,
            1000
        )
        end = time.time()
        self.time_value = end - start
        return samples, fid

    # def get_mirror_fidelity(self, qasm_file, mirror_qasm_file):
    #     zero_prob = self.julia_module.get_mirror_fidelity(
    #         qasm_file,
    #         mirror_qasm_file,
    #         64,   # bond_dimension
    #         4,     # entdim
    #         0.01, # targerr
    #         "mpo1z" # meth
            
    #     )
    #     return zero_prob

    def get_precision(self):
        return "double"

