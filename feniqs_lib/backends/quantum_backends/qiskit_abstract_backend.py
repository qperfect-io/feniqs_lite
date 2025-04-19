
from __future__ import annotations

try:
    import qiskit
except ImportError:
    qiskit = None
from qiskit import QuantumCircuit, transpile
from qiskit import qasm2
from abc import ABCMeta, abstractmethod
from .abstract_backend import AbstractBackend
from .abstract_config import BasicConfig
from qiskit_aer import AerSimulator

class QiskitAbstractBackend(AbstractBackend, metaclass=ABCMeta):

    # Need to implement constructor
    @abstractmethod
    def __init__(self, config: BasicConfig):
        """
        Instantiates the Qiskit backend Handler.

        Args:
            config (BasicConfig): The configuration of the backend
        """
        if qiskit is None:
            raise RuntimeError("qiskit is not installed correctly in its venv")
   
        super().__init__(config)

    @AbstractBackend._measure_time
    def parse(self):
        """
        Parses the loaded QASM string into a Qiskit quantum circuit.\n
      
        Results stored in self._qiskit_circuit.
        """
        if self._qasm_str is None:
            raise ValueError(
                "Error: No QASM loaded. Please load a QASM file first.")
        self._qiskit_circuit = qasm2.loads(
            self._qasm_str, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        # updates the number of qubits if first naive check was wrong
        self.config.nb_qubits = self._qiskit_circuit.num_qubits

        # Check if measurements are present in the circuit
        ops = self._qiskit_circuit.count_ops()
        if 'measure' not in ops:
            self._qiskit_circuit.measure_all()

    def plot_circuit(self):
        print(self._qiskit_circuit)

    def _compile ( self , circuit, opt_level) :
        return transpile ( circuit, optimization_level = opt_level)

    def _import_qasm ( self , qasm_file ) :
        with open ( qasm_file , 'r') as f :
            qasm_str = f.read()
        return QuantumCircuit.from_qasm_str(qasm_str)

    def get_mirror_fidelity(self, qasm_file, mirror_qasm_file):
        qc = self._compile(self._import_qasm(qasm_file), self.config.opt_level)
        qc.remove_final_measurements(inplace=True)
        qc_mirror = self._compile(self._import_qasm(mirror_qasm_file), self.config.opt_level)
        qc_mirror.remove_final_measurements(inplace=True)
        qc.barrier()
        qc = qc.compose(qc_mirror)
        self._qiskit_circuit = qc
        res = self.execute_and_sample()    
        counts = res.get_counts()
        initial_state = '0' * qc.num_qubits
        return counts.get(initial_state, 0)/self.config.nb_shots

    @AbstractBackend._measure_time
    def format_sample(self, samples):
        """
        Formats the result of a Qiskit simulation into a dictionary of counts.
        Args:
            sample (qiskit.Result): The raw samples from the Qiskit simulation. 

        Returns:
            dict: The dictionary of counts
        """
        # For some reson qiskit get all the results in the wrong order
        samples = samples.get_counts()
        res = {}
        for key, val in samples.items():
            key = key.split(' ', 1)[0]  # in case creg were measured
            res[key[::-1]] = val
        return res

    @AbstractBackend._measure_time
    def sample_only(self):
        """
        Samples measurement outcomes from the quantum circuit.
        """
        # !IMPORTANT: the execution and sampling cannot be separated for qiskit
        # This method is in place simply to keep the structure of the code consistent
        results = self._job.result()

        # if self.config.device_type == 'matrix_product_state':
        #     self.fidelity = 1 - \
        #         results.results[0].metadata['matrix_product_state_truncation_threshold']  
        self.config.device_type = str(results.results[0].metadata['method'])
        return results

    @AbstractBackend._measure_time
    def execute_only(self):
        """
        Executes the quantum circuit on the backend.\n
        Results stored in self._job.
        """

        self._qiskit_circuit = transpile(self._qiskit_circuit, optimization_level = self.config.opt_level)

        self._qiskit_circuit.measure_all()

        self._job = self._backend.run(self._qiskit_circuit, shots=self.config.nb_shots)        

      

    @AbstractBackend._measure_time
    def execute_and_sample(self):
        """
        Executes and samples the circuit on the backend.
        Is the default way to execute and sample on the benchmark as some simulator do not allow for differentiations between the two.
        """
        self.execute_only()
        return self.sample_only()

    def get_precision(self):
        return self.config.precision

