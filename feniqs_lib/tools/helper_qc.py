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


from qiskit import QuantumCircuit, qasm2
import os
import logging

class QuantumCircuitHelper:
    def __init__(self, qasm_file):
        """
        Initialize the QuantumCircuitHelper with a QASM file.
        param qasm_file: Path to the QASM file.
        """
        self.logger = logging.getLogger("feniqs")
        self.qasm_file = qasm_file
        self._load_original_circuit()

    def _load_original_circuit(self):
        """
        Load the original circuit from the QASM file.
        """
        if not os.path.exists(self.qasm_file):
            self.logger.error(f"QASM file '{self.qasm_file}' does not exist.")
            raise FileNotFoundError(f"QASM file '{self.qasm_file}' does not exist.")
        self.original_circuit = QuantumCircuit.from_qasm_file(self.qasm_file)
        self.logger.info(f"Original circuit loaded from '{self.qasm_file}'.")


    def create_check_circuit(self):
        """
    	Create a combined circuit with the original and its inverse with a barrier.
    	Measurements at the end of the circuit are moved to the end of the combined circuit.
    	return: Combined QuantumCircuit.
    	"""
        try:
            combined_circuit = QuantumCircuit(self.original_circuit.num_qubits)

            # Extract measurement operations and their positions
            measurement_ops = []
            for index, instr in enumerate(self.original_circuit.data):
                if instr[0].name == 'measure':
                    measurement_ops.append((index, instr))
        

            last_gate_index = len(self.original_circuit.data) - 1
            while last_gate_index >= 0 and self.original_circuit.data[last_gate_index][0].name == 'measure':
                self.original_circuit.data.pop(last_gate_index)
                last_gate_index -= 1
            
            inverse_circuit = self.original_circuit.inverse()

            # Combine the original circuit, barrier, and inverse circuit
            combined_circuit.compose(self.original_circuit, inplace=True)
            combined_circuit.barrier()
            combined_circuit.compose(inverse_circuit, inplace=True)
        
            # Add the measurements back at the end of the combined circuit
            for _,instr in measurement_ops:
               combined_circuit.append(instr)
        
            self.logger.info("Combined circuit with original and inverse created successfully.")
            return combined_circuit
        except Exception as e:
            self.logger.error(f"Error creating combined circuit: {e}")
            raise

    def save_circuit(self, circuit, filename):
        """
        Save the given circuit to a QASM file.
        param circuit: QuantumCircuit to be saved.
        param filename: Name of the QASM file to save.
        """
        try:
            qasm2.dump(circuit, filename)    
        except Exception as e:
            self.logger.error(f"Error saving circuit to '{filename}': {e}")
            raise


    def get_circuit_stats(self, circuit):
        """
        Get statistics of the given circuit.
        Args:
            circuit: QuantumCircuit to analyze.
        Returns:
            dict: Dictionary containing circuit statistics.
        """
        try:
            stats = {
                'width': circuit.width(),
                'depth': circuit.depth(),
                'num_qubits': circuit.num_qubits,
                'num_gates': circuit.size(),
                'num_nonlocal_gates': circuit.num_nonlocal_gates(),
                'num_operations': len(circuit)
            }
            self.logger.info(f"Circuit statistics: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"Error getting circuit statistics: {e}")
            raise

def process_circuit(qasm_file, output_file):
    helper = QuantumCircuitHelper(qasm_file)
    combined_circuit = helper.create_check_circuit()
    helper.save_circuit(combined_circuit, output_file)

# Example usage
process_circuit("path/to/your/qasm_file.qasm", "combined_circuit.qasm")

# Example usage
if __name__ == "__main__":

    qasm_file = 'qft.qasm'
    helper = QuantumCircuitHelper(qasm_file)
    
    combined_circuit = helper.create_check_circuit()
    helper.save_circuit(combined_circuit, "combined_circuit.qasm")
    
    stats = helper.get_circuit_stats(combined_circuit)
    print(stats)

