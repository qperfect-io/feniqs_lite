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


from datetime import datetime
from feniqs.db.benchmark_db import Benchmark_db
import csv
import json


def test_insert_and_query():
    try:
        db = Benchmark_db()

        # Data to be inserted
        data_to_insert = [
            {
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'backend': 'qiskit',
                'version_backend': '0.23.0',
                'backend_type': 'statevector',
                'bench_file': 'example_circuit_1.qasm',
                'function_name': 'execution',
                'nb_qubits': 5,
                'exception': None,
                'settings': json.dumps({'fusion': True}),
                'metrics': json.dumps({'execution_time': 0.0002})
            },
            {
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'backend': 'qiskit',
                'version_backend': '0.23.1',
                'backend_type': 'statevector',
                'bench_file': 'example_circuit_2.qasm',
                'function_name': 'parsing',
                'nb_qubits': 7,
                'exception': None,
                'settings': json.dumps({'fusion': False}),
                'metrics': json.dumps({'parsing_time': 0.0003})
            },
            {
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'backend': 'cirq',
                'version_backend': '0.9.1',
                'backend_type': 'mps',
                'bench_file': 'example_circuit_3.qasm',
                'function_name': 'execution',
                'nb_qubits': 6,
                'exception': 'RuntimeError',
                'settings': json.dumps({'fusion': True}),
                'metrics': json.dumps({'execution_time': 0.0004})
            }
        ]

        # Insert data into the table
        db.insert_many_into_table(db.TABLE_NAME, data_to_insert)

        # Perform a complex query
        result = db.query_on_conditions(
            backend='qiskit',
            function_name='execution',
            metrics={'execution_time': 0.0002}
        )

        # Print out the results to console
        print("Query Results:")
        for row in result:
            print(row)


        # Save the results to a CSV file
        with open('query_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time', 'benchmark_id', 'backend', 'version_backend', 'backend_type', 'bench_file', 'function_name', 'nb_qubits', 'exception', 'settings', 'metrics'])
            for row in result:
                writer.writerow(row)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure the database connection is closed
        db.close_connection()

# Run tests
if __name__ == "__main__":
    test_insert_and_query()


