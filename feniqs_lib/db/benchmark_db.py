
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

import psycopg2
import psycopg2.extras
from feniqs_lib.db.abstract_db import Abstract_db
from datetime import datetime
import json
import logging


# logger instance for db
logger_db = logging.getLogger(__name__)


class Benchmark_db(Abstract_db):
    def __init__(self):
        """
        Initialize the BenchmarkDB class.
        """
        super().__init__()

    def create_all_tables(self):
        """
        Create all necessary tables by executing SQL file.
        """
        self.execute_file(self.FILE_PATH)

    def drop_all_tables(self):
        """
        Drop the benchmark table.
        """
        self.drop_one_table(self.TABLE_NAME)

    def query_all(self):
        """
        Retrieve all data from the benchmark table.
        """
        return self.get_all_from_table(self.TABLE_NAME)

    @staticmethod
    def _format_value(value):
        """
        Format a value for an SQL query, depending on its type.

        Args:
            value (Any): The value to format

        Returns:
            str: The formatted string for the query
        """
        if isinstance(value, str):
            # If the value is a string, add quotes around it
            return f"'{value}'"
        elif isinstance(value, datetime):
            # If the value is a datetime, format it for the query
            return f"TIMESTAMP('{value}', 'YYYY-MM-DD HH24:MI:SS')"
        elif isinstance(value, dict):
            # Convert dict to JSON string for SQL insertion
            return f"'{json.dumps(value)}'"
        # In every other case, return the string representation of the value
        return str(value)

    def create_query_on_conditions(self, select_columns=['*'], **conditions):
        """
        Create a complex query based on given conditions.

        Args:
            select_columns (list): Columns to select
            conditions (dict): Conditions for the query

        Returns:
            tuple: SQL query string and parameters
        """
        select_clause = ', '.join(select_columns)
        query = f"SELECT {select_clause} FROM {self.TABLE_NAME} WHERE "
        query_conditions = []
        params = []
        order_clause = ""
        limit_clause = "LIMIT 100"

        backends_settings = json.loads(conditions.pop('backends', '{}'))

        # Handle bench_file and nb_qubits as common parameters
        bench_file = conditions.pop('bench_file', None)
        nb_qubits = conditions.pop('nb_qubits', None)

        if bench_file:
            query_conditions.append("bench_file = %s")
            params.append(bench_file)

        if nb_qubits:
            query_conditions.append("nb_qubits = %s")
            params.append(nb_qubits)

        for key, value in conditions.items():
            if value is None:
                continue
            if key == 'time':
                exact_time_query = f"SELECT EXISTS (SELECT 1 FROM {self.TABLE_NAME} WHERE time = TO_TIMESTAMP(%s, 'YYYY-MM-DD HH24:MI:SS'))"
                self.execute_query(
                    exact_time_query, (value,), commit_changes=False)
                exists = self.cur.fetchone()[0]

                if exists:
                    query_conditions.append(
                        "time = TO_TIMESTAMP(%s, 'YYYY-MM-DD HH24:MI:SS')")
                    params.append(value)
                else:
                    order_clause = "ORDER BY ABS(EXTRACT(EPOCH FROM(time - TO_TIMESTAMP(%s, 'YYYY-MM-DD HH24:MI:SS')))) ASC"
                    params.append(value)
            elif key == 'settings':
                if isinstance(value, dict):
                    query_conditions.append(f"{key} @> %s::jsonb")
                    params.append(json.dumps(value))
            else:
                query_conditions.append(f"{key} = %s")
                params.append(value)

        # Handle backends' specific settings
        backend_conditions = []
        target_backends = []
        for backend, settings in backends_settings.items():
            if len(settings) == 0:
                print("BACKENDO : ", backend)
                target_backends.append(f" backend = '{str(backend)}'")
            else:
                for setting_key, setting_value in settings.items():
                    backend_conditions.append(
                        f"(backend = %s AND settings @> %s::jsonb)")
                    params.extend([backend, json.dumps(
                        {setting_key: setting_value})])

        if target_backends:
            query_conditions.append("(" + " OR ".join(target_backends) + ")")

 
        if backend_conditions:
            query_conditions.append(
                "(" + " OR ".join(backend_conditions) + ")")

        if query_conditions:
            query += ' AND '.join(query_conditions)
        else:
            query = query.rstrip(' WHERE ')

        if order_clause:
            query += f" {order_clause} {limit_clause}"
        else:
            query += f" ORDER BY time {limit_clause}"

        return query, params

    def insert_into_table(self, data):
        """
        Insert data into the benchmark table.

        Args:
            data (list or dict): Data to insert
        """
        if isinstance(data, list):
            self.insert_many_into_table(self.TABLE_NAME, data)
        else:
            self.insert_one_into_table(self.TABLE_NAME, data)

    def query_on_conditions(self, **conditions):
        """
        Execute a query based on conditions and return results.

        Args:
            conditions (dict): Conditions for the query

        Returns:
            list: Query results
        """
        try:
            query, params = self.create_query_on_conditions(**conditions)
            logger_db.debug(f"Executing query: {query} with params: {params}")
            return self.fetch_query(query, params)
        except psycopg2.Error as e:
            logger_db.error(
                f"Error executing query with conditions {conditions}: {e}")
            self.conn.rollback()
            raise
        except Exception as e:
            logger_db.error(f"An unexpected error occurred: {e}")
            self.conn.rollback()
            raise

    def insert_result(self, data, check_presence=True):
        """
        Insert benchmark result into the benchmark table.

        Args:
            data (dict): Data to insert
        """
        if check_presence:
            # TODO update to check settings too
            conditions = {"backend": data['backend'], "function_name": data['function_name'],
                          "nb_qubits": data['nb_qubits'], "package_version": data['package_version'],
                          "bench_file": data['bench_file'], "backend_type": data['backend_type']}
            res = self.query_on_conditions(**conditions)
            if len(res) != 0:
                # should delete here
                self.update_values(self.TABLE_NAME, data, conditions)
            else:
                self.insert_one_into_table(self.TABLE_NAME, data)
        else:
            self.insert_one_into_table(self.TABLE_NAME, data)

    def get_distinct_values(self, field):
        """
        Get all possible values for a specific field.

        Args:
            field (str): Name of the targeted field
        Returns:
            list(str): List of all the possible value
        """
        query = f"SELECT DISTINCT {field} FROM {self.TABLE_NAME}"
        self.cur.execute(query)
        return [row[0] for row in self.cur.fetchall()]

    def get_backend_distinct_values(self, backend, field):
        """
        Get all the possible values for one specific backend.
        Args:
            backend (str): The targeted backend 
            field (str): The targeted field
    
        Returns:
            list(str): list of all the possible values

        """
        query = f"SELECT DISTINCT {field} FROM {self.TABLE_NAME} WHERE backend = '{backend}'"
        self.cur.execute(query)
        return [row[0] for row in self.cur.fetchall()]

    def query_all(self):
        query = "SELECT * FROM benchmark"
        self.cur.execute(query)
        return self.cur.fetchall()
