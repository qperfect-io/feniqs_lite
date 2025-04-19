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
import argparse
import json
from psycopg2.extras import RealDictCursor

# Database connection parameters
DB_PARAMS = {
    "dbname": "feniqs",
    "user": "feniqs_user",
    "password": "nopassword",
    "host": "localhost"
}

def recreate_table_from_file(sql_file):
    """
    @brief Recreates a database table from an SQL file.
    @param sql_file Path to the SQL file containing the table creation statements.
    @details This function reads an SQL file, appends a drop table statement to ensure the table is recreated, and executes the SQL to create a new table.
    """
    try:
        with open(sql_file, 'r') as file:
            create_table_query = file.read()
        
        # Append DROP TABLE statement to ensure the table is recreated
        create_table_query = "DROP TABLE IF EXISTS public.benchmark CASCADE;\n" + create_table_query
        
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        cur.execute(create_table_query)
        conn.commit()

        cur.close()
        conn.close()
        print("Table recreated successfully")
    except Exception as e:
        print(f"An error occurred: {e}")

def insert_benchmark(config_file):
    """
    @brief Inserts benchmark data into the database from a configuration file.
    @param config_file Path to the JSON configuration file containing benchmark data.
    @details This function reads a JSON configuration file, processes the data to ensure it fits the expected schema, and inserts it into the database.
    """
    try:
        with open(config_file, 'r') as file:
            data = json.load(file)
        
        # Check and process keys
        if 'package_version' not in data and 'version_backend' in data:
            data['package_version'] = data.pop('version_backend')
        elif 'package_version' in data and 'version_backend' not in data:
            data['version_backend'] = data['package_version']
        
        # Convert JSON strings to objects
        if isinstance(data['settings'], str):
            data['settings'] = json.loads(data['settings'])
        data['settings'] = json.dumps(data['settings'])
        
        if isinstance(data['metrics'], str):
            data['metrics'] = json.loads(data['metrics'])
        data['metrics'] = json.dumps(data['metrics'])

        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        query = """
        INSERT INTO public.benchmark (
            "time",
            backend,
            package_version,
            backend_type,
            bench_file,
            function_name,
            nb_qubits,
            exception,
            settings,
            metrics,
            benchmark_group
        ) VALUES (
            %(time)s,
            %(backend)s,
            %(package_version)s,
            %(backend_type)s,
            %(bench_file)s,
            %(function_name)s,
            %(nb_qubits)s,
            %(exception)s,
            %(settings)s::jsonb,
            %(metrics)s::jsonb,
            %(benchmark_group)s
        );
        """
        
        cur.execute(query, data)
        conn.commit()

        cur.close()
        conn.close()
        print("Data inserted successfully")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def cleanup_db():
    """
    @brief Cleans up the database by deleting all records in the benchmark table.
    @details This function deletes all records from the public.benchmark table.
    """
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    query = "DELETE FROM public.benchmark;"
    
    cur.execute(query)
    conn.commit()

    cur.close()
    conn.close()

def print_all_records():
    """
    @brief Prints all records from the benchmark table.
    @details This function fetches and prints all records from the public.benchmark table.
    """
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    query = "SELECT * FROM public.benchmark;"
    
    cur.execute(query)
    records = cur.fetchall()

    for record in records:
        print(record)

    cur.close()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database manipulation tool")
    parser.add_argument("--clean", action="store_true", help="Clean up the database")
    parser.add_argument("--insert", metavar="CONFIG_FILE", help="Insert a fake record defined in the external file")
    parser.add_argument("--print", action="store_true", help="Print all records in the benchmark table")
    parser.add_argument("--recreate", metavar="SQL_FILE", help="Recreate the table from an SQL file")

    args = parser.parse_args()

    if args.clean:
        cleanup_db()
    if args.insert:
        insert_benchmark(args.insert)
    if args.print:
        print_all_records()
    if args.recreate:
        recreate_table_from_file(args.recreate)
