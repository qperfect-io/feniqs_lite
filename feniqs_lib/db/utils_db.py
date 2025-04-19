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


import os
import psycopg2
from config_db import load_config
import logging

# logger instance for db
logger_db = logging.getLogger(__name__)

def table_exists(connection, table_name):
    """
    Check if a table exists in the database.
    """
    try:
        cur = connection.cursor()
        cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}');")
        return cur.fetchone()[0]
    except psycopg2.Error as e:
        logger_db.error(f"Error checking if table exists: {e}")
        raise
    finally:
        cur.close()

def execute_sql_file(sql_file, connection_params):
    """
    Execute SQL commands from a file.
    """
    if not os.path.exists(sql_file):
        logger_db.error(f"SQL file is not found.")
        raise FileNotFoundError(f"SQL file '{sql_file}' not found.")

    try:
        conn = psycopg2.connect(**connection_params)
        cur = conn.cursor()
        with open(sql_file, 'r') as file:
            sql_commands = file.read()
            cur.execute(sql_commands)
            conn.commit()
        logger_db.error(f"Successfully executed the SQL file '{sql_file}'.")
    except psycopg2.Error as e:
        logger_db.error(f"Error executing the SQL file '{sql_file}': {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def db_check():
    """
    Check if the database is correctly set up and execute SQL file if necessary.
    """
    try:
        config = load_config()
        connection_params = {
            'dbname': config['database']['name'],
            'user': config['database']['user'],
            'host': config['database']['host'],
            'password': config['database']['password']
        }

        sql_file = config['paths']['file_path']

        # Check if the table exists before attempting to create it
        conn = psycopg2.connect(**connection_params)
        if table_exists(conn, 'benchmark'):
            logger_db.warning("Table 'benchmark' already exists. Skipping creation.")
        else:
            execute_sql_file(sql_file, connection_params)
    except FileNotFoundError as fnfe:
        logger_db.error(fnfe)
    except Exception as e:
        logger_db.error(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()


def main():
    # Check the database setup
    db_check()

if __name__ == "__main__":
    main()