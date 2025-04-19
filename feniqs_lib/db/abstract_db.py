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


from abc import ABC, abstractmethod
import psycopg2
import json
import logging
from .config_db import load_config

# logger instance for db
logger_db = logging.getLogger(__name__)


class Abstract_db(ABC):

    config = load_config()
    DATABASE_NAME = config['database']['name']
    USER = config['database']['user']
    HOST = config['database']['host']
    PASSWORD = config['database']['password']
    TABLE_NAME = config['paths']['table_name']
    FILE_PATH = config['paths']['file_path']

    def __init__(self):
        """
        Initialize the database connection and cursor.
        """
        try:
            self.conn = psycopg2.connect(
                database=self.DATABASE_NAME,
                user=self.USER,
                host=self.HOST,
                password=self.PASSWORD
            )
            self.cur = self.conn.cursor()
        except psycopg2.Error as e:
            logger_db.error(f"Error connecting to database: {e}")
            raise

    def close_connection(self):
        """
        Close the database connection.
        """
        try:
            self.conn.close()
        except psycopg2.Error as e:
            logger_db.error(f"Error closing connection: {e}")
            raise

    def commit_changes(self):
        """
        Commit the current transaction.
        """
        try:
            self.conn.commit()
        except psycopg2.Error as e:
            logger_db.error(f"Error committing changes: {e}")
            raise

    def execute_query(self, query, params=None, commit_changes=True):
        """
        Execute a query on the database.

        Args:
            query (str): The SQL query to execute.
            params (tuple, optional): Parameters for the SQL query. Defaults to None.
            commit_changes (bool, optional): Whether to commit the transaction. Defaults to True.
        """
        try:
            self.cur.execute(query, params)
            if commit_changes:
                self.commit_changes()
        except psycopg2.Error as e:
            logger_db.error(f"Query failed: {query} - Error: {e}")
            self.conn.rollback()
            raise

    def fetch_query(self, query, params=None):
        """
        Fetch the results of a query.

        Args:
            query (str): The SQL query to execute.
            params (tuple, optional): Parameters for the SQL query. Defaults to None.

        Returns:
            list: The result of the query.
        """
        self.execute_query(query, params, commit_changes=False)
        try:
            return self.cur.fetchall()
        except psycopg2.Error as e:
            logger_db.error(f"Error fetching query result: {e}")
            raise

    def execute_file(self, file_path):
        """
        Execute SQL commands from a file.

        Args:
            file_path (str): Path to the SQL file.
        """
        try:
            with open(file_path, 'r') as file:
                self.execute_query(file.read())
        except (IOError, psycopg2.Error) as e:
            logger_db.error(f"Error executing file {file_path}: {e}")
            raise

    def get_columns_from_table(self, table, columns):
        """
        Get data from specific columns in a table.

        Args:
            table (str): Table name.
            columns (list or str): Columns to fetch.

        Returns:
            list: Data from the specified columns.
        """
        if isinstance(columns, list):
            columns = ', '.join(columns)
        query = f"SELECT {columns} FROM {table}"
        return self.fetch_query(query)

    def get_all_from_table(self, table):
        """
        Get all data from a table.

        Args:
            table (str): Table name.

        Returns:
            list: All data from the table.
        """
        return self.get_columns_from_table(table, '*')

    def delete_row_from_table(self, table, column, value, condition='='):
        """
        Delete a specific row from a table.

        Args:
            table (str): Table name.
            column (str): Column name to match for deletion.
            value (str): Value to match for deletion.
            condition (str, optional): Condition for matching. Defaults to '='.
        """
        query = f"DELETE FROM {table} WHERE {column} {condition} %s"
        try:
            self.execute_query(query, (value,))
            logger_db.warning(
                f"Row with {column} {condition} {value} deleted from table '{table}'.")
        except psycopg2.Error as e:
            logger_db.error(f"Error deleting row from table '{table}': {e}")
            self.conn.rollback()
            raise

    def delete_all_from_table(self, table):
        """
        Delete all rows from a table.

         Args:
            table (str): Table name.
        """
        query = f"DELETE FROM {table}"
        try:
            self.execute_query(query)
            logger_db.warning(f"All rows deleted from table '{table}'.")
        except psycopg2.Error as e:
            logger_db.error(
                f"Error deleting all rows from table '{table}': {e}")
            self.conn.rollback()
            raise

    def update_values(self, table, updates, conditions):
        """
        Update values in a table based on conditions.

        Args:
            table (str): Table name.
            updates (dict): Columns and their new values.
            conditions (dict): Conditions to match for updating.
        """
        set_clause = ', '.join([f"{k} = %s" for k in updates.keys()])
        condition_clause = ' AND '.join(
            [f"{k} = %s" for k in conditions.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {condition_clause}"
        params = list(updates.values()) + list(conditions.values())
        try:
            self.execute_query(query, params)
            logger_db.info(
                f"Updated values in table '{table}' with conditions {conditions}.")
        except psycopg2.Error as e:
            logger_db.error(f"Error updating values in table '{table}': {e}")
            self.conn.rollback()
            raise

    @abstractmethod
    def drop_all_tables(self):
        """
        Abstract method to drop all tables. Must be implemented in subclasses."""
        raise NotImplementedError

    def drop_one_table(self, table_name):
        """
        Drop a specific table.

        Args:
            table_name (str): Table name.
        """
        query = f"DROP TABLE IF EXISTS {table_name}"
        try:
            self.execute_query(query)
            logger_db.info(f"Table '{table_name}' dropped successfully.")
        except psycopg2.Error as e:
            logger_db.error(f"Error dropping table '{table_name}': {e}")
            self.conn.rollback()
            raise

    @abstractmethod
    def create_all_tables(self):
        """
        Abstract method to create all tables. Must be implemented in subclasses.
        """
        raise NotImplementedError

    def create_one_table(self, table_name, columns):
        """
        Create a specific table.

        Args:
            table_name (str): Table name.
            columns (list): List of tuples containing column name and data type.
        """
        columns_clause = ', '.join(
            [f"{name} {dtype}" for name, dtype in columns])
        query = f"CREATE TABLE {table_name} ({columns_clause})"
        try:
            self.execute_query(query)
            logger_db.info(
                f"Table '{table_name}' created successfully with columns: {columns_clause}.")
        except psycopg2.Error as e:
            logger_db.error(f"Error creating table '{table_name}': {e}")
            self.conn.rollback()
            raise

    @staticmethod
    def _format_value_insert(data):
        """
        Format a value for SQL insertion, depending on its type.

        Args:
            data (Any): The data to format before adding to the query.

        Returns:
            str: The formatted string for the query.
        """
        if isinstance(data, dict):
            # Convert dict to JSON string for SQL insertion
            return f"'{json.dumps(data)}'"
        return f"'{str(data)}'"

    def insert_one_into_table(self, table, data):
        """
        Insert a single row into a table.

        Args:
            table (str): Table name.
            data (dict): Data to insert.
        """
        columns = ', '.join(data.keys())
        values_placeholder = ', '.join(['%s'] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({values_placeholder})"
        try:
            self.execute_query(query, list(data.values()))
        except psycopg2.Error as e:
            logger_db.error(f"Error inserting one row: {e}")
            self.conn.rollback()
            raise

    def insert_many_into_table(self, table, data_list):
        """
        Insert multiple rows into a table.

        Args:
            table (str): Table name.
            data_list (list): List of data dictionaries to insert.
        """
        if not data_list:
            return
        columns = ', '.join(data_list[0].keys())
        values_placeholder = ', '.join(['%s'] * len(data_list[0]))
        query = f"INSERT INTO {table} ({columns}) VALUES ({values_placeholder})"
        params = [list(data.values()) for data in data_list]
        try:
            self.cur.executemany(query, params)
            self.commit_changes()
        except psycopg2.Error as e:
            logger_db.error(f"Error inserting multiple rows: {e}")
            self.conn.rollback()
            raise
