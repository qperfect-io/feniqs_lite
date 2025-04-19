<div style="display: flex; align-items: center; justify-content: space-between;">
  <h1 style="margin: 0;">feniqs_lite: Database Manipulation Tool</h1>
  <img src="../../assets/logo.png" alt="Feniqs Lite Logo" style="width: 100px;">
</div>

## Database Manipulation Tool - helper_db.py

This tool is designed to help developers with basic database operations on a PostgreSQL database, 
specifically targeting a table called benchmark within a database named feniqs. 
It simplifies the process of managing benchmark records in FENIQS PostgreSQL database by providing easy-to-use command-line options for common tasks.
It provides command-line options to perform the following actions:

- **Clean up the database:** 

Command: --clean

Description: This option deletes all records from the benchmark table, effectively cleaning up the database.


- **Insert a Fake Record:**

Command: --insert JSON_FILE

Description: This option inserts a fake benchmark record into the benchmark table using the data provided in an external JSON file. This file must contain the necessary fields to populate the table.

Example of File: The provided file (insert_ex.json) contains the fake benchmark record data in JSON format. 


- **Print All Records:**

Command: --print

Description: This option prints all records currently present in the benchmark table to the console.


### Usage Examples
- **To clean up the database**: 
```sh
python feniqs_tools/db/helper_db.py --clean
```
- **To insert a fake record using an example file in json format**: 
```sh
python feniqs_tools/db/helper_dp.py --insert insert_ex.json
```
- **To print all records in the benchmark table**: 
```sh
python feniqs_tools/db/helper_dp.py --print
```