-- Set the maximum duration allowed for a statement to execute to infinite (0 means no timeout).
SET statement_timeout = 0;

-- Set the maximum duration a statement will wait to acquire a lock to infinite (0 means no timeout).
SET lock_timeout = 0;

-- Set the maximum time a transaction can be idle (with no running queries) before being automatically aborted to infinite (0 means no timeout).
SET idle_in_transaction_session_timeout = 0;

-- Set the minimum level of messages sent to the client to "warning", filtering out less severe messages.
SET client_min_messages = warning;

-- Disable row-level security policies, allowing all users to see all rows regardless of policy constraints.
SET row_security = off;

-- Set the client's character set encoding to UTF-8.
SET client_encoding = 'UTF8';

-- Ensure that strings are treated as per the SQL standard, which helps avoid SQL injection risks.
SET standard_conforming_strings = on;

-- Clear the search_path setting, ensuring that only the default schemas are searched for unqualified objects.
SELECT pg_catalog.set_config('search_path', '', false);

-- Disable the check of function bodies for syntax errors at creation time.
SET check_function_bodies = false;

-- Set XML data to be stored in "content" mode, focusing on the content within XML tags.
SET xmloption = content;

-- Disable the creation of object identifiers (OIDs) with new table rows by default.
SET default_with_oids = false;

-- Install the Tablefunc extension if it doesn't exist. It provides functions for handling tables, like pivot table creation.
CREATE EXTENSION IF NOT EXISTS tablefunc WITH SCHEMA public;

-- Attach a comment to the Tablefunc extension, explaining its purpose.
COMMENT ON EXTENSION tablefunc IS 'Functions that manipulate whole tables';

-- Reset the default tablespace to the PostgreSQL default, which usually means the primary tablespace.
SET default_tablespace = '';

-- Create a new role with login privileges if it does not exist.
-- DO
-- $$
-- BEGIN
--     IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'feniqs_user') THEN
--         CREATE ROLE feniqs_user WITH LOGIN PASSWORD 'nopassword';
--     END IF;
-- END
-- $$;

-- -- Create a new database if it does not exist and set its owner to feniqs_user.
-- DO
-- $$
-- BEGIN
--     IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'feniqs') THEN
--         CREATE DATABASE feniqs WITH OWNER = feniqs_user;
--     END IF;
-- END
-- $$;

-- -- Connect to the newly created database.
-- \connect feniqs

-- Create a new table named 'benchmark' within the 'public' schema. This table includes multiple columns with various data types.
CREATE TABLE public.benchmark (
    "time" timestamp with time zone NOT NULL, -- Timestamp with timezone, not nullable.
    benchmark_id serial PRIMARY KEY, -- ID of benchmark, which is unique.
    backend text NOT NULL, -- Backend (e.g., qiskit, cirq, qulacs, etc.), not nullable.
    package_version text NOT NULL, -- Version of backend (e.g., version of qiskit_aer or qiskit_aer_gpu).
    backend_type text NOT NULL, -- Type of backend (e.g., sv, mps, tn), not nullable.
    bench_file text NOT NULL, -- Name of the OpenQASM file used in this benchmark, not nullable.
    function_name text NOT NULL, -- Benchmarked function (e.g., parsing, execute, total, etc.).
    nb_qubits integer CHECK (nb_qubits >= 0), -- Number of qubits.
    exception text, -- Text field for exception messages.
    settings jsonb, -- JSONB field for storing benchmark settings (e.g., seed, nb_shots, max_threads, fusion_enable).
    metrics jsonb,  -- JSONB field for storing metrics (e.g., nb_fails, median_rt, avg_rt, min_rt, max_rt, avg_nb_runs_per_tunit, 95_per_rtn, avg_fidelity).
    benchmark_group text NOT NULL -- String identifier used to select a group of benchmarks. for example can be "QFT", and will be used to retrieved all benchmark related to this name
);

-- Change the owner of the 'benchmark' table to feniqs_user.
ALTER TABLE public.benchmark OWNER TO feniqs_user;

