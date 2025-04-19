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

from __future__ import annotations

import atexit
import os
import platform
import socket
import sys
import configargparse
import textwrap
import gevent
import requests
import logging

from feniqs_lib.tools.constants import DB_CONFIG
from feniqs_lib.tools.constants import BACKEND_CONFIG
from feniqs_lib.tools.constants import QASM_FOLDER
from feniqs_lib.tools.constants import VENV_CONFIG

# Module version
version = "0.0.0"  # feniqs.__version__

# Setting up the logger for statistics
logger = logging.getLogger("feniqs.stats_logger")

CURRENT_DIR = "."
LOG_FILE = "./feniqs_daemon.log"
BACKEND_CONFIG = "./yaml/config_backend.yaml"
VENV_CONFIG = "./yaml/venv_deps.yaml"

# Convert relative paths to absolute paths
QASM_FOLDER = os.path.abspath(QASM_FOLDER)
LOG_FILE = os.path.abspath(LOG_FILE)
BACKEND_CONFIG = os.path.abspath(BACKEND_CONFIG)
DATA_DIR = os.path.abspath(CURRENT_DIR)

class FeniqsArgument(configargparse.ArgumentParser):
    """
    Custom ArgumentParser that includes additional attributes for arguments
    to be included in the interface and to mark secret arguments.
    """
    def add_argument(self, *args, **kwargs) -> configargparse.Action:
        include_in_interface = kwargs.pop("include_in_interface", True)
        is_secret = kwargs.pop("is_secret", False)
        action = super().add_argument(*args, **kwargs)
        action.include_in_interface = include_in_interface
        action.is_secret = is_secret
        return action

    @property
    def args_included_in_interface(self) -> dict[str, configargparse.Action]:
        """
        Return arguments that are included in the interface.
        """
        return {a.dest: a for a in self._actions if hasattr(a, "include_in_interface") and a.include_in_interface}

    def print_help(self, file=None):
        """
        Override the print_help method to call print_welcome_message() before displaying the help message.
        """
        super().print_help(file)

def get_empty_argument_parser(add_help: bool = True) -> FeniqsArgument:
    """
    Create an argument parser with predefined options for Feniqs.
    """
    parser = FeniqsArgument(
        default_config_files="",  # Placeholder for default config files
        add_env_var_help=False,
        add_config_file_help=False,
        add_help=add_help,
        formatter_class=configargparse.RawDescriptionHelpFormatter,
        usage=configargparse.SUPPRESS,
        description=textwrap.dedent(
            """
            Usage: feniqs [options]
            """
        ),
        epilog="""Example to run: python feniqs_daemon/main.py --help""",
    )

    return parser

def set_options(parser: configargparse.ArgumentParser):
    """
    Define general and group-specific options for the parser.
    """
    parser._optionals.title = "General options"
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        help="Show FENIQS version number",
        version=f"feniqs {version} from {os.path.dirname(__file__)} (python {platform.python_version()})",
    )

    log_group = parser.add_argument_group("Logger's options")
    log_group.add_argument(
        "--logfile",
        default=LOG_FILE, 
        help="Path to log file. If not set, log will go to stderr.",
        metavar="<filename>",
        env_var="feniqs_LOGFILE",
    )
    log_group.add_argument(
        "-l",
        "--level",
        default="INFO",
        help="Available logger level: DEBUG-INFO-WARNING-ERROR-CRITICAL. INFO is used by default.",
        metavar="<level>",
        env_var="feniqs_LOGLEVEL",
    )

    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run FENIQS in a foreground mode to printing the messages out to the console.",
    )

    parser.add_argument(
        "--watchfolder",
        default=QASM_FOLDER,
        metavar="<path>",
        help="The folder to watch for modifications when running as a daemon.",
    )
    
    # Enable or disable database usage (default: true)
    parser.add_argument(
        "--db",
        default=True,
        type=lambda x: x.lower() in ["true", "1", "yes"],
        help="Enable database usage. Set to false if the database is not installed (default: true).",
    )
    # Save benchmark results in JSON format (default: false)
    parser.add_argument(
        "--json",
        default=False,
        type=lambda x: x.lower() in ["true", "1", "yes"],
        help="Save benchmark results in JSON format. Default is false (i.e. results are not saved in JSON format unless specified).",
    )
    # Enable or disable mirror fidelity usage (default: false)
    parser.add_argument(
        "--mf",
        default=False,
        type=lambda x: x.lower() in ["true", "1", "yes"],
        help="Enable mirror fidelity functionality. Default is false.",
    )
    # Timeout value for simulation (default: 300 seconds)
    parser.add_argument(
        "--timeout",
        default=300,
        type=int,
        help="Timeout value for simulation in seconds. Default is 300 seconds.",
    )
    # Number of runs for each banchmark (default: 3)
    parser.add_argument(
        "--runs",
        default=3,
        type=int,
        help="Number of runs for each benchmark. Default is 3 times.",
    )
    # Important for julia - by default the first (most slow) iteration is not included
    parser.add_argument(
        "--incl_first",
        default=True,
        type=int,
        help="Number of runs for each benchmark. Default is 3 times.",
    )

def parse_options(args=None) -> configargparse.Namespace:
    """
    Parse command-line options using the predefined parser settings.
    """
    parser = get_empty_argument_parser(add_help=True)
    set_options(parser)
    parsed_opts = parser.parse_args(args=args)
    return parsed_opts

def _is_package(path: str) -> bool:
    """
    Check if the given path is a Python package.
    """
    return (
        os.path.isdir(path) and
        os.path.isfile(os.path.join(path, '__init__.py'))
    )





