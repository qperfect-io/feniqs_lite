
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
import signal
import subprocess
import pluggy
import yaml
import json
from .hook_specs import BackendSpec
from ..tools import constants
import os
from os import cpu_count
import logging


class PluginManager:

    LOG_FILE = "./feniqs_plugin_manager.log"
    # Set up logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ])
    logger = logging.getLogger(__name__)

    _instance = None

    def __new__(cls, plugin_folder=constants.PLUGIN, venv_folder=None, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def _determine_class_name(self, backend_name):
        """
        Determines the Class name of the Backend from the name of the environment

        :param backend_name: Name of the environment in the yaml file
        :type backend_name: str
        """
        class_name = ''
        up = True
        for char in backend_name:
            # ignores the _ and add a maj for the next character
            if char == '_':
                up = True
                continue

            if up:
                char = char.upper()

            class_name += char
            up = False

        return class_name

    def path_to_module(self, path):
        return path.replace('/', '.')

    def env_from_venv_deps(self):
        """
        This function uses the file VENV_CONFIG to generate a dictionnary with all important information regarding the path to the environment and plugins for each simulation

        :return: The dictonnary of all the information
        :rtype: dict
        """

        envs = {}
        with open(constants.VENV_CONFIG, 'r') as file:
            venv_deps = yaml.safe_load(file)

        for name in venv_deps['venv_description'].keys():
            venv_name = '.' + name + '_env'
            plugin_name = '.' + name + '_plugin'
            yaml_path = name
            class_name = self._determine_class_name(name)
            # Environment will be created at the same location as the plugins in a folder name venv
            envs[class_name] = {'venv_name': venv_name,
                                'plugin_path': self.path_to_module(constants.PLUGIN) + plugin_name, 'yaml_name': yaml_path}
 

        return envs

    def __init__(self, plugin_folder=constants.PLUGIN):
        """
        The Plugin manager is the interface to handle the plugins by calling them in the right environment and with the right options

        :param plugin_folder: The path to the folder containing all the plugins, defaults to 'constants.PLUGIN'.
        :type plugin_folder: str, optional
        """
        self.env_from_venv_deps()
        if hasattr(self, '_initialized') and self._initialized:
            return
        plugin_folder = plugin_folder.replace('/', '.') + '.'
        self.plugin_manager = pluggy.PluginManager("feniqs")
        self.plugin_manager.add_hookspecs(BackendSpec)

        # TODO: this eventually have to be changed to use a yaml file
        self.envs = self.env_from_venv_deps()
        self.discovered_plugins = set()
        self.loaded_plugins = set()
        self._initialized = True

    def load_plugins(self):
        # Discover available plugins
        for backend_name in self.envs.keys():
            self.discovered_plugins.add(backend_name)

    def _register_plugin(self, plugin_name):
        if plugin_name not in self.envs:
            raise ValueError(f"Environment for plugin {plugin_name} not found")

        self.loaded_plugins.add(plugin_name)

    def register_plugin(self, plugin_name):
        if plugin_name in self.loaded_plugins:
            self.logger.info(f"Plugin {plugin_name} is already registered.")
        else:
            self._register_plugin(plugin_name)
            self.logger.info(f"Plugin {plugin_name} has been registered.")

    def unregister_plugin(self, plugin_name):
        if plugin_name in self.loaded_plugins:
            self.loaded_plugins.remove(plugin_name)
            self.logger.info(f"Plugin {plugin_name} unregistered.")
        else:
            self.logger.info(f"Plugin {plugin_name} is not registered.")

    def _add_common_options(self, options):
        """
        Some options are common to all backends, this function will add them to the options dict

        Args:
            options (dict): The same dict with the common options added
        """
        # The value doesn't matter as it will be overwritten by the user
        options['nb_shots'] = None
        options['test_case'] = None
        return options

    def _get_backend_config_options(self, backend_name, **option_given):
        """
        This function will read the yaml file to retrieve all the supported backend options

        Args:
            backend_name (str): The name of the target backend
            option_given (dict): Are the options the user wants to use, they will be filtered using what is supported by the backend 
        """

        with open(constants.BACKEND_CONFIG, 'r') as file:
            backend_options = yaml.safe_load(file)

        # all options is a dict containing all the options for the backend with the default value
        all_options = backend_options['backends'][backend_name]
        all_options = self._add_common_options(all_options)
        if 'env' in option_given:
            all_options['env'] = option_given['env']
        for key in option_given.keys():
            # overwrite the default value with the one given by the user
            if key in all_options:
                all_options[key] = option_given[key]
            else:
                self.logger.info(
                    f"Option {key} is not supported by backend {backend_name}")

        return all_options

    def run_backend(self, backend_name, file, nb_shots, **kwargs):
        """
        This function will run the given simulation with the given backend

        :param backend_name: Name of the backend to use for the simulation
        :type backend_name: str
        :param file: The path to the qasm file containing the target benchmark
        :type file: str
        :param nb_shots: Nb of shots to use on the simulation
        :type nb_shots: int
        :raises ValueError: The environment to run the simulation on the precised backend was not setup properly 
        :raises RuntimeError: The environment to run the simulation on the precised backend might not have been setup properly 
        :raises RuntimeError: An error might have happened during the runtime of the simulation in the separate env 
        :return: the samples, metrics, and config retrieved from the simulation
        :rtype: dict, dict, dict
        """

        if backend_name not in self.envs:
            raise ValueError(f"Backend {backend_name} environment not found")

        if backend_name not in self.loaded_plugins:
            raise RuntimeError(f"Backend {backend_name} is not registered")

        env_path = constants.PLUGIN + '/venv/' + \
            self.envs[backend_name]['venv_name']

        plugin_path = self.envs[backend_name]['plugin_path']

        backend_options = self._get_backend_config_options(
            self.envs[backend_name]['yaml_name'], test_case=file, nb_shots=nb_shots, **kwargs)

        if backend_name in ['YaoCpu', 'MimiqJuliaCpu']:
            backend_options = self._get_backend_config_options(
                self.envs[backend_name]['yaml_name'], test_case=file, nb_shots=nb_shots, env=env_path, **kwargs)

        # para_threads = 1 if not backend_options.get(
        #     'max_parallel_threads', None) else backend_options.get('max_parallel_threads')
        para_threads = cpu_count()
        command = (
            f'export QULACS_NUM_THREADS={para_threads}'
            f'export OPM_NUM_THEADS={para_threads}; '
            f'export OMP_PROC_BIND=\'spread\'; '
            f'{env_path}/bin/python -u -c '
            f'"from {plugin_path} import plugin; '
            f'all_options = {backend_options}; '
            f'(plugin().run_backend(**all_options))"'
        )


        plugin_path = self.envs[backend_name]['plugin_path']

        result = subprocess.run(command, shell=True,                   text=True)
  

        # Writting all the results in the same file implies we can't parallelize the simulations

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to run backend {backend_name}: {result.stderr}")
        # Check where to find the file containing the results
        with open(constants.BACKEND_CONFIG, 'r') as file:
            path_to_results = yaml.safe_load(file)["result_file"]

        # Load the results from the yaml at the precised location
        with open(path_to_results, 'r') as file:
            results = yaml.safe_load(file)

        # returns all the results
        metrics = results['metrics']
        config = results['config']

        return  metrics, config

    def find_key_by_value(dictionary, value):
        for outer_key, inner_dict in dictionary.envs.items():
            if value in inner_dict.values():
                return outer_key
        return None

    def register_all_plugins(self):
        """Register all discovered plugins."""
        self.logger.info("Registering all plugins")
        self.load_plugins()
        for plugin_name in self.discovered_plugins:
            self.register_plugin(plugin_name)
        self.logger.info("All available plugins have been registered.")

    def unregister_all_plugins(self):
        """Unregister all loaded plugins."""
        self.logger.info("Unregistering all plugins")
        for plugin_name in list(self.loaded_plugins):
            self.unregister_plugin(plugin_name)
        self.logger.info("All loaded plugins have been unregistered.")
