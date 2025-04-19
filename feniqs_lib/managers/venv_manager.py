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
import subprocess
import shutil
import yaml
import logging
from ..tools import constants


class VenvManager:
    LOG_FILE = "./feniqs_venv_manager.log"
    # Set up logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ])
    logger = logging.getLogger(__name__)

    def __init__(self, yaml_file_path):
        """
        Initialize the environment manager.

        Args:
            yaml_file_path (str): Path to the YAML file defining the environments.
        """
        self.yaml_file_path = yaml_file_path
        with open(self.yaml_file_path, 'r') as file:
            venv_info = yaml.safe_load(file)

        self.base_env_dir = constants.PLUGIN + '/venv'
        if not os.path.exists(self.base_env_dir):
            os.makedirs(self.base_env_dir)

        self.venv_to_create = venv_info['venv_description']
        self.installed_envs = []  # Track installed environments

    def create_envs(self):
        """
        Create virtual environments and install dependencies as described in the YAML file.
        """
        self.logger.info("Installing Environments.")
        for env_name, env in self.venv_to_create.items():

            self._install_env(env, env_name)

        self.logger.info("All Environments were installed.")

    def create_env_by_name(self, env_name):
        """
        Install a virtual environment by name.

        Args: 
            env_name (str) : Name of the environment to install.
        """
        if env_name not in self.venv_to_create:
            self.logger.info(
                f"Environment '{env_name}' not found in the YAML file.")
            return
        env = self.venv_to_create[env_name]

        self._install_env(env, env_name)

    def _install_env(self, env, env_name):
        """
        Will procees the environment installation

        Args: 
            env (dict): all the envs information
            env_name (str): name of the targeted env
        """

        if not env.get('install', False):
            self.logger.info(
                f"Skipping installation for '{env_name}' as it is disabled (install: false).")
            return
        folder_name = env['folder_name']
        env_path = os.path.join(self.base_env_dir, folder_name)
        if not os.path.exists(env_path):
            self._create_virtualenv(env_path)
        else:
            self.logger.info(
                f"Environment '{folder_name}' already exists.")
        
        self._install_dependencies(
            env_path, env['dependencies'], ignore_version=env.get('ignore_version', False))
        if env.get('julia_packages'):
            self._install_julia_packages(env_path, env['julia_packages'])
        # Track the installed environment
        self.installed_envs.append(env_name)
        self.logger.info(
            f"Environment '{env_name}' installed to '{env_path}'.")

    def _create_virtualenv(self, env_path):
        """
        Create a virtual environment.

        Args: 
            env_path (str): Path where the virtual environment will be created.
        """
        self.logger.info(f"Creating virtual environment at '{env_path}'")

        subprocess.run([self._get_python_executable(), '-m', 'venv', env_path],  text=True)
        

    def _install_dependencies(self, env_path, dependencies, ignore_version=False):
        """
        Install the dependencies in the virtual environment.

        Args: 
            env_path (str): Path to the virtual environment.
            dependencies (list(str)): List of dependencies to install.
            ignore_version (bool): If true will ignore the version precised in the list, optional.
        """
        packages_to_install = []
        if ignore_version:
            packages_to_install = [dep.split('==')[0] for dep in dependencies]
        else:
            packages_to_install = dependencies
        packages_to_install.append('pluggy')
        packages_to_install.append('pyyaml')
        packages_to_install.append('colorlog')
        pip_executable = self._get_pip_executable(env_path)
        python_executable = self._get_python_executable_by_path(env_path)
        if packages_to_install:
            self.logger.info(
                f"Installing dependencies in '{env_path}': {packages_to_install}")
            print([python_executable, pip_executable, 'install', *packages_to_install, '--upgrade'])
            subprocess.run([python_executable, '-m',  pip_executable, 'install', *
                           packages_to_install, '--upgrade'],  check=True)
        else:
            self.logger.info(
                f"All dependencies are already installed in '{env_path}'.")

    def _install_julia_packages(self, env_path, julia_packages):
        """
        Install the Julia packages in the virtual environment.
        
        Args: 
            env_path (str): Path to the virtual environment.
            julia_packages (list(srt)): List of Julia packages to install.
        """
        self.logger.info(
            f"Installing Julia packages in '{env_path}': {julia_packages}")
        julia_executable = self._get_julia_executable()
        for package in julia_packages:
            subprocess.run([julia_executable, '--project=' +
                           env_path, '-e', f'using Pkg; Pkg.add("{package}")'])
        subprocess.run([julia_executable, '--project=' + env_path,
                       '-e', 'using Pkg; Pkg.update(); Pkg.instantiate()'])

    def _get_python_executable(self):
        """
        Get the correct Python executable depending on the platform.

        Returns:
            str: The python executable
        """
        return 'python3' if os.name != 'nt' else 'python'

    def _get_python_executable_by_path(self, env_path):
        if os.name == 'nt':
            return os.path.join(env_path, 'bin', 'python')
        else:
            return os.path.join(env_path, 'bin', 'python3')

    def _get_pip_executable(self, env_path):
        """
        Get the correct pip executable depending on the platform.
        Args: 
            env_path (str): Path to the virtual environment.

        Returns:
            str: Pip executable.
        """
        if os.name == 'nt':
            return os.path.join(env_path, 'Scripts', 'pip.exe')
        else:
            return os.path.join('pip')

    def _get_julia_executable(self):
        """
        Get the Julia executable depending on the platform.
        
        Returns: 
            str: Julia executable.
        
        """
        return 'julia'

    def delete_all_envs(self):
        """
        Delete all virtual environments.
        """
        for _, env in self.venv_to_create.items():
            folder_name = env['folder_name']
            env_path = os.path.join(self.base_env_dir, folder_name)
            if os.path.exists(env_path):
                shutil.rmtree(env_path)
            else:
                self.logger.info(
                    f"Environment '{folder_name}' does not exist.")

    def delete_env_by_name(self, env_name):
        """
        Delete a virtual environment by name.
        Args: 
            env_name (str): Name of the environment to delete.
        """
        if env_name not in self.venv_to_create:
            self.logger.warning(
                f"Environment '{env_name}' not found in the YAML file.")
            return
        env = self.venv_to_create[env_name]
        folder_name = env['folder_name']
        env_path = os.path.join(self.base_env_dir, folder_name)
        if os.path.exists(env_path):
            shutil.rmtree(env_path)
        else:
            self.logger.warning(f"Environment '{folder_name}' does not exist.")


if __name__ == '__main__':
    # Example usage:
    yaml_file_path = 'feniqs/yaml/venv_deps.yaml'
    venv_manager = VenvManager(yaml_file_path)
    venv_manager.create_envs()

