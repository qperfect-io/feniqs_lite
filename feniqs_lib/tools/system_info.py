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
import platform
import sys
from decimal import Decimal

class SystemInfo:
    def __init__(self):
        """
        Initialize SystemInfo with basic system information.
        """
        self.hostname = platform.node()
        self.os = platform.platform(aliased=True)
        self.cpu_model = self.get_cpu_model()
        self.cpu_cores = self.get_cpu_cores()
        self.memory = self.get_memory()
        self.turbo_boost_enabled = self.is_turbo_boost_enabled()

    def get_cpu_model(self):
        """
        Get the CPU model.
        """
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Darwin":  # macOS
            return self._run_command("sysctl -n machdep.cpu.brand_string")
        elif platform.system() == "Linux":
            return self._read_linux_cpu_info("model name")
        return "Unknown"

    def get_cpu_cores(self):
        """
        Get the number of CPU cores.
        """
        return os.cpu_count()

    def get_memory(self):
        """
        Get total memory in adaptive units.
        """
        memory_bytes = 0
        if platform.system() == "Windows":
            memory_bytes = int(self._run_command("wmic ComputerSystem get TotalPhysicalMemory").split()[1])
        elif platform.system() == "Darwin":  # macOS
            memory_bytes = int(self._run_command("sysctl -n hw.memsize"))
        elif platform.system() == "Linux":
            mem_info = self._read_linux_mem_info("MemTotal")
            if mem_info.endswith(" kB"):
                memory_bytes = int(mem_info[:-3]) * 1024  # Convert to bytes

        return self._format_memory(memory_bytes)

    def is_turbo_boost_enabled(self):
        """
        Check if Turbo Boost is enabled.
        """
        if platform.system() == "Linux":
            turbo_boost_file = "/sys/devices/system/cpu/cpufreq/boost"
            turbo_boost_pstate_file = "/sys/devices/system/cpu/intel_pstate/no_turbo"
            if os.path.exists(turbo_boost_file):
                return self._read_file(turbo_boost_file) != "0"
            if os.path.exists(turbo_boost_pstate_file):
                return self._read_file(turbo_boost_pstate_file) == "0"
        return "Unknown"

    def _format_memory(self, bytes):
        """
        Format memory size into human-readable form with appropriate units.
        """
        if bytes < 1024:
            return f"{bytes} B"
        elif bytes < 1024 ** 2:
            return f"{bytes / 1024:.2f} KB"
        elif bytes < 1024 ** 3:
            return f"{bytes / 1024 ** 2:.2f} MB"
        elif bytes < 1024 ** 4:
            return f"{bytes / 1024 ** 3:.2f} GB"
        else:
            return f"{bytes / 1024 ** 4:.2f} TB"

    def _run_command(self, command):
        """
        Run a system command and return its output.
        """
        try:
            return os.popen(command).read().strip()
        except Exception as e:
            print(f"Failed to run command '{command}': {e}")
            return "Unknown"

    def _read_file(self, filepath):
        """
        Read the content of a file.
        """
        try:
            with open(filepath, 'r') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Failed to read file '{filepath}': {e}")
            return "Unknown"

    def _read_linux_cpu_info(self, key):
        """
        Read CPU information from /proc/cpuinfo on Linux.
        """
        cpu_info_file = "/proc/cpuinfo"
        if os.path.isfile(cpu_info_file) and os.access(cpu_info_file, os.R_OK):
            with open(cpu_info_file, "r") as file:
                for line in file:
                    if line.startswith(key):
                        return line.split(":")[1].strip()
        return "Unknown"

    def _read_linux_mem_info(self, key):
        """
        Read memory information from /proc/meminfo on Linux.
        """
        mem_info_file = "/proc/meminfo"
        if os.path.isfile(mem_info_file) and os.access(mem_info_file, os.R_OK):
            with open(mem_info_file, "r") as file:
                for line in file:
                    if line.startswith(key):
                        return line.split(":")[1].strip()
        return "Unknown"


# Example usage
if __name__ == "__main__":
    sys_info = SystemInfo()
    print(f"Hostname: {sys_info.hostname}")
    print(f"OS: {sys_info.os}")
    print(f"CPU Model: {sys_info.cpu_model}")
    print(f"CPU Cores: {sys_info.cpu_cores}")
    print(f"Memory: {sys_info.memory}")
    print(f"Turbo Boost Enabled: {sys_info.turbo_boost_enabled}")

