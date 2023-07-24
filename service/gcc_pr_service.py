#! /usr/bin/python3
import logging
from pathlib import Path
from typing import Optional, Tuple
from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import (
        ActionSpace,
        NamedDiscreteSpace,
        Event,
        Space,
        ObservationSpace,
        DoubleRange,
        Int64Range,
        )
from compiler_gym.service.runtime import create_and_run_compiler_gym_service
from shutil import copytree
from compiler_gym.datasets import (BenchmarkUri, Benchmark)
from subprocess import *
from time import *

class GccPRCompilationSession(CompilationSession):

    compiler_version: str = "7.3.0"

    action_spaces = [
        ActionSpace(
            name="default",
            space=Space(
                named_discrete=NamedDiscreteSpace(
                    name=[
                        "pass1",
                        "pass2",
                        "pass3",
                        ],
                    ),
                ),
            )
        ]

    observation_spaces = [
        ObservationSpace(
            name="runtime",
            space=Space(
                double_value=DoubleRange(min=0),
                ),
            deterministic=False,
            platform_dependent=True,
            default_observation=Event(
                double_value=0,
                ),
            ),
        ObservationSpace(
            name="size",
            space=Space(
                int64_value=Int64Range(min=0),
                ),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(
                int64_value=0,
                ),
            ),
        ObservationSpace(
            name="base_runtime",
            space=Space(
                double_value=DoubleRange(min=0),
                ),
            deterministic=False,
            platform_dependent=True,
            default_observation=Event(
                double_value=0,
                ),
            ),
        ObservationSpace(
            name="base_size",
            space=Space(
                int64_value=Int64Range(min=0),
                ),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(
                int64_value=0,
                ),
            ),
        ]

    def __init__(self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark):
        super().__init__(working_directory, action_space, benchmark)
        self.parsed_bench = BenchmarkUri.from_string(benchmark.uri)
        self.baseline_size = None
        self.baseline_runtime = None
        self._binary_valid = False
        self._src_copied = False
        self._wd_valid = False
        logging.info("Started a compilation session for %s", benchmark.uri)

    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        action_string = action.string_value
        logging.info("Applying action %s", action_string)

        # self.action_spaces[0].space[action_string]

        # Insert pass into corresponding list and ask shuffler to generate a new space 
        # If new action space is empty - return True as first element
        self._binary_valid = False
        return False, None, False

    def get_observation(self, observation_space: ObservationSpace) -> Event:
        logging.info("Computing observation from space %s", observation_space.name)
        if observation_space.name == "runtime":
            return Event(double_value=self.get_runtime())
        elif observation_space.name == "size":
            return Event(int64_value=self.get_size())
        elif observation_space.name == "base_runtime":
            if self.baseline_runtime == None:
                self.get_baseline()
            return Event(double_value=self.baseline_runtime)
        elif observation_space.name == "base_size":
            if self.baseline_size == None:
                self.get_baseline()
            return Event(int64_value=self.baseline_size)
        else:
            raise KeyError(observation_space.name)

    def copy_bench(self):
        if not self._src_copied:
            print(self.working_dir)
            copytree(self.parsed_bench.path, self.working_dir.joinpath('bench'), dirs_exist_ok=True)
            self._src_copied = True

    def prep_wd(self):
        if not self._wd_valid:
            call('touch list1.txt list2.txt list3.txt', shell=True, cwd=self.working_dir.joinpath('bench'))
            self._wd_valid = True

    def get_baseline(self):
        self.compile_baseline()
        self.baseline_size = self.get_size()
        self.baseline_runtime = self.get_runtime()

    def compile_baseline(self):
        self.copy_bench()
        base_opt = " ".join(self.parsed_bench.params.get("base_opt", ["-O2"]))
        src_dir = " ".join(self.parsed_bench.params.get("src_dir"))
        build_arg = " ".join(self.parsed_bench.params.get("build"))
        call(f'''$AARCH_GCC {base_opt} {build_arg} {src_dir}*.c -o bench.elf''', shell=True, cwd=self.working_dir.joinpath('bench'))
        self._binary_valid = True

    def compile(self):
        self.copy_bench()
        self.prep_wd()
        src_dir = " ".join(self.parsed_bench.params.get("src_dir"))
        build_arg = " ".join(self.parsed_bench.params.get("build"))
        plugin_args = "-fplugin-arg-plugin-pass_file=list1.txt -fplugin-arg-plugin-pass_file=list2.txt -fplugin-arg-plugin-pass_file=list3.txt"
        call(f'''$AARCH_GCC -fplugin=$GCC_PLUGIN -fplugin-arg-plugin-pass_replace {plugin_args} {build_arg} {src_dir}*.c -o bench.elf''', shell=True, cwd=self.working_dir.joinpath('bench'))
        self._binary_valid = True

    def get_runtime(self):
        if not self._binary_valid:
            self.compile()
        arg = " ".join(self.parsed_bench.params.get("run"))
        start_time = clock_gettime(CLOCK_MONOTONIC)
        run(f'qemu-aarch64 -L /usr/aarch64-linux-gnu ./bench.elf {arg}', shell=True, cwd=self.working_dir.joinpath('bench'), check=True)
        end_time = clock_gettime(CLOCK_MONOTONIC)
        return end_time - start_time

    def get_size(self):
        if not self._binary_valid:
            self.compile()
        return int(run('size bench.elf', shell=True, capture_output=True, cwd=self.working_dir.joinpath('bench')).stdout.split()[6])


if __name__ == "__main__":
    create_and_run_compiler_gym_service(GccPRCompilationSession)


