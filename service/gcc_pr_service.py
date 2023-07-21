#! /usr/bin/env python3
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
        Benchmark,
        )
from compiler_gym.service.runtime import create_and_run_compiler_gym_service

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
        ]

    def __init__(
            self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark
            ):
        super().__init__(working_directory, action_space, benchmark)
        logging.info("Started a compilation session for %s", benchmark.uri)

    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        action_string = action.string_value
        logging.info("Applying action %s", action_string)

        # self.action_spaces[0].space[action_string]

        # Insert pass into corresponding list and ask shuffler to generate a new space 
        # If new action space is empty - return True as first element
        return False, None, False

    def get_observation(self, observation_space: ObservationSpace) -> Event:
        logging.info("Computing observation from space %s", observation_space.name)
        if observation_space.name == "runtime":
            return Event(double_value=100.101)
        elif observation_space.name == "size":
            return Event(int64_value=75570)
        else:
            raise KeyError(observation_space.name)

if __name__ == "__main__":
    create_and_run_compiler_gym_service(GccPRCompilationSession)


