from pathlib import Path

from compiler_gym.spaces import Reward
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

from compiler_gym.envs.gcc_pr.datasets import *

GCC_PR_SERVICE_BINARY: Path = runfiles_path("compiler_gym/envs/gcc_pr/service/gcc-pr-service")

class SizeRuntimeReward(Reward):

    def __init__(self):
        super().__init__(
                name="size_runtime",
                observation_spaces=["runtime", "size", "base_size", "base_runtime"],
                default_value=0,
                default_negates_returns=False,
                deterministic=False,
                platform_dependent=True,
        )
        self.base_runtime = None
        self.base_size = None

    def reset(self, benchmark: str, observation_view):
        self.base_runtime = None
        self.base_size = None

    def update(self, action, observations, observation_view):
        del action
        #reward = observations[0] + observations[1]
        reward = float(58008)
        return reward

register(
        id="gcc_pr-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": GCC_PR_SERVICE_BINARY,
            "rewards": [SizeRuntimeReward()],
            "datasets": [CBenchDataset()],
        },
)
