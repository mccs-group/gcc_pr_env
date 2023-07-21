from pathlib import Path

from compiler_gym.spaces import Reward
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

GCC_PR_SERVICE_BINARY: Path = runfiles_path("compiler_gym/envs/gcc_pr/service/gcc_pr_service.py")

class SizeRuntimeReward(Reward):

    def __init__(self):
        super().__init__(
                name="size_runtime",
                observation_spaces=["runtime", "size"],
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

# 		if self.previous_runtime is None:
# 			self.previous_runtime = observation_view["runtime"]
# 
# 		if self.previous_size is None:
# 			self.previous_size = observation_view["size"]

        reward = observations[0] + observations[1]
        return reward

register(
        id="gcc_pr-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": GCC_PR_SERVICE_BINARY,
            "rewards": [SizeRuntimeReward()],
        },
)
