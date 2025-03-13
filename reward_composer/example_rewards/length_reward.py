import math
from typing import List

from ..rewards import RewardFunction, timed_execution

def gaussian(length: int, target_length: int, sigma: float) -> float:
    return math.exp(-((length - target_length) ** 2) / (2 * sigma ** 2))

def asymmetric_linear_with_plateau(length: int, target_length: int, plateau: int, decay_right: float) -> float:
    if length <= target_length - plateau/2:
        return length / (target_length - plateau/2)
    elif length < target_length + plateau/2:
        return 1.0
    else:
        return max(0.0, 1 - (length - (target_length + plateau/2)) / ((target_length - plateau/2) * decay_right))

class LengthReward(RewardFunction):
    def __init__(self, target_length: int, weight: float = 1.0, plateau: int = 200):
        super().__init__(name="length_reward", weight=weight)
        self.target_length = target_length
        self.plateau = plateau

    @timed_execution
    def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> list[float]:
        rewards = []

        for completion in completions:
            length = len(completion)
            # if length < MIN_LEN or length > MAX_LEN:
            #     rewards.append(0.0) # get fucked

            reward = asymmetric_linear_with_plateau(length, self.target_length, self.plateau, 0.5)
            reward = max(0.0, min(1.0, reward))
            rewards.append(reward)

        return rewards
