from abc import ABC, abstractmethod
from typing import List

from .rewards import RewardFunction, MultiTurnInput


class MultiTurnRewardFunction(RewardFunction, ABC):
    @abstractmethod
    def __call__(self, completions: MultiTurnInput, prompts: MultiTurnInput, **kwargs) -> List[float]:
        pass
