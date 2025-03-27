from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from reward_composer import RewardFunction

@dataclass
class Message: # multiturn convo message
    role: str  # "user" | "assistant" | "system"
    content: str

MultiTurnInput = List[List[Message]]

class MultiTurnRewardFunction(RewardFunction, ABC):
    @abstractmethod
    def __call__(self, completions: MultiTurnInput, prompts: MultiTurnInput, **kwargs) -> List[float]:
        pass



