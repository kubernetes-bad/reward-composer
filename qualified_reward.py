from typing import List

from qualifiers import Qualifier
from rewards import RewardFunction

class QualifiedReward(RewardFunction):
    """A reward function that only applies if a qualifier returns True."""

    def __init__(self, reward_fn: RewardFunction, qualifier: Qualifier, fallback_value: float = 0.0):
        """
        Initialize a qualified reward function.

        :param reward_fn: Reward function to apply if the qualifier returns True.
        :param qualifier: Qualifier that determines if the reward function should be applied.
        :param fallback_value: Value to return if the qualifier returns False.
        """
        super().__init__(name=f"qualified_{reward_fn.name}", weight=reward_fn.weight)
        self.reward_fn = reward_fn
        self.qualifier = qualifier
        self.fallback_value = fallback_value

    def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        base_rewards = self.reward_fn(completions, prompts, **kwargs)

        qualified_rewards = []
        for i, (completion, score) in enumerate(zip(completions, base_rewards)):
            context = {
                "reward": score,
                "prompt": prompts[i] if i < len(prompts) else "",
                self.reward_fn.name: score,
                **kwargs,
            }

            qualified_rewards.append(score if self.qualifier(completion, context) else self.fallback_value)

        return qualified_rewards
