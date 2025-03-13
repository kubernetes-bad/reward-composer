import logging
from typing import Union, List, Tuple, Optional

import numpy as np

from .rewards import RewardFunction, get_wandb
from .qualifiers import Qualifier


class CompositeReward(RewardFunction):
    """
    A reward function that combines multiple reward functions.

    The total reward is split into two parts:
    1. Base rewards: capped at base_max (e.g., 0.8)
    2. Bonus rewards: responsible for the remaining portion (e.g., 0.2)

    Bonus rewards can be conditionally applied based on qualifiers.
    """

    def __init__(self,
         base_rewards: List[RewardFunction],
         bonus_rewards: List[Union[RewardFunction, Tuple[RewardFunction, Qualifier]]] = None,
         base_max: float = 0.8,
         total_max: float = 1.0,
         base_qualifier: Optional[Qualifier] = None,
         name: str = "composite_reward",
         weight: float = 1.0):
        """
        Initialize a composite reward function.

        Args:
            :param base_rewards: List of reward functions that form the foundation
            :param bonus_rewards: List of reward functions or (reward_function, qualifier) tuples
                           that provide bonus points
            :param base_max: Maximum value for base rewards (e.g., 0.8)
            :param total_max: Maximum total reward value (e.g., 1.0)
            :param base_qualifier: Optional qualifier that must be satisfied for any base rewards to apply
            :param name: Name of this composite reward
            :param weight: Weight of this reward function
        """
        super().__init__(name=name, weight=weight)
        self.base_rewards = base_rewards
        self.bonus_rewards = []
        self.bonus_qualifiers = []

        if bonus_rewards:
            for item in bonus_rewards:
                if isinstance(item, tuple) and len(item) == 2: # reward_fn, qualifier
                    reward_fn, qualifier = item
                    self.bonus_rewards.append(reward_fn)
                    self.bonus_qualifiers.append(qualifier)
                else: # just reward_fn
                    self.bonus_rewards.append(item)
                    self.bonus_qualifiers.append(None)

        self.base_max = base_max
        self.total_max = total_max
        self.bonus_max = total_max - base_max
        self.base_qualifier = base_qualifier

        # logging stuff
        self._base_values = {}
        self._bonus_values = {}
        self._normalized_base_scores = []
        self._normalized_bonus_scores = []
        self._applied_bonuses = []

    def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        all_reward_values = {}
        # reset
        self._base_values = {}
        self._bonus_values = {}
        self._normalized_base_scores = []
        self._normalized_bonus_scores = []
        self._applied_bonuses = [[] for _ in range(len(completions))]

        base_values = {}
        for reward_fn in self.base_rewards:
            scores = reward_fn(completions, prompts, **kwargs)
            base_values[reward_fn.name] = scores
            all_reward_values[reward_fn.name] = scores
            self._base_values[reward_fn.name] = scores

        bonus_values = {}
        for reward_fn in self.bonus_rewards:
            scores = reward_fn(completions, prompts, **kwargs)
            bonus_values[reward_fn.name] = scores
            all_reward_values[reward_fn.name] = scores
            self._bonus_values[reward_fn.name] = scores

        final_scores = []
        for completion_index in range(len(completions)):
            context = { # context for qualifiers
                "prompt": prompts[completion_index] if completion_index < len(prompts) else "",
                **{name: values[completion_index] for name, values in all_reward_values.items()},
                **kwargs
            }

            base_qualifies = True
            if self.base_qualifier:
                base_qualifies = self.base_qualifier(completions[completion_index], context)

            if base_qualifies:
                base_sum = sum(values[completion_index] * reward_fn.weight
                               for reward_fn, values in zip(self.base_rewards, base_values.values()))

                if self.base_rewards:
                    base_weights_sum = sum(reward.weight for reward in self.base_rewards)
                    normalized_base_sum = (base_sum / base_weights_sum) * self.base_max if base_weights_sum > 0 else 0
                    normalized_base_sum = min(normalized_base_sum, self.base_max)
                else:
                    normalized_base_sum = 0
            else:
                normalized_base_sum = 0

            self._normalized_base_scores.append(normalized_base_sum)

            bonus_sum = 0
            bonus_weights_sum = 0

            for reward_fn, qualifier in zip(self.bonus_rewards, self.bonus_qualifiers):
                bonus_qualifies = True
                if qualifier:
                    bonus_qualifies = qualifier(completions[completion_index], context)

                if bonus_qualifies:
                    reward_name = reward_fn.name
                    bonus_sum += bonus_values[reward_name][completion_index] * reward_fn.weight
                    bonus_weights_sum += reward_fn.weight
                    self._applied_bonuses[completion_index].append(reward_name)

            if bonus_weights_sum > 0:
                normalized_bonus_sum = (bonus_sum / bonus_weights_sum) * self.bonus_max
                normalized_bonus_sum = min(normalized_bonus_sum, self.bonus_max)
            else:
                normalized_bonus_sum = 0

            self._normalized_bonus_scores.append(normalized_bonus_sum)

            final_score = normalized_base_sum + normalized_bonus_sum
            final_scores.append(final_score)

        return final_scores

    def log_to_wandb(self, scores: List[float]) -> None:
        """Log scores to wandb with base and bonus components."""
        try:
            _wandb = get_wandb()
            if not _wandb or not _wandb.run:
                return

            wandb_data = dict({
                f"train/rewards/{self.name}": np.mean(scores) if scores else 0,
                f"train/rewards/{self.name}_base": np.mean(self._normalized_base_scores),
                f"train/rewards/{self.name}_bonus": np.mean(self._normalized_bonus_scores)
            })

            for reward_fn in self.base_rewards:
                if reward_fn.name not in self._base_values:
                    continue

                wandb_data[f"train/rewards/components/{self.name}/{reward_fn.name}"] =\
                    np.mean(self._base_values[reward_fn.name])

            for reward_fn in self.bonus_rewards:
                if reward_fn.name not in self._base_values:
                    continue

                wandb_data[f"train/rewards/components/{self.name}/{reward_fn.name}"] = \
                    np.mean(self._bonus_values[reward_fn.name]) if self._bonus_values[reward_fn.name] else 0
                wandb_data[f"train/rewards/components/{self.name}/{reward_fn.name}_application_rate"] = \
                    np.mean([1 if reward_fn.name in applied else 0 for applied in self._applied_bonuses]) \
                    if self._applied_bonuses else 0

            _wandb.log(wandb_data)
        except Exception as e:
            logging.warning(f"Failed to log to wandb: {e}")
