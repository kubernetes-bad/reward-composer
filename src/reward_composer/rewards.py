import functools
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, TypeVar
import logging

import numpy as np

from .qualifiers import Qualifier

T = TypeVar('T', bound='RewardFunction')

@functools.cache
def get_wandb():
    try:
        import wandb
        return wandb
    except ImportError:
        logging.warning(f"wandb not installed. Skipping logging for reward functions.")
        return None

def timed_execution(method):
    """
    Decorator to measure execution time of a reward function call.
    Stores the execution time in the instance's last_execution_time attribute.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = method(self, *args, **kwargs)
        self.last_execution_time = time.time() - start_time
        return result
    return wrapper


class RewardFunction(ABC):
    def __init__(self, name: str = None, weight: float = 1.0):
        self.name = name or self.__class__.__name__
        self.weight = weight
        self.last_execution_time = None

    @abstractmethod
    def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        """Calculate reward scores for completions"""
        pass

    def log_to_wandb(self, scores: List[float]) -> None:
        try:
            _wandb = get_wandb()
            if not _wandb or not _wandb.run:
                return

            log_dict = {f"train/rewards/{self.name}": np.mean(scores)}

            if self.last_execution_time is not None:
                log_dict[f"perf/rewards/{self.name}_time"] = self.last_execution_time

            _wandb.log(log_dict)

        except Exception as e:
            logging.warning(f"Failed to log to wandb: {e}")

    def with_weight(self: T, weight: float) -> T:
        self.weight = weight
        return self

def master_reward(
        completions: List[str],
        prompts: List[str],
        reward_functions: List[RewardFunction],
        log_to_wandb: bool = True,
        global_qualifier: Optional[Qualifier] = None,
        fallback_value: float = 0.0,
        **kwargs) -> List[float]:
    """
    Master reward function that combines all other reward functions and qualifiers.

    Args:
        :param completions: List of model outputs to evaluate
        :param prompts: List of prompts used to generate completions
        :param reward_functions: List of reward functions to apply
        :param log_to_wandb: Whether to log results to wandb
        :param global_qualifier: If set, whenever this qualifier returns False - total reward is set to `fallback_value`
        :param fallback_value Reward value to return when global qualifier conditions are not met
        **kwargs: All other arguments to pass to reward functions

    Returns:
        :return List of final reward scores
    """
    if log_to_wandb:
        import pandas as pd # needed for completion logging

    all_scores: Dict[str, List[float]] = {}
    all_scores_weighted: Dict[str, List[float]] = {}

    for reward_fn in reward_functions:
        scores = reward_fn(completions, prompts, **kwargs)
        all_scores[reward_fn.name] = scores
        all_scores_weighted[reward_fn.name] = [score * reward_fn.weight for score in scores]

        if log_to_wandb:
            reward_fn.log_to_wandb(scores)

    final_scores: List[float] = []
    for completion_index in range(len(completions)):
        context = { # global qualifier context
            "prompt": prompts[completion_index] if completion_index < len(prompts) else "",
            **{name: values[completion_index] for name, values in all_scores.items()},
            **kwargs,
        }
        if global_qualifier and not global_qualifier(completions[completion_index], context):
            final_scores.append(fallback_value)
        else:
            final_score = sum(scores[completion_index] for scores in all_scores_weighted.values())
            final_scores.append(final_score)

    # log completions with each function raw scores
    if log_to_wandb:
        try:
            _wandb = get_wandb()
            if _wandb is not None and _wandb.run is not None:
                table_data = {
                    "prompt": prompts,
                    "completion": completions,
                    "reward": final_scores,
                }
                for reward_name, scores in all_scores.items(): # NOT WEIGHTED
                    table_data[reward_name] = scores
                df = pd.DataFrame(table_data)
                _wandb.log({
                    "reward_completions": _wandb.Table(dataframe=df),
                    "train/rewards/total_reward_score": np.mean(final_scores), # WEIGHTED
                })
        except Exception as e:
            logging.warning(f"Failed to log to wandb: {e}")

    return final_scores
