"""
Reward Framework: A composable framework for reward functions in reinforcement learning from human feedback.
"""

__version__ = "0.1.1"

from .rewards import (
    RewardFunction,
    Qualifier,
    timed_execution,
    master_reward,
    get_wandb,
)
from .multiturn_reward import MultiTurnRewardFunction, MultiTurnInput, Message

from .qualified_reward import QualifiedReward
from .scaled_reward import ScaledReward
from .llm_reward import LLMReward
from .composite_reward import CompositeReward
from .blacklist_qualifier import BlacklistQualifier
from .ngram_blacklist_qualifier import NgramBlacklistQualifier
from .async_reward import AsyncReward
from .async_llm_reward import AsyncLLMReward
from .multiturn_reward_wrapper import MultiTurnRewardWrapper

from .qualifiers import (
    Qualifier,
    QualifierInput,
    QualifierFn,
    ThresholdQualifier,
    AllQualifier,
    AnyQualifier,
    NoneQualifier,
)

__all__ = [
    "RewardFunction",
    "ThresholdQualifier",
    "master_reward",
    "MultiTurnRewardFunction",
    "MultiTurnRewardWrapper",
    "MultiTurnInput",
    "Message",
    "Qualifier",
    "QualifierInput",
    "QualifierFn",
    "AllQualifier",
    "AnyQualifier",
    "NoneQualifier",
    "QualifiedReward",
    "ScaledReward",
    "LLMReward",
    "CompositeReward",
    "NgramBlacklistQualifier",
    "AsyncReward",
    "AsyncLLMReward",
    "timed_execution",
    "get_wandb",
]
