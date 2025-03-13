"""
Reward Framework: A composable framework for reward functions in reinforcement learning from human feedback.
"""

__version__ = "0.1.0"

from .rewards import (
    RewardFunction,
    Qualifier,
    timed_execution,
    master_reward,
    Message,
    get_wandb,
)

from .qualified_reward import QualifiedReward
from .scaled_reward import ScaledReward
from .llm_reward import LLMReward
from .composite_reward import CompositeReward
from .blacklist_qualifier import BlacklistQualifier
from .ngram_blacklist_qualifier import NgramBlacklistQualifier

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
    "Qualifier",
    "ThresholdQualifier",
    "master_reward",
    "Message",
    "Qualifier",
    "QualifierInput",
    "QualifierFn",
    "ThresholdQualifier",
    "timed_execution",
    "get_wandb",
]
