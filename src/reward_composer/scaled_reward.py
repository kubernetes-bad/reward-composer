import math
from enum import Enum, auto
from typing import Callable, Optional, List

from .rewards import RewardFunction, timed_execution


class ScalingFunction(Enum):
    """Predefined scaling functions for reward transformation."""
    LINEAR = auto()           # No change: f(x) = x
    QUADRATIC = auto()        # f(x) = x²
    CUBIC = auto()            # f(x) = x³
    EXPONENTIAL = auto()      # f(x) = e^(k*(x-1))
    SIGMOID = auto()          # f(x) = 1/(1+e^(-k*(x-0.5)))
    STEP = auto()             # f(x) = 0 if x < threshold else 1
    SQRT = auto()             # f(x) = sqrt(x)
    LOG = auto()              # f(x) = log(1 + (e-1)*x)/log(e)


class ScaledReward(RewardFunction):
    """
    Wrapper that rescales the output of another reward function.

    This allows for non-linear transformations of reward scores, such as:
    - Exponential scaling to heavily penalize low scores
    - Sigmoid scaling to create a soft threshold
    - Step function to create a hard threshold
    - Custom scaling functions
    """

    def __init__(
        self,
        reward: RewardFunction,
        scaling: ScalingFunction = ScalingFunction.LINEAR,
        custom_scaling_fn: Optional[Callable[[float], float]] = None,
        k: float = 10.0, # FOR scaling=SIGMOID or EXPONENTIAL
        threshold: float = 0.5,  # FOR scaling=STEP
        name: str = None,
        weight: float = 1.0,
    ):
        """
        Initialize a scaled reward function.

        Args:
            :param reward: The base reward function to scale
            :param scaling: Predefined scaling function to use
            :param custom_scaling_fn: Custom scaling function (takes a float, returns a float)
            :param k: Parameter for exponential and sigmoid scaling (controls steepness)
            :param threshold: Parameter for step function (cutoff point)
            :param name: Name of this reward function
            :param weight: Weight of this reward function
        """
        name = name or f"scaled_{reward.name}"
        super().__init__(name=name, weight=weight)
        self.reward_fn = reward
        self.scaling = scaling
        self.custom_scaling_fn = custom_scaling_fn
        self.k = k
        self.threshold = threshold

    def _apply_scaling(self, value: float) -> float:
        if self.custom_scaling_fn:
            return self.custom_scaling_fn(value)

        if self.scaling == ScalingFunction.LINEAR:
            return value
        elif self.scaling == ScalingFunction.QUADRATIC:
            return value ** 2
        elif self.scaling == ScalingFunction.CUBIC:
            return value ** 3
        elif self.scaling == ScalingFunction.EXPONENTIAL:
            # exponential: e^(k*(x-1))
            # gives 1.0 when x=1, and drops quickly as x decreases
            return math.exp(self.k * (value - 1))
        elif self.scaling == ScalingFunction.SIGMOID:
            # sigmoid: 1/(1+e^(-k*(x-0.5)))
            # creates an S-curve centered at x=0.5
            return 1 / (1 + math.exp(-self.k * (value - 0.5)))
        elif self.scaling == ScalingFunction.STEP:
            # maps values to either 0 or 1
            return 0.0 if value < self.threshold else 1.0
        elif self.scaling == ScalingFunction.SQRT:
            # sqrt(x)
            # brings low values up but keeps high values relatively same
            return math.sqrt(value)
        elif self.scaling == ScalingFunction.LOG:
            # logarithmic scaling: log(1 + (e-1)*x)/log(e)
            return math.log(1 + (math.e - 1) * value) / math.log(math.e)
        else:
            return value # default linear

    @timed_execution
    def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        base_scores = self.reward_fn(completions, prompts, **kwargs)
        return [self._apply_scaling(score) for score in base_scores]
