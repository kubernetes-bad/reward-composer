from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List

QualifierInput = Dict[str, Any]
QualifierFn = Callable[[str, QualifierInput], bool]


class Qualifier(ABC):
    """Qualifiers determine if rewards should be applied."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def __call__(self, completion: str, context: QualifierInput = None) -> bool:
        """
        Determine if a completion meets some qualification criteria.

        Args:
            :param completion: The text completion to evaluate
            :param context: Additional context like reward values, prompt, etc.

        Returns:
            :return True if the completion qualifies, False otherwise
        """
        pass


class ThresholdQualifier(Qualifier):
    """Qualifier that checks if a specific reward value exceeds a threshold."""

    def __init__(self, reward_name: str, threshold: float, name: str = None):
        super().__init__(name=name or f"threshold_{reward_name}_{threshold}")
        self.reward_name = reward_name
        self.threshold = threshold

    def __call__(self, completion: str, context: QualifierInput = None) -> bool:
        context = context or {}
        reward_value = context.get(self.reward_name, 0.0)
        return reward_value > self.threshold


class AllQualifier(Qualifier):
    """Qualifier that checks if all other qualifiers return True."""
    def __init__(self, qualifiers: List[Qualifier], name: str = None):
        super().__init__(name=name or f'logic_and_{"_".join([q.name for q in qualifiers])}')
        self.qualifiers = qualifiers

    def __call__(self, completion: str, context: QualifierInput = None) -> bool:
        return all([q(completion, context) for q in self.qualifiers])


class AnyQualifier(Qualifier):
    """Qualifier that checks if any other qualifiers return True."""
    def __init__(self, qualifiers: List[Qualifier], name: str = None):
        super().__init__(name=name or f'logic_or_{"_".join([q.name for q in qualifiers])}')
        self.qualifiers = qualifiers

    def __call__(self, completion: str, context: QualifierInput = None) -> bool:
        return any([q(completion, context) for q in self.qualifiers])

class NoneQualifier(Qualifier):
    """Qualifier that checks if none of the other qualifiers return True."""
    def __init__(self, qualifiers: List[Qualifier], name: str = None):
        super().__init__(name=name or f'logic_or_{"_".join([q.name for q in qualifiers])}')
        self.qualifiers = qualifiers

    def __call__(self, completion: str, context: QualifierInput = None) -> bool:
        return not any([q(completion, context) for q in self.qualifiers])
