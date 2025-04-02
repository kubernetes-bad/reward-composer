import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Union

from .rewards import RewardFunction, timed_execution, MultiTurnInput, Message

logger = logging.getLogger("AsyncReward")


class AsyncReward(RewardFunction, ABC):
    def __init__(self, name: str, weight: float = 1.0, default_score: float = 0.5):
        super().__init__(name=name, weight=weight)
        self.default_score = default_score

    @timed_execution
    def __call__(self,
        completions: Union[MultiTurnInput, List[str]],
        prompts: Union[MultiTurnInput, List[str]],
        **kwargs,
    ) -> List[float]:
        try:
            return asyncio.run(self._evaluate_batch(completions, prompts, **kwargs))
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                raise RuntimeError(
                    "Cannot call AsyncReward synchronously from an async context. "
                    "Use 'await reward.evaluate_async(...)' instead."
                ) from e
            raise
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return [self.default_score] * len(completions)

    async def evaluate_async(self,
        completions: List[Union[List[Message], str]],
        prompts: List[Union[List[Message], str]],
        **kwargs,
    ) -> List[float]:
        """Async interface - alternative entrypoint to use in async context"""
        try:
            return await self._evaluate_batch(completions, prompts, **kwargs)
        except Exception as e:
            logger.error(f"Async evaluation failed: {e}")
            return [self.default_score] * len(completions)

    async def _evaluate_batch(self,
        completions: List[Union[List[Message], str]],
        prompts: List[Union[List[Message], str]],
        **kwargs,
    ) -> List[float]:
        """Same as __call__, but async"""
        tasks = [
            self.evaluate_one(completion=completion, prompt=prompt, **kwargs)
            for completion, prompt
            in zip(completions, prompts)
        ]

        scores = await asyncio.gather(*tasks, return_exceptions=True)
        for i, score in enumerate(scores):  # replace exceptions with default score
            if isinstance(score, Exception):
                logger.warning(f"Exception while evaluating item {i} in batch: {str(score)}")
                scores[i] = self.default_score

        return scores

    @abstractmethod
    async def evaluate_one(self, completion: Union[List[Message], str], prompt: Union[List[Message], str], **kwargs) -> float:
        """Score a single pair of (completion, prompt).
        If exception is raised during evaluation, default score is used instead."""
        raise NotImplementedError
