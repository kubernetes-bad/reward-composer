import asyncio
import time

import pytest
from typing import List, Union
from unittest.mock import patch

from reward_composer import Message
from reward_composer.async_reward import AsyncReward

class SimpleAsyncReward(AsyncReward):
    def __init__(self, delay=0.1, fail_indices=None, name="simple", weight=1.0, default_score=0.5):
        super().__init__(name=name, weight=weight, default_score=default_score)
        self.delay = delay
        self.fail_indices = fail_indices or []
        self.call_count = 0

    async def evaluate_one(self, completion: Union[List[Message], str], prompt: Union[List[Message], str], **kwargs) -> float:
        await asyncio.sleep(self.delay)  # mocked

        current_index = self.call_count
        self.call_count += 1

        if current_index in self.fail_indices:
            raise ValueError(f"Simulated failure for index {current_index}")

        return self.default_score * 2


@pytest.fixture
def simple_reward():
    return SimpleAsyncReward()


@pytest.fixture
def failing_reward():  # will fail for indices 1 and 3
    return SimpleAsyncReward(fail_indices=[1, 3])


class TestAsyncReward:
    def test_does_not_explode(self, simple_reward):
        completions = ["Test", "Teest", "Teeest"]
        prompts = ["prompty"] * len(completions)

        scores = simple_reward(completions, prompts)

        assert len(scores) == 3

    def test_individual_failures(self, failing_reward):
        completions = ["Good", "Bad", "Good again", "Also bad"]
        prompts = ["prompty"] * len(completions)

        scores = failing_reward(completions, prompts, index=0)

        assert len(scores) == 4
        assert scores[0] != failing_reward.default_score
        assert scores[1] == failing_reward.default_score  # fail
        assert scores[2] != failing_reward.default_score
        assert scores[3] == failing_reward.default_score  # fail

    @pytest.mark.asyncio
    async def test_async_does_not_explode(self, simple_reward):
        completions = ["Test", "Teest"]
        prompts = ["prompty"] * len(completions)

        scores = await simple_reward.evaluate_async(completions, prompts)

        assert len(scores) == 2

    @pytest.mark.asyncio
    async def test_async_error_handling(self, failing_reward):
        completions = ["Good", "Bad", "Good again", "Also bad"]
        prompts = ["prompty"] * len(completions)

        scores = await failing_reward.evaluate_async(completions, prompts, index=0)

        assert len(scores) == 4
        assert scores[1] == failing_reward.default_score
        assert scores[3] == failing_reward.default_score

    def test_string_inputs(self, simple_reward):
        completions = ["Test", "Teest"]
        prompts = ["prompty"] * len(completions)

        scores = simple_reward(completions, prompts)
        assert len(scores) == 2

    def test_message_inputs(self, simple_reward):
        completions = [
            [Message(role="assistant", content="Test")],
            [Message(role="assistant", content="Teest")]
        ]
        prompts = [
            [Message(role="user", content="Please pass")],
            [Message(role="user", content="Let the tests be green")]
        ]

        scores = simple_reward(completions, prompts)
        assert len(scores) == 2

    def test_exception_in_evaluate_async(self, simple_reward):
        async def test():
            with patch.object(simple_reward, '_evaluate_batch', side_effect=Exception("Badaboom")):
                scores = await simple_reward.evaluate_async(["test"], ["prompt"])
                assert scores == [simple_reward.default_score]

        asyncio.run(test())

    @pytest.mark.asyncio
    async def test_synchronous_wrapper_runs_async(self):
        mock_reward = SimpleAsyncReward(name="test_sync_wrapper", weight=1.0, delay=1.0)
        completions = ["Test", "Teest", "Teeest"]
        prompts = ["prompty"] * len(completions)

        start_time = time.time()
        scores = await mock_reward.evaluate_async(completions, prompts)  # synchronous call
        end_time = time.time()
        elapsed = end_time - start_time

        assert len(scores) == 3
        assert elapsed < 2.0, (
            f"Expected concurrency via asyncio.run: total time ({elapsed:.2f}s) should be < 2s "
            "if executed in parallel."
        )
