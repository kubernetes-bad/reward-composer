from typing import List, Optional, Callable

from reward_composer import MultiTurnRewardFunction, MultiTurnInput, RewardFunction, Message, timed_execution


class MultiTurnRewardWrapper(MultiTurnRewardFunction):
    """
    Makes a single-turn reward accept multi-turn inputs.
    Provide prompt and completion parsers to convert list of messages to a string.
    """
    def __init__(self,
        single_turn_reward_function: RewardFunction,
        name: Optional[str] = None,
        weight: float = 1.0,
        prompt_parser: Optional[Callable[[List[Message]], str]] = None,
        completion_parser: Optional[Callable[[List[Message]], str]] = None,
    ):
        self.reward_fn = single_turn_reward_function
        self.prompt_parser = prompt_parser if prompt_parser is not None else self._default_turn_parser
        self.completion_parser = completion_parser if completion_parser is not None else self._default_turn_parser

        super().__init__(
            name=name if name is not None else single_turn_reward_function.name,
            weight=weight,
        )

    @staticmethod
    def _default_turn_parser(turns: List[Message]) -> str:
        """just extracts content from last message"""
        if len(turns) == 0:
            return "" # it's an empty convo bro
        return turns[-1].content

    @timed_execution
    def __call__(self, completions: MultiTurnInput, prompts: MultiTurnInput, **kwargs) -> List[float]:
        preprocessed_completions = []
        preprocessed_turns = []

        for completion_turns, prompt_turns in zip(completions, prompts):
            completion = self.completion_parser(completion_turns)
            prompt = self.prompt_parser(prompt_turns)
            preprocessed_completions.append(completion)
            preprocessed_turns.append(prompt)

        return self.reward_fn(completions=preprocessed_completions, prompts=preprocessed_turns, **kwargs)

