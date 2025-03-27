import re
from typing import List, Union

from reward_composer import RewardFunction, timed_execution

GRADE_GOOD = 1.0
GRADE_BAD = 0.0

class RepetitionReward(RewardFunction):
    """
    Reward that promotes creativity by penalizing spans of words copied from the prompt.
    """

    def __init__(self, max_copied_words: int, weight: float = 1.0):
        super().__init__(name="repetition_reward", weight=weight)
        self.max_copied_words = max_copied_words

    @timed_execution
    def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        name: Union[List[str], None] = kwargs['name']
        rewards = []

        for idx, (completion, prompt) in enumerate(zip(completions, prompts)):
            current_name = name[idx] if name is not None else None
            prompt_lower = prompt.lower()
            completion_lower = completion.lower()

            if name is not None:
                name_pattern = re.escape(current_name.lower())
                placeholder = "NAME_PLACEHOLDER" # we don't want to penalize "Prince Ludwig The Third"
                prompt_processed = re.sub(name_pattern, placeholder, prompt_lower)
                completion_processed = re.sub(name_pattern, placeholder, completion_lower)
            else:
                prompt_processed = prompt_lower
                completion_processed = completion_lower

            prompt_words = prompt_processed.split()
            completion_words = completion_processed.split()

            max_copied_sequence = RepetitionReward._find_max_copied_sequence(prompt_words, completion_words)

            reward = GRADE_GOOD if max_copied_sequence <= self.max_copied_words else GRADE_BAD
            rewards.append(reward)

        return rewards

    @staticmethod
    def _find_max_copied_sequence(prompt_words, completion_words):
        prompt_sequences = {} # all possible sequences
        for length in range(1, len(prompt_words) + 1):
            for i in range(len(prompt_words) - length + 1):
                sequence = tuple(prompt_words[i:i+length])
                prompt_sequences[sequence] = length

        max_length = 0
        for length in range(min(len(completion_words), max(prompt_sequences.values())), 0, -1):
            if max_length >= length:
                break
            for i in range(len(completion_words) - length + 1):
                sequence = tuple(completion_words[i:i+length])
                if sequence in prompt_sequences:
                    max_length = max(max_length, length)
                    break

        return max_length
