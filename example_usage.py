from typing import List

from reward_composer import master_reward
from reward_composer.composite_reward import CompositeReward
from reward_composer.example_rewards import LengthReward, RepetitionReward, LLMJudge
from reward_composer.qualifiers import AllQualifier, ThresholdQualifier
from reward_composer.blacklist_qualifier import BlacklistQualifier
from reward_composer.ngram_blacklist_qualifier import NgramBlacklistQualifier

MODEL_NAME = "grok-2-1212"
REQUESTS_PER_KEY = 2
API_KEYS = [
    "sk-test-1",
    "sk-test-2",
    "sk-test-3",
    "sk-test-4",
]

# rewards
length_reward = LengthReward(target_length=2000, plateau=200, weight=1.3)
repetition_reward = RepetitionReward(max_copied_words=5)

composite_gimmick_reward = CompositeReward( # don't actually use this - this is for demonstration purposes only!
    base_rewards=[length_reward],
    bonus_rewards=[
        (repetition_reward, ThresholdQualifier(reward_name=length_reward.name, threshold=0.3)),  # unscaled!
    ],
    weight=0.6,
)

llm_judge_reward = LLMJudge(
    api_url="https://api.x.ai/v1/chat/completions",
    api_key=API_KEYS,
    model=MODEL_NAME,
    max_concurrent_requests=len(API_KEYS) * REQUESTS_PER_KEY,
    weight=1.5,
)

# qualifiers
phrase_slop_qualifier = BlacklistQualifier(blacklist_path='./data/slop.json')
ngram_slop_qualifier = NgramBlacklistQualifier(blacklist_path='./data/slop_ngram.json')
slop_qualifier = AllQualifier(
    qualifiers=[
        phrase_slop_qualifier,
        ngram_slop_qualifier,
    ]
)

llm_judge_qualifier = ThresholdQualifier(reward_name=llm_judge_reward.name, threshold=0.3)


# trainer entry point
def total_reward(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    return master_reward(
        completions=completions,
        prompts=prompts,
        reward_functions=[
            length_reward,
            repetition_reward,
            llm_judge_reward,
            composite_gimmick_reward,
        ],
        global_qualifier=AllQualifier(qualifiers=[
            slop_qualifier,
            llm_judge_qualifier,
        ]),
        **kwargs)
