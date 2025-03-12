# Reward Composer
Reward function design building blocks, compatible with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is this thing?
Reward Composer is a collection of simple building blocks for making your perfect reward function for Reinforcement Learning training of language models.

It enables researchers to build complex evaluation criteria by combining simple re-usable components in a declarative way.
It is useful for reinforcement learning from human feedback (RLHF), model evaluation, and automated text quality assessment.

It's like Lego for GRPO.

## Features
- Composable reward functions: Build complex evaluation criteria from simple building blocks
- Flexible qualifiers: Apply rewards conditionally based on qualification logic
- Non-linear scaling: Transform reward scores with various scaling functions
- Weights & Biases integration: Automatic logging of individual reward function scores and full completions
- Tracking performance: log individual reward function wall-clock times to WandB
- Extensible: Easily add new reward functions and qualifiers

## How to use

First, write your main reward function:

> See [example_usage.py](example_usage.py)

In Axolotl, add your main reward function as the only reward function:
```yaml
rl: grpo
trl:
  beta: 0.001
  # ...
  reward_funcs:
    - your_reward_file_name_without_extension.your_main_reward_function_name
```

## Core Components

Reward Composer is built around a few key abstractions. Those enable flexible composition of your evaluation criteria.


#### Reward Function
The `RewardFunction` is the base unit of evaluation. It takes model outputs and their corresponding prompts and returns a score for each completion.

```python
class RewardFunction:
    def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        # evaluate completions and return scores
        pass
```

### Qualifiers
`Qualifier`s determine whether a reward should be applied based on certain conditions. They act as filters or gates for rewards.

```python
class Qualifier:
    def __call__(self, completion: str, context: Dict[str, Any]) -> bool:
        # return True if the completion qualifies, False otherwise
        pass
```

## Reward Composition

### LLMReward
`LLMReward` calls any OpenAI compatible API to grade completions. Supports retries, ensemble scoring (more than 1 grading per completion) and custom response parsing logic.
```python
reward = LLMReward(
    name="my_llm_reward",
    api_url="https://api.x.ai/v1/chat/completions",
    api_key=['api-key-1', 'api-key-2'], # will round robin your keys
    model='grok-3',
    temperature=0.6,
    top_p=0.9,
    max_tokens=8192,
    system_prompt=MY_SYSTEM_PROMPT,
    prompt_template=MY_USER_PROMPT,
    max_concurrent_requests=8,
    output_parser=my_llm_response_parser, # takes a str, returns a float
    ensemble_size=3, # how many times to rate each completion
    ensemble_aggregation="median",  # or "mean", "mode", "min", "max",
)
```


### QualifiedReward
`QualifiedReward` sets the reward value of the underlying reward function only if a qualifier returns True, otherwise it returns a fallback value (typically 0.0).

```python
reward = QualifiedReward(
    reward_fn=my_reward,
    qualifier=my_qualifier,
    fallback_value=0.0,
)
```

### CompositeReward
`CompositeReward` combines multiple reward functions, with optional qualifiers for bonus rewards.

```python
my_composite_reward = CompositeReward(
    base_rewards=[reward1, reward2],
    bonus_rewards=[
        (reward3, qualifier1), # will apply only when qualifier returns True
        reward4, # or, it doesn't have to be conditional
    ],
    base_max=0.8, # portion of total reward that's dedicated to base
    total_max=1.0,
)
```

### ScaledReward
`ScaledReward` applies a non-linear transformation to a reward function's output, allowing for more nuanced scoring.

```python
reward = ScaledReward(
    reward_fn=my_reward,
    scaling=ScalingFunction.EXPONENTIAL, # or QUADRATIC, CUBIC, SIGMOID, ...
    k=10.0,  # controls steepness of the scaling
)
```

## Qualifier Types

### BlacklistQualifier
`BlacklistQualifier` disqualifies completions containing any blacklisted words or phrases.

```python
qualifier = BlacklistQualifier(
    blacklist_path='path/to/your/bad_words_array.json',
    case_sensitive=False,
)
```

### NgramBlacklistQualifier
`NgramBlacklistQualifier` disqualifies completions containing specific n-grams, after removing stop words and punctuation.
```python
qualifier = NgramBlacklistQualifier(
    blacklist_path='path/to/your/blacklist_array.json',
    n_min=2,
    n_max=4,
)
```

### Threshold Qualifier
Returns True only a specific reward function score is above the threshold.

```python
qualifier = ThresholdQualifier(
    reward_name=some_reward.name,
    threshold=0.7,
)
```

### Logic Qualifiers
- AllQualifier (AND): Requires all child qualifiers to return True.
- AnyQualifier (OR): Requires at least one child qualifier to return True.
- NoneQualifier (NOT): Requires that not a single child qualifier to return True. 
```python
all_qualifier = AllQualifier([qualifier1, qualifier2, qualifier3])
any_qualifier = AnyQualifier([qualifier1, qualifier2, qualifier3])
none_qualifier = NoneQualifier([qualifier1, qualifier2, qualifier3])
```

## Example Reward Functions
Reward Composer also includes several built-in example reward functions.

### LengthReward
`LengthReward` scores completions based on their closeness to ideal length, with configurable plateau region for reward stability. 
```python
reward = LengthReward(
    target_length=1500,
    plateau=200,
)
```

### LLMJudge
`LLMJudge` wraps `LLMReward` and demonstrates how to preprocess prompts for templating, and use custom output parsers.

```python
llm_judge_reward = LLMJudge(
    api_url="https://api.x.ai/v1/chat/completions",
    api_key=['api-key-1', 'api-key-2'], # will round robin your keys
    model='grok-3',
    max_concurrent_requests=2,
)
```

### RepetitionReward
`RepetitionReward` penalizes completions that copy long chunks of text from the prompt.

```python
repetition_reward = RepetitionReward(max_copied_words=5)
```

## Customization

You can make your own reward functions and classifiers that work for your specific reward design.
With Reward Composer, it's relatively straightforward:

### Making a Custom Reward Function

```python
from rewards import RewardFunction, timed_execution

class MyCustomReward(RewardFunction):
    def __init__(self, param1: float = 0.5, weight: float = 1.0):
        super().__init__(name="my_custom_reward", weight=weight)
        self.param1 = param1

    @timed_execution # add this to capture reward execution time per training step in WandB
    def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        scores = []
        for completion in completions:
            # your custom scoring logic here
            score = len(completion) / 1000  # Example: score based on length
            scores.append(min(score, 1.0))  # Cap at 1.0
        return scores
```

### Creating a Custom Qualifier

```python
from qualifiers import Qualifier, QualifierInput

class MyCustomQualifier(Qualifier):
    def __init__(self, has_to_include_this_word: str):
        super().__init__(name="my_custom_qualifier")
        self.has_to_include_this_word = has_to_include_this_word
    
    def __call__(self, completion: str, context: QualifierInput = None) -> bool:
        # your custom qualification logic here
        return self.has_to_include_this_word in completion
```

## Weights & Biases Integration

Reward Composer automatically logs completions and their individual reward scores to Weights & Biases if it's available:

```python
import wandb
wandb.init(project="my_rlhf_project")

scores = master_reward(
    completions=completions,
    prompts=prompts,
    reward_functions=[reward_fn],
    log_to_wandb=True  # Enable wandb logging
)
```

This will log:
- Individual reward function scores
- Execution time for each reward function
- A table of completions with all reward scores
- The final combined reward score

## Citation

If you use Reward Composer in your research, please cite:

```
@software{kubernetesbad_rewardcomposer,
  author = {Kubernetes Bad},
  title = {Reward Composer: A Framework for Composable Reward Functions},
  year = {2025},
  url = {https://github.com/kubernetes-bad/reward-composer}
}
```

