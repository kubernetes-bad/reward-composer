from unittest.mock import patch, MagicMock

from reward_composer import (
    RewardFunction,
    master_reward,
    timed_execution,
    get_wandb,
    Qualifier, QualifierInput,
)


class DummyRewardFunction(RewardFunction):
    def __init__(self, factor: float = 1.0, name: str = None, weight: float = 1.0):
        super().__init__(name=name, weight=weight)
        self.factor = factor

    @timed_execution
    def __call__(self, completions, prompts, **kwargs):
        return [len(c) * self.factor for c in completions]


class PassingQualifier(Qualifier):
    def __call__(self, completion: str, context: QualifierInput = None) -> bool:
        return True


class FailingQualifier(Qualifier):
    def __call__(self, completion: str, context: QualifierInput = None) -> bool:
        return False


def test_timed_execution():
    dummy = DummyRewardFunction()
    assert dummy.last_execution_time is None

    completions = ["Test", "Teest"]
    prompts = ["prompty"] * len(completions)

    dummy(completions, prompts)
    assert dummy.last_execution_time is not None
    assert dummy.last_execution_time >= 0


def test_reward_function_basic():
    dummy = DummyRewardFunction(factor=2.0)
    completions = ["Test", "Teest"]
    prompts = ["prompty"] * len(completions)
    scores = dummy(completions, prompts)
    # "Test" -> length=4 * factor=2 = 8, "Teest" -> length=5 * factor=2 = 10
    assert scores == [8, 10]


def test_with_weight():
    dummy = DummyRewardFunction(factor=1.0, weight=1.0)
    assert dummy.weight == 1.0

    dummy.with_weight(2.5)
    assert dummy.weight == 2.5


def test_master_reward_no_functions():
    completions = ["Test", "Teest"]
    prompts = ["prompty"] * len(completions)
    scores = master_reward(completions, prompts, reward_functions=[])
    assert scores == [0.0, 0.0]


def test_master_reward_single_function():
    dummy = DummyRewardFunction(factor=1.0, weight=2.0)
    completions = ["T", "ee"]
    prompts = ["prompty"] * len(completions)

    # weight 2.0 -> [ length("T")*1.0*2, length("ee")*1.0*2 ] -> [2,4]
    scores = master_reward(
        completions,
        prompts,
        reward_functions=[dummy],
        log_to_wandb=False,
    )
    assert scores == [2.0, 4.0]


def test_master_reward_multiple_functions():
    dummy1 = DummyRewardFunction(name='f1w1', factor=1.0, weight=1.0)
    dummy2 = DummyRewardFunction(name='f2w05', factor=2.0, weight=0.5)
    completions = ["ABC", "Hey"]
    prompts = ["prompty"] * len(completions)
    # dummy1: [3, 3]
    # dummy2: [6, 6]
    # weighted -> dummy1: [3*1.0=3, 3*1.0=3],
    #             dummy2: [6*0.5=3, 6*0.5=3].
    # summed = [3+3=6, 3+3=6].
    scores = master_reward(
        completions,
        prompts,
        reward_functions=[dummy1, dummy2],
        log_to_wandb=False,
    )
    assert scores == [6, 6]


def test_master_reward_with_global_qualifier_pass():
    dummy = DummyRewardFunction(factor=2.0, weight=1.0)
    completions = ["Test", "Teest"]
    prompts = ["prompty"] * len(completions)
    qualifier = PassingQualifier()

    # lengths: 4, 5 -> factor=2 -> 8, 10
    scores = master_reward(
        completions,
        prompts,
        reward_functions=[dummy],
        global_qualifier=qualifier,
        fallback_value=-1.0,
        log_to_wandb=False
    )
    assert scores == [8.0, 10.0]


def test_master_reward_with_global_qualifier_fail():
    dummy = DummyRewardFunction(factor=2.0, weight=1.0)
    completions = ["Test", "Teest"]
    prompts = ["prompty"] * len(completions)
    qualifier = FailingQualifier()

    scores = master_reward(
        completions,
        prompts,
        reward_functions=[dummy],
        global_qualifier=qualifier,
        fallback_value=-1.0,
        log_to_wandb=False,
    )
    assert scores == [-1.0, -1.0]


@patch("reward_composer.rewards.get_wandb")
def test_log_to_wandb_with_run(mock_get_wandb):
    mock_wandb = MagicMock()
    mock_wandb.run = True
    mock_get_wandb.return_value = mock_wandb

    dummy = DummyRewardFunction()
    completions = ["Test"]
    prompts = ["prompty"]
    master_reward(completions, prompts, reward_functions=[dummy], log_to_wandb=True)

    assert mock_wandb.log.called


@patch("reward_composer.rewards.get_wandb")
def test_log_to_wandb_no_run(mock_get_wandb):
    mock_wandb = MagicMock()
    mock_wandb.run = None  # no run
    mock_get_wandb.return_value = mock_wandb

    dummy = DummyRewardFunction()
    completions = ["Test"]
    prompts = ["prompty"]
    master_reward(completions, prompts, reward_functions=[dummy], log_to_wandb=True)

    mock_wandb.log.assert_not_called()


def test_get_wandb_importerror():
    with patch("builtins.__import__", side_effect=ImportError):
        wandb_module = get_wandb()
        assert wandb_module is None
