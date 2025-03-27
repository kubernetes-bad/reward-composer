import pytest
from unittest.mock import MagicMock, patch

from typing import List
from reward_composer.qualifiers import Qualifier, QualifierInput
from reward_composer.rewards import RewardFunction
from reward_composer.composite_reward import CompositeReward


class MockReward(RewardFunction):
    def __init__(self, name=None, weight=1.0, return_values=None):
        super().__init__(name=name, weight=weight)
        self.return_values = return_values or []

    def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        scores = []
        for i in range(len(completions)):
            if i < len(self.return_values):
                scores.append(self.return_values[i])
            else:
                scores.append(0.0) # whatever, this is a test
        return scores


class MockQualifier(Qualifier):
    def __init__(self, should_qualify=True):
        super().__init__(name='mock_qualifier')
        self.should_qualify = should_qualify

    def __call__(self, completion: str, context: QualifierInput = None) -> bool:
        return self.should_qualify


@pytest.fixture
def simple_base_reward():
    return MockReward(name="base_mock", weight=1.0, return_values=[1, 2, 3])


@pytest.fixture
def simple_bonus_reward():
    return MockReward(name="bonus_mock", weight=1.0, return_values=[0.5, 0.6, 0.7])


def test_init_minimal(simple_base_reward):
    comp_reward = CompositeReward(
        base_rewards=[simple_base_reward],
        bonus_rewards=None,
        base_max=0.8,
        total_max=1.0,
        name="test_composite",
    )
    assert comp_reward.name == "test_composite"
    assert len(comp_reward.base_rewards) == 1
    assert comp_reward.bonus_rewards == []
    assert comp_reward.base_max == 0.8
    assert comp_reward.total_max == 1.0
    assert pytest.approx(comp_reward.bonus_max) == 0.2


def test_init_with_bonus(simple_base_reward, simple_bonus_reward):
    comp_reward = CompositeReward(
        base_rewards=[simple_base_reward],
        bonus_rewards=[simple_bonus_reward],
        base_max=0.5,
        total_max=1.0,
        name="test_composite_with_bonus",
    )
    assert comp_reward.name == "test_composite_with_bonus"
    assert len(comp_reward.base_rewards) == 1
    assert len(comp_reward.bonus_rewards) == 1
    assert comp_reward.base_max == 0.5
    assert comp_reward.total_max == 1.0
    assert comp_reward.bonus_max == 0.5


def test_reward_calculation_without_qualifiers():
    base_r1 = MockReward("base_r1", weight=1.0, return_values=[1, 2, 3])
    base_r2 = MockReward("base_r2", weight=2.0, return_values=[2, 2, 2])
    bonus_r1 = MockReward("bonus_r1", weight=1.0, return_values=[0.5, 1.0, 1.5])
    completions = ["c1", "c2", "c3"]
    prompts = ["prompty"] * len(completions)

    comp_reward = CompositeReward(
        base_rewards=[base_r1, base_r2],
        bonus_rewards=[bonus_r1],
        base_max=0.6,
        total_max=1.0,
    )

    # base sums by index:
    # index 0 -> base_r1(1)*1.0 + base_r2(2)*2.0 = 1 + 4 = 5
    # index 1 -> 2*1.0 + 2*2.0 = 2 + 4 = 6
    # index 2 -> 3*1.0 + 2*2.0 = 3 + 4 = 7
    # sum of weights = 1.0 + 2.0 = 3
    # normalized base[i] = (base_sum[i] / 3) * 0.6

    # bonus sums by index:
    # index 0 -> 0.5 * 1.0 = 0.5
    # index 1 -> 1.0 * 1.0 = 1.0
    # index 2 -> 1.5 * 1.0 = 1.5
    # sum of bonus weights = 1.0
    # normalized bonus[i] = (bonus_sum[i] / 1.0) * 0.4 (total_max=1.0, base_max=0.6 => bonus_max=0.4)

    # base sums: [5, 6, 7]; base avg: [5/3, 6/3, 7/3] => [1.666..., 2.0, 2.333...]
    # normalized_base: [1.666*0.6, 2.0*0.6, 2.333*0.6] => approx [1.0, 1.2, 1.4]
    # bonus sums: [0.5, 1.0, 1.5]; normalized_bonus: [0.5 * 0.4, 1.0 * 0.4, 1.5 * 0.4] => [0.2, 0.4, 0.6]

    scores = comp_reward(completions, prompts)
    assert pytest.approx(scores[0], 0.01) == 1.2
    assert pytest.approx(scores[1], 0.01) == 1.6
    assert pytest.approx(scores[2], 0.01) == 2.0


def test_base_qualifier_blocks_base_rewards(simple_base_reward):
    qualifier = MockQualifier(should_qualify=False)
    comp_reward = CompositeReward(
        base_rewards=[simple_base_reward],
        bonus_rewards=None,
        base_max=0.8,
        total_max=1.0,
        base_qualifier=qualifier,
    )
    completions = ["c1", "c2", "c3"]
    prompts = ["prompty"] * len(completions)

    scores = comp_reward(completions, prompts)
    for val in scores:
        assert val == 0.0


def test_bonus_qualifier_blocks_bonus_rewards(simple_base_reward, simple_bonus_reward):
    qualifier = MockQualifier(should_qualify=False)
    comp_reward = CompositeReward(
        base_rewards=[simple_base_reward],
        bonus_rewards=[(simple_bonus_reward, qualifier)],
        base_max=0.8,
        total_max=1.0,
    )
    completions = ["c1", "c2", "c3"]
    prompts = ["prompty"] * len(completions)

    # base scores = [1, 2, 3]
    # base_sum * 0.8 = [0.8, 1.6, 2.4]
    scores = comp_reward(completions, prompts)
    assert pytest.approx(scores[0], 0.001) == 0.8
    assert pytest.approx(scores[1], 0.001) == 1.6
    assert pytest.approx(scores[2], 0.001) == 2.4


def test_multiple_bonus_qualifiers_mixed():
    base_r = MockReward("base_r", weight=1.0, return_values=[5, 5, 5])
    bonus_r1 = MockReward("bonus_r1", weight=1.0, return_values=[1, 2, 3])
    bonus_r2 = MockReward("bonus_r2", weight=1.0, return_values=[5, 6, 7])
    qualifier_true = MockQualifier(True)
    qualifier_false = MockQualifier(False)

    comp = CompositeReward(
        base_rewards=[base_r],
        bonus_rewards=[
            (bonus_r1, qualifier_true),
            (bonus_r2, qualifier_false),
        ],
        base_max=0.8,
        total_max=1.0,
    )

    completions = ["A", "B", "C"]
    prompts = ["prompty"] * len(completions)
    scores = comp(completions, prompts)

    # base normalization = (base_sum / sum_of_weights) * base_max
    # (5/1) * 0.8 = 4.0

    # bonus part:
    # bonus_r1 => [1,2,3], qualifier_true => included [1,2,3]
    # bonus_r2 => [5,6,7], qualifier_false => excluded [0,0,0]
    # bonus_max = 0.2 (total_max - base_max)
    # normalized_bonus => [1,2,3] * 0.2 => [0.2, 0.4, 0.6]

    # finals = base + bonus = [4 + 0.2, 4 + 0.4, 4 + 0.6]

    assert pytest.approx(scores[0], 0.001) == 4.2
    assert pytest.approx(scores[1], 0.001) == 4.4
    assert pytest.approx(scores[2], 0.001) == 4.6


def test_no_base_rewards():
    bonus_r = MockReward("bonus_r", weight=1.0, return_values=[0.3, 0.3, 0.3])
    comp = CompositeReward(
        base_rewards=[],
        bonus_rewards=[bonus_r],
        base_max=0.8,
        total_max=1.0
    )
    completions = ["a", "b", "c"]
    prompts = ["prompty"] * len(completions)
    scores = comp(completions, prompts)
    # bonus sum = 0.3 => normalized by bonus_max(0.2) => 0.3 / 1.0 * 0.2 = 0.06 each
    for s in scores:
        assert pytest.approx(s, 0.001) == 0.06


@patch("reward_composer.composite_reward.get_wandb")
def test_log_to_wandb(mock_get_wandb, simple_base_reward, simple_bonus_reward):
    wandb_mock = MagicMock()
    wandb_mock.run = MagicMock()  # Ensure run attribute exists
    mock_get_wandb.return_value = wandb_mock

    comp = CompositeReward(
        base_rewards=[simple_base_reward],
        bonus_rewards=[simple_bonus_reward],
        name="comp_test",
    )
    completions = ["Test", "Teest", "Teeest"]
    prompts = ["prompty"] * len(completions)
    scores = comp(completions, prompts)
    comp.log_to_wandb(scores)

    mock_get_wandb.assert_called_once()
    wandb_mock.log.assert_called_once()

    logged_args = wandb_mock.log.call_args[0][0]
    assert f"train/rewards/{comp.name}" in logged_args
    assert f"train/rewards/{comp.name}_base" in logged_args
    assert f"train/rewards/{comp.name}_bonus" in logged_args
    assert f"train/rewards/components/{comp.name}/{simple_base_reward.name}" in logged_args
    assert f"train/rewards/components/{comp.name}/{simple_bonus_reward.name}" in logged_args
    assert f"train/rewards/components/{comp.name}/{simple_bonus_reward.name}_application_rate" in logged_args
