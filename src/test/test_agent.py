import numpy as np
import json
import os
import sys
import copy
import pandas as pd
import pytest
sys.path.insert(1, "src/utils/")
from agent import Environment
from Q_learning import QLearningTrainer
sys.path.insert(1, os.getcwd())
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@pytest.fixture
def rl_env():
    json_config = {
        "states_variables": ["pos_y", "acceleration_y", "speed_y", "angle"],
        "agent_variables": ["booster"],
        "initial_values": {
            "pos_y": [175.0],
            "acceleration_y": [0.0],
            "speed_y": [0.0],
            "angle": [0.0],
            "booster": [0.0],
            "m_fuel": [100],
            "futur_pos_y": [175.0],
            "weight_rocket": [105],
            "weight_dry_rocket": [5],
            "G": [1.62],
            "m_fuel_ini": [100.0],
            "pos_y_star": [0.0]
        },
        "_limit": ["min", "max", "n_bins"],
        "limit": {
            "pos_y": [0.0, 300.0, 61],
            "acceleration_y": [-20.0, 20.0, 21],
            "speed_y": [-50.0, 50.0, 21],
            "angle" : [-0.8, 0.8, 3],
            "booster": [0.0, 1.0, 3],
            "m_fuel": [0.0, 100.0, 101]
        },
        "n_action": {
            "booster": {"0": 0.0, "1": 0.5, "2": 1.0}
        },
        "action_to_take": {
            "booster": {"$booster$": "$action$"}
        },
        "equations_variables": {
            "$F$": "600",
            "$m_fuel$": "$m_fuel$ - $booster$ *10 -$angle$ *10",
            "$weight_rocket$": "$weight_dry_rocket$ + $m_fuel$",
            "dt": "0.5",
            "$theta$": "0.0",
            "$y_0$": "$pos_y$",
            "$Vy_0$": "$speed_y$",
            "$acceleration_y$": "($F$/(5+$weight_rocket$) * np.cos($angle$)) * $booster$ - $G$",
            "$speed_y$": "($F$/(5+$weight_rocket$) * np.cos($angle$)) * $booster$ * $dt$ - $G$ * $dt$ + $Vy_0$",
            "$pos_y$": "(0.5 * $F$/(5+$weight_rocket$) * np.cos($angle$)) * $booster$ * $dt$**2 - $G$ * $dt$**2 + $Vy_0$ * $dt$ + $y_0$",
            "$futur_pos_y$": "$pos_y$ + 3 * $speed_y$"
        },
        "equations_rewards": {
            "$booster$": "-($pos_y$ - $pos_y_star$)**2 + $m_fuel$/$m_fuel_ini$"
        },
        "stop_episode": {
            "pos_y": [0, 1],
            "acceleration_y": [-2, 2],
            "speed_y": [-2, 2]
        },
        "condition_stop_episode": "all"
    }
    return Environment(json_config)

def test_q_learning_train_short_run(rl_env):
    trainer = QLearningTrainer(env=rl_env, num_episodes=3, run_limit=100, convergence_criterion=0.01)
    trainer.q_learning()

    assert not trainer.q_table.empty
    assert isinstance(trainer.q_table.columns, pd.MultiIndex)
    assert len(trainer.states_for_all_episodes) > 0
    assert len(trainer.q_table_for_all_episodes) > 0
    assert trainer.q_table.to_numpy().sum() != 0

def test_reaches_goal_or_stops(rl_env):
    trainer = QLearningTrainer(env=rl_env, num_episodes=2, run_limit=200)
    trainer.q_learning()
    # Either the training reaches a goal or stops due to run_limit
    assert len(trainer.loss_episodes) > 0

def test_state_tracking_consistency(rl_env):
    trainer = QLearningTrainer(env=rl_env, num_episodes=2)
    trainer.q_learning()
    for state_log in trainer.states_for_all_episodes:
        assert "pos_y" in state_log.columns
        assert "reward_booster" in state_log.columns
