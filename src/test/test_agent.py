import numpy as np
import json
import os
import sys
import re
import copy
import pytest
sys.path.insert(1, "src/utils/")
from agent import Environment
sys.path.insert(1, os.getcwd())
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@pytest.fixture
def env_instance():
  json_file = {
      "states_variables" : ["pos_y", "acceleration_y", "speed_y", "angle"],
        "agent_variables" : ["booster"],
        "initial_values" : {
          "pos_y" : [175.0],
          "acceleration_y": [0.0],
          "speed_y": [0.0],
          "booster" : [0.0],
          "futur_pos_y" : [175.0],
          "m_fuel" : [100],
          "weight_rocket" : [105],
          "weight_dry_rocket" : [5],
          "G" : [1.62],
          "m_fuel_ini" : [100.0],
          "pos_y_star": [0.0],
          "angle" : [0]
          },
        "_limit" : ["min", "max", "n_bins"],
        "limit" : {
          "pos_y" : [0.0, 300.0, 61],
          "acceleration_y": [-20.0, 20.0, 21],
          "speed_y": [-50.0, 50.0, 21],
          "booster" : [0.0, 1.0, 3],
          "m_fuel" : [0.0, 100, 101]
        },
        "n_action" : {
          "booster": {"0" : 0.0, "1" : 0.5, "2" : 1.0}
        },
        "_action_to_take" : "How action change agent_variables, value is taken from n_action dictionary",
        "action_to_take" : {
          "booster": {"$booster$" : "$action$"}
        },
        "_equations" : "compute values for each equation. Variable are updated after the loop",
        "equations_variables": {
            "$F$" : "600",
            "$m_fuel$" : "$m_fuel$ - $booster$ *10 -$angle$ *10",
            "$weight_rocket$" : "$weight_dry_rocket$ + $m_fuel$",
            "dt" : "0.5",
            "$theta$" : "0.0",
            "$y_0$" : "$pos_y$",
            "$Vy_0$" : "$speed_y$",
            "$acceleration_y$" : "($F$/(5+$weight_rocket$) * np.cos($angle$)) * $booster$ - $G$",
            "$speed_y$" : "($F$/(5+$weight_rocket$) * np.cos($angle$)) * $booster$ * $dt$ - $G$ * $dt$ + $Vy_0$",
            "$pos_y$": "(0.5 * $F$/(5+$weight_rocket$) * np.cos($angle$)) * $booster$ * $dt$**2 - $G$ * $dt$**2 + $Vy_0$ * $dt$ + $y_0$",
            "$futur_pos_y$" : "$pos_y$ + 3 * $speed_y$",
        },
        "equations_rewards": {
          "$booster$" : "-($pos_y$ - $pos_y_star$)**2 + $m_fuel$/$m_fuel_ini$"
        },
        "_stop_episode" : "stop episode when goal is reach.", 
        "_stop_episode_info1" : "If feature has 1 value, its feature's value must be equal", 
        "_stop_episode_info2" : "if feature has 2 values [min_limit, max_limit]. Criterion is bounded feature >= min_limit and feature <= max_limit",
        "stop_episode" : {
          "pos_y" : [1, 0],
          "acceleration_y" : [-2,2],
          "speed_y" : [-2,2]
      }
    }
  env = Environment(json_file)
  return env

# Test initialization of Environment class
def test_environment_initialization(env_instance):
    assert isinstance(env_instance, Environment)

def test_environment_reset(env_instance):
    # Perform some actions before resetting
    assert len(env_instance.all_states()["pos_y"]) == 1
    new_state, rewards, done, problem, info = env_instance.step(["0"])
    assert len(env_instance.all_states()["pos_y"]) == 2
    # Then reset the environment
    env_instance.reset()
    assert len(env_instance.all_states()["pos_y"]) == 1