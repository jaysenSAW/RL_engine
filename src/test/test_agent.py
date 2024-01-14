import numpy as np
import json
import os
import sys
from sympy import sympify
import re
import copy
import unittest
sys.path.insert(1, "src/utils/")
from text2equation import resolve_equations, debug_resolve_equations
from agent import Environment

def test_all_states():
    agent = Environment("src/test/input.json")
    assert isinstance(agent.all_states(), dict)
    colnames = list(agent.action_space.keys())
    assert isinstance(agent.all_states(colnames), dict)
    assert list(agent.all_states(colnames).keys())[0] == colnames[0]

def test_last_state():
    agent = Environment("src/test/input.json")
    assert isinstance(agent.last_state(), dict)

def test_select_states():
    dic = {
      "states_variables" : ["X1"],
      "agent_variables" : ["U1"],
      "initial_values" : {
        "X1" : [0.0],
        "U1" : [0.0]
      },
      "_limit" : ["min", "max", "n_bins"],
      "limit" : {
        "X1" : [0.0, 10.0, 11],
        "U1" : [0.0, 1.0, 2]
      },
      "n_action" : {
        "U1": {"0" : 0.0, "1" : 1.0}
      },
      "_action_to_take" : "How action change trigger_variable, value is taken from n_action dictionary",
      "action_to_take" : {
        "U1": {"$U1$" : "$action$"}
      },

      "_equations" : "compute values for each equation. Variable are updated after the loop",
      "equations_variables": {
          "$l$" : "10.0",
          "$target$" : "5.0",
          "$dX1$" : "1 - $U1$ - ($X1$ - $target$)/$l$",
          "$X1$": "$X1$ + $dX1$",
          "$U1$": "$U1$"
      },
      "equations_rewards": {
        "$target$" : "5.0",
        "$U1$" : "-1 * ($X1$ - $target$)^2"
      }
    }
    agent = Environment(dic)
    assert isinstance(agent.select_states(), dict)
    assert isinstance(
        agent.select_states(colnames = agent.action_space.keys()),
        dict)
    # check we get an empty list when we are out of bound
    assert all([
        len(agent.select_states(start = 2)[var]) == 0 for var in agent.variable_names])
