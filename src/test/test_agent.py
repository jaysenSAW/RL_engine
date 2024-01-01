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
    colnames = agent.action_space.keys()


def test_last_state():
    agent = Environment("src/test/input.json")
    assert isinstance(agent.last_state(), dict)

def test_select_states():
    agent = Environment("src/test/input.json")
    assert isinstance(agent.select_states(), dict)
    assert isinstance(
        agent.select_states(colnames = agent.action_space.keys()),
        dict)
    # check we get an empty list when we are out of bound
    assert all([
        len(agent.select_states(start = 2)[var]) == 0 for var in agent.variable_names])
