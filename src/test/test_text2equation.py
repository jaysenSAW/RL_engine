import json
import re
import numpy as np
from sympy import sympify
import unittest
import sys
sys.path.insert(1, "src/utils/")
from text2equation import resolve_equations

def test_resolve_equations():
    features_values = {'x': np.array([4.0]),
    'a': np.array([-1.0]),
    'c': np.array([6])}
    list_equations = {'$z$' : '$a$ * $x$^2 + $x$ + $c$'}
    result = resolve_equations(features_values, list_equations)
    assert len(result.keys()) == 4
    assert result["z"][0] == -6
    assert isinstance(result, dict)
