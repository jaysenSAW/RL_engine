import json
import re
import numpy as np
from sympy import sympify



def solv_equation(var_and_val : dict,
                  list_equations : dict,
                  delimeter : str = "$") -> dict:
    """
    Solve equations represented stored as string by using sympy

    Args:
        var_and_val (dict): A dictionary containing the last known values for variables.
        list_equations (dict): A dictionary of equations, where keys are variables and values are equations.
        delimeter (str): character uses to delimeter variable name 

    Returns:
        dict: A dictionary with the solved values of terms from the equations.
    """
    solv_eq = {}
    for term, equation in list_equations.items():
        # convert string to SymPy expressions
        expr = sympify(equation.replace(delimeter, ''))
        # flag "subs": take a dictionary of Sylmbol: point pairs.
        solv_eq[term.replace(delimeter, '')] = np.array([float(
                sympify(expr).evalf(
                    subs={
                        key: value.item() for key, value in var_and_val.items()
                        }
                )
            )])
        if term.replace('$', '') not in var_and_val.keys():
            # if term is not in var_and_val it means it temporary variable we add it
            var_and_val[term.replace('$', '')] = np.array(
                [solv_eq[term.replace('$', '')]]
                )
    return solv_eq
