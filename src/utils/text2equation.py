import json
import re
import numpy as np
from sympy import sympify

def resolve_equations(features_values : dict,
                  list_equations : dict,
                  delimeter : str= "$") -> dict:
    solved_values = {}
    for term, equation in list_equations.items():
        # convert string to SymPy expressions
        expr = sympify(equation.replace(delimeter, ''))
        # flag "subs": take a dictionary of Sylmbol: point pairs.
        # check we get goot format
        if all([isinstance(value, np.ndarray) for key, value in features_values.items()]):
            solved_values[term.replace(delimeter, '')] = np.array([float(
                    sympify(expr).evalf(
                        subs={
                            key: value.item() for key, value in features_values.items()
                            }
                    )
                )])
        elif all([isinstance(value, list) for key, value in features_values.items()]):
            solved_values[term.replace(delimeter, '')] = np.array([float(
                    sympify(expr).evalf(
                        subs={
                            key: np.array(value).item() for key, value in features_values.items()
                            }
                    )
                )])
        else:
            solved_values[term.replace(delimeter, '')] = np.array([float(
                    sympify(expr).evalf(
                        subs={
                            key: value for key, value in features_values.items()
                            }
                    )
                )])
        features_values[term.replace('$', '')] = np.array(
            [solved_values[term.replace('$', '')]]
            ).flatten()
    return features_values

def debug_resolve_equations(features_values : dict,
                  list_equations : dict,
                  delimeter : str= "$") -> None:
    # dictionary with all state variables. values used to solve
    # equation are taken from features_values
    # dictionary with all computed variables. New values computed
    # are stored into it
    solved_values = {}
    print("Equation :")
    for term, equation in list_equations.items():
        # convert string to SymPy expressions
        expr = sympify(equation.replace(delimeter, ''))
        # flag "subs": take a dictionary of Sylmbol: point pairs.
        print("{0} = {1}".format(
            term.replace(delimeter, ''),
            sympify(expr).evalf(
                subs={key: value.item() for key, value in features_values.items()}
            )
        ))
    print("\n\nAttempt to solve equations with initial condition")
    for term, equation in list_equations.items():
        # convert string to SymPy expressions
        expr = sympify(equation.replace(delimeter, ''))
        # flag "subs": take a dictionary of Sylmbol: point pairs.
        print("{0} = {1:.3f}".format(
            term.replace(delimeter, ''),
            sympify(expr).evalf(
                subs={key: value.item() for key, value in features_values.items()}
            )
        ))
        solved_values[term.replace(delimeter, '')] = np.array([float(
                sympify(expr).evalf(
                    subs={
                        key: value.item() for key, value in features_values.items()
                        }
                )
            )])
        features_values[term.replace('$', '')] = np.array(
            [solved_values[term.replace('$', '')]]
            ).flatten()
