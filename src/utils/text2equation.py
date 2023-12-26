import json
import re
import numpy as np
from sympy import sympify



def solve_equations(features_values : dict,
                  list_equations : dict,
                  delimeter : str= "$") -> dict:
    """
    Solve a system of equations represented as a dictionary of equations and their terms.

    Args:
        features_values (dict): A dictionary containing the last known values for variables.
        list_equations (dict): A dictionary of equations, where keys are terms and values are equations.
        delimeter (str): character uses to delimeter terms

    Returns:
        dict: A dictionary with the solved values of terms from the equations.

    Example:
        features_values -> {'a' : np.array([1]),
                            'x' : np.array([2]),
                            'b' : np.array([-3]),
                            'c' : np.array([-1]),
                            'm' : np.array([10])}
        If 'list_equations' is a dictionary with the following structure:
        list_equations -> {'$z$': '$a$*$x$**2 + $b$*$x$ + $c$',
                             '$v$': '$m$*$x$ + $b$',
                             '$x$': '$v$ + $z$'}

        >>> list_equations = {'$term1$': '$a$*$x$**2 + $b$*$x$ + $c$'}
        >>> solved_values = solv_equation(features_values, list_equations)
        >>> print(solved_values)
        {"term1": array([-3.0])}
    """
    solved_values = {}
    for term, equation in list_equations.items():
        # convert string to SymPy expressions
        expr = sympify(equation.replace(delimeter, ''))
        # flag "subs": take a dictionary of Sylmbol: point pairs.
        try:
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
            if term.replace(delimeter, '') not in features_values.keys():
                # if term is not in features_values it means it temporary variable we add it
                features_values[term.replace('$', '')] = np.array(
                    [solved_values[term.replace('$', '')]]
                    )
        except ValueError:
            print("Value is not define")
    return solved_values
