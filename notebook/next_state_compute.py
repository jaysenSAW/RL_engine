
import numpy as np
import json

def compute_equations_variables(state) -> dict:
	F = np.array([ 600 ]).flatten()
	state["m_fuel"] = np.array([ state["m_fuel"] - state["booster"] ]).flatten()
	weight_rocket = np.array([ 5 + state["m_fuel"] ]).flatten()
	dt = np.array([ 0.5 ]).flatten()
	theta = np.array([ 0.0 ]).flatten()
	x_0 = np.array([ state["pos_x"] ]).flatten()
	y_0 = np.array([ state["pos_y"] ]).flatten()
	Vx_0 = np.array([ state["speed_x"] ]).flatten()
	Vy_0 = np.array([ state["speed_y"] ]).flatten()
	state["angle"] = np.array([ theta + state["alpha"] ]).flatten()
	state["acceleration_x"] = np.array([ (F/(5+weight_rocket) * np.sin(state["angle"])) * state["booster"] ]).flatten()
	state["acceleration_y"] = np.array([ (F/(5+weight_rocket) * np.cos(state["angle"])) * state["booster"] - state["G"] ]).flatten()
	state["speed_x"] = np.array([ (F/(5+weight_rocket) * np.sin(state["angle"])) * state["booster"] * dt + Vx_0 ]).flatten()
	state["speed_y"] = np.array([ (F/(5+weight_rocket) * np.cos(state["angle"])) * state["booster"] * dt - state["G"] * dt + Vy_0 ]).flatten()
	state["pos_x"] = np.array([ (0.5 * F/(5+weight_rocket) * np.sin(state["angle"])) * state["booster"] * dt**2 + Vx_0 * dt + x_0 ]).flatten()
	state["pos_y"] = np.array([ (0.5 * F/(5+weight_rocket) * np.cos(state["angle"])) * state["booster"] * dt**2 - state["G"] * dt**2 + Vy_0 * dt + y_0 ]).flatten()
	state["futur_pos_y"] = np.array([ state["pos_y"] + 3 * state["speed_y"] ]).flatten()
	state["futur_pos_x"] = np.array([ state["pos_x"] + 3 * state["speed_x"] ]).flatten()
	return state


def compute_equations_rewards(state) -> dict:
	state["booster"] = np.array([ -(state["pos_y"] - state["pos_y_star"])**2 + state["m_fuel"]/state["m_fuel_ini"] ]).flatten()
	state["alpha"] = np.array([  -(state["pos_x"] - state["pos_x_star"])**2 - np.sin(state["alpha"]) ]).flatten()
	return state


def compute_action(state : dict, action : float, trigger_var : str) -> dict:
	if trigger_var == "booster":
		state["booster"] = np.array([ action ]).flatten()
	if trigger_var == "alpha":
		state["alpha"] = np.array([ state["alpha"] + action ]).flatten()
	return state
