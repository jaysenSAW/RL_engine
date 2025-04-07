
import numpy as np
import json

def compute_equations_variables(state) -> dict:
	F = np.array([ 600 ]).flatten()
	state["m_fuel"] = np.array([ state["m_fuel"] - state["booster"] *10 -state["angle"] *10 ]).flatten()
	state["weight_rocket"] = np.array([ state["weight_dry_rocket"] + state["m_fuel"] ]).flatten()
	dt = np.array([ 0.5 ]).flatten()
	theta = np.array([ 0.0 ]).flatten()
	y_0 = np.array([ state["pos_y"] ]).flatten()
	Vy_0 = np.array([ state["speed_y"] ]).flatten()
	state["angle"] = np.array([ theta + state["alpha"] ]).flatten()
	state["acceleration_y"] = np.array([ (F/(5+state["weight_rocket"]) * np.cos(state["angle"])) * state["booster"] - state["G"] ]).flatten()
	state["speed_y"] = np.array([ (F/(5+state["weight_rocket"]) * np.cos(state["angle"])) * state["booster"] * dt - state["G"] * dt + Vy_0 ]).flatten()
	state["pos_y"] = np.array([ (0.5 * F/(5+state["weight_rocket"]) * np.cos(state["angle"])) * state["booster"] * dt**2 - state["G"] * dt**2 + Vy_0 * dt + y_0 ]).flatten()
	state["futur_pos_y"] = np.array([ state["pos_y"] + 3 * state["speed_y"] ]).flatten()
	return state


def compute_equations_rewards(state) -> dict:
	distance_y_reward = np.array([ np.abs( state["pos_y"] - state["pos_y_star"]) ]).flatten()
	acceleration_limit = np.array([ 5 ]).flatten()
	acceleration_y_reward = np.array([ np.exp(1) - np.exp( np.max([ np.max( np.abs(state["acceleration_y"])/acceleration_limit), 1 ]) ) ]).flatten()
	speed_limit = np.array([ 17 ]).flatten()
	speed_y_reward = np.array([ np.exp(1) - np.exp( np.max([ np.max( np.abs(state["speed_y"])/speed_limit ), 1 ]) ) ]).flatten()
	y_lower_limit = np.array([ 0 ]).flatten()
	y_upper_limit = np.array([ 200 ]).flatten()
	upper_boundary = np.array([ np.exp(np.min([ np.min(-state["futur_pos_y"] + y_upper_limit), 0])) ]).flatten()
	lower_boundary = np.array([ np.exp(np.min([ np.min(state["futur_pos_y"] -y_lower_limit), 0])) ]).flatten()
	height_boundaries = np.array([ -2 + lower_boundary + upper_boundary ]).flatten()
	ratio_fuel = np.array([ state["m_fuel"]/state["m_fuel_ini"] ]).flatten()
	_booster = np.array([ 3 * distance_y_reward + acceleration_y_reward + speed_y_reward + 2 * height_boundaries + 0.5* ratio_fuel ]).flatten()
	state["booster"] = np.array([ -distance_y_reward - np.abs(state["speed_y"]) ]).flatten()
	return state


def compute_action(state : dict, action : float, trigger_var : str) -> dict:
	if trigger_var == "booster":
		state["booster"] = np.array([ action ]).flatten()
	return state
