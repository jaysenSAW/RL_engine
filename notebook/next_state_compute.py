
import numpy as np
import json

def compute_equations_variables(state) -> dict:
	F = np.array([ 600 ]).flatten()
	state["m_fuel"] = np.array([ state["m_fuel"] - state["booster"] *10 -state["angle"] *10 ]).flatten()
	state["weight_rocket"] = np.array([ state["weight_dry_rocket"] + state["m_fuel"] ]).flatten()
	dt = np.array([ 2.0 ]).flatten()
	theta = np.array([ 0.0 ]).flatten()
	x_0 = np.array([ state["pos_x"] ]).flatten()
	y_0 = np.array([ state["pos_y"] ]).flatten()
	Vx_0 = np.array([ state["speed_x"] ]).flatten()
	Vy_0 = np.array([ state["speed_y"] ]).flatten()
	state["angle"] = np.array([ theta + state["alpha"] ]).flatten()
	state["acceleration_x"] = np.array([ (F/(5+state["weight_rocket"]) * np.sin(state["angle"])) * state["booster"] ]).flatten()
	state["acceleration_y"] = np.array([ (F/(5+state["weight_rocket"]) * np.cos(state["angle"])) * state["booster"] - state["G"] ]).flatten()
	state["speed_x"] = np.array([ (F/(5+state["weight_rocket"]) * np.sin(state["angle"])) * state["booster"] * dt + Vx_0 ]).flatten()
	state["speed_y"] = np.array([ (F/(5+state["weight_rocket"]) * np.cos(state["angle"])) * state["booster"] * dt - state["G"] * dt + Vy_0 ]).flatten()
	state["pos_x"] = np.array([ (0.5 * F/(5+state["weight_rocket"]) * np.sin(state["angle"])) * state["booster"] * dt**2 + Vx_0 * dt + x_0 ]).flatten()
	state["pos_y"] = np.array([ (0.5 * F/(5+state["weight_rocket"]) * np.cos(state["angle"])) * state["booster"] * dt**2 - state["G"] * dt**2 + Vy_0 * dt + y_0 ]).flatten()
	state["futur_pos_y"] = np.array([ state["pos_y"] + 3 * state["speed_y"] ]).flatten()
	state["futur_pos_x"] = np.array([ state["pos_x"] + 3 * state["speed_x"] ]).flatten()
	y_lower_limit = np.array([ 0 ]).flatten()
	y_upper_limit = np.array([ 200 ]).flatten()
	state["upper_boundary"] = np.array([ -np.exp(0) + np.exp(np.min([ np.min(-state["futur_pos_y"] + y_upper_limit), 0])) ]).flatten()
	state["lower_boundary"] = np.array([ -np.exp(0) + np.exp(np.min([ np.min(state["futur_pos_y"] -y_lower_limit), 0])) ]).flatten()
	return state


def compute_equations_rewards(state) -> dict:
	_distance_y_reward = np.array([ np.abs( state["pos_y"] - state["pos_y_star"]) ]).flatten()
	distance_y_reward = np.array([ np.abs( (state["pos_y"] - state["pos_y_star"])/(state["pos_y_ini"] - state["pos_y_star"]) ) ]).flatten()
	distance_x_reward = np.array([ np.abs( (state["pos_x"] - state["pos_x_star"])/(state["pos_x_ini"] - state["pos_x_star"]) ) ]).flatten()
	acceleration_limit = np.array([ 5 ]).flatten()
	acceleration_y_reward = np.array([ np.exp(1) - np.exp( np.max([ np.max( np.abs(state["acceleration_y"])/acceleration_limit), 1 ]) ) ]).flatten()
	speed_limit = np.array([ 15 ]).flatten()
	speed_y_reward = np.array([ np.exp(1) - np.exp( np.max([ np.max( np.abs(state["speed_y"])/speed_limit ), 1 ]) ) ]).flatten()
	y_lower_limit = np.array([ 0 ]).flatten()
	y_upper_limit = np.array([ 200 ]).flatten()
	upper_boundary = np.array([ -np.exp(0) + np.exp(np.min([ np.min(-state["futur_pos_y"] + y_upper_limit), 0])) ]).flatten()
	lower_boundary = np.array([ -np.exp(0) + np.exp(np.min([ np.min(state["futur_pos_y"] -y_lower_limit), 0])) ]).flatten()
	height_boundaries = np.array([ -2 + lower_boundary + upper_boundary ]).flatten()
	ratio_fuel = np.array([ state["m_fuel"]/state["m_fuel_ini"] ]).flatten()
	_booster = np.array([ 3 * distance_y_reward + acceleration_y_reward + speed_y_reward + 2 * height_boundaries + 0.5* ratio_fuel ]).flatten()
	state["booster"] = np.array([ -distance_y_reward * 3 + acceleration_y_reward + speed_y_reward + 0.5* ratio_fuel ]).flatten()
	state["alpha"] = np.array([  -(state["pos_x"] - state["pos_x_star"])**2 - np.sin(state["alpha"]) ]).flatten()
	return state


def compute_action(state : dict, action : float, trigger_var : str) -> dict:
	if trigger_var == "booster":
		state["booster"] = np.array([ action ]).flatten()
	if trigger_var == "alpha":
		state["alpha"] = np.array([ state["alpha"] + action ]).flatten()
	return state
