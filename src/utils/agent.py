import numpy as np
import json
import os
import sys
from sympy import sympify
import re

class Environment():
    def __init__(self, json_file, delimiter = "$"):
        """
        Initialize the environment and its rules.
        """
        if isinstance(json_file, str):
            with open(json_file, 'r') as config_file:
                syst_dic = json.load(config_file)
        elif isinstance(json_file, dict):
            syst_dic = json_file
        else:
            print("expect JSON file or a dictionary")
        initial_system = {tmpkey.replace(delimiter, ''): value for tmpkey, value in syst_dic["initial_values"].items()}
        for key, value in initial_system.items():
            if key == "rewards":
                print("Fatal error feature name in initial_values cannot be reward")
                sys.exit()
            setattr(self, key, np.array(value))
        # gloabal score for each step
        self.global_reward = np.array([np.nan])
        # reward for each agents
        self.rewards = {key : [] for key in syst_dic["trigger_variables"]}
        self.start_pos = initial_system
        self.current_pos = np.array([value for tmpkey, value in self.start_pos.items()]).flatten()
        self.goal_pos = syst_dic["goal_pos"]
        self.action_space = {tmpkey.replace(delimiter, '') : len(value) for tmpkey, value in syst_dic["n_action"].items()}
        self.actions = {tmpkey.replace(delimiter, '') : value for tmpkey, value in syst_dic["n_action"].items()}
        # Define the observation space based on your state variables
        self.lower_lim = np.array([np.min(val) for key, val in syst_dic['limit'].items()]).flatten()
        self.upper_lim = np.array([np.max(val) for key, val in syst_dic['limit'].items()]).flatten()
        if 'n_bins' in syst_dic.keys():
            self.n_bins = syst_dic["n_bins"]
        else:
            # use upper and lower limit to discretize space with 1 unit step
            self.n_bins = self.upper_lim - self.lower_lim + 1
        self.target_variable = syst_dic["target_variable"]
        self.trigger_variables = syst_dic["trigger_variables"]
        self.variable_names = tuple([key.replace(delimiter, '') for key in syst_dic["initial_values"].keys()])
        self.json = syst_dic
        self.action_to_take = syst_dic["action_to_take"]

    def all_states(self, colnames = None):
        """
        Get values of all features for each state.

        Returns:
            dict: Values for each feature.
        """
        if colnames is None:
            if isinstance(self.variable_names, list):
                colnames = self.variable_names  + ["global_reward"]
            elif isinstance(self.variable_names, tuple):
                colnames = list(self.variable_names)  + ["global_reward"]
        state = {}
        for key in colnames:
            state[key] = self.__dict__[key]
        return state

    def last_state(self, colnames = None):
        """
        Get the current state of the system.

        Returns:
            dict: Dictionary with values for all state variables.
        """
        if colnames is None:
            if isinstance(self.variable_names, list):
                colnames = self.variable_names  + ["global_reward"]
            elif isinstance(self.variable_names, tuple):
                colnames = list(self.variable_names)  + ["global_reward"]
        state = {}
        for key in colnames:
            # intiate state
            # state[key] = np.array([self.__dict__[key][-1]])
        # return state
            if isinstance(self.__dict__[key], np.ndarray):
                state[key] = np.array([self.__dict__[key][-1]])
            else:
                state[key] = self.__dict__[key]
        return state

    def select_states(self, start : int = None, end : int = None, colnames = None):
        """display features values for a specific range of state

        Args:
            start (int): first state
            end (int): last state (not display)

        Returns:
            _type_: _description_
        """
        if colnames is None:
            if isinstance(self.variable_names, list):
                colnames = self.variable_names  + ["global_reward"]
            elif isinstance(self.variable_names, tuple):
                colnames = list(self.variable_names)  + ["global_reward"]
        state = {}
        for key in colnames:
            # intial state
            if start is None and end is None:
                state[key] = self.__dict__[key]
            elif start is None and end is not None:
                state[key] = self.__dict__[key][ : end]
            elif start is not None and end is None:
                state[key] = self.__dict__[key][start : ]
            else:
                state[key] = self.__dict__[key][start : end]
        return state

    def uppdate_state_variables(self, new_state, colnames = None):
        if colnames is None:
            if isinstance(self.variable_names, list):
                colnames = self.variable_names  + ["global_reward"]
            elif isinstance(self.variable_names, tuple):
                colnames = list(self.variable_names)  + ["global_reward"]
        # uppdate state variable
        for attr_name in colnames:
            setattr(self, attr_name,
                    np.append(
                        getattr(self, attr_name),
                        new_state[attr_name]
                    )
            )

    def delete_last_state(self, colnames = None):
        """Remove the last visited state from the system."""
        if colnames is None:
            if isinstance(self.variable_names, list):
                colnames = self.variable_names  + ["global_reward"]
            elif isinstance(self.variable_names, tuple):
                colnames = list(self.variable_names)  + ["global_reward"]
        for attr_name in colnames:
            current_value = getattr(self, attr_name)
            setattr(self, attr_name, current_value[:-1])
