import numpy as np
import json
import os
import sys
from sympy import sympify
import re
import copy
from text2equation import resolve_equations, debug_resolve_equations

class Environment():

    def compute_reward_for_agents(self, start = -1, end = None, trigger_variables = None):
        if trigger_variables is None:
            trigger_variables = self.trigger_variables
        return {trigger_var : resolve_equations(
            self.select_states(start, end),
            self.json["equations_rewards"])[trigger_var]
            for trigger_var in trigger_variables }

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
        self.json = syst_dic
        self.target_variable = syst_dic["target_variable"]
        self.trigger_variables = syst_dic["trigger_variables"]
        self.variable_names = tuple([key.replace(delimiter, '') for key in syst_dic["initial_values"].keys()])
        self.action_to_take = syst_dic["action_to_take"]
        initial_system = {tmpkey.replace(delimiter, ''): value for tmpkey, value in syst_dic["initial_values"].items()}
        for key, value in initial_system.items():
            setattr(self, key, np.array(value))
        # gloabal score for each step
        self.global_reward = np.array([np.nan])
        # reward for each agents
        self.rewards = self.compute_reward_for_agents()
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

    def reset(self):
        for key, value in self.select_states(0,1).items():
            setattr(self, key, np.array(value))
        self.global_reward = np.array([np.nan])
        self.rewards = self.compute_reward_for_agents()
        self.current_pos = copy.deepcopy(self.start_pos)

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

    def select_rewards(self, start : int = None, end : int = None, colnames = None):
            """display features values for a specific range of state

            Args:
                start (int): first state
                end (int): last state (not display)

            Returns:
                _type_: _description_
            """
            if colnames is None:
                colnames = self.rewards.keys()
            state = {}
            for key in colnames:
                # intial state
                if start is None and end is None:
                    state[key] = self.rewards[key]
                elif start is None and end is not None:
                    state[key] = self.rewards[key][ : end]
                elif start is not None and end is None:
                    state[key] = self.rewards[key][start : ]
                else:
                    state[key] = self.rewards[key][start : end]
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
        setattr(self, "rewards", {key: values[:-1] for key, values in self.rewards.items()})

    def discretized_space(self, dico = False):
        """
        Discretizes the system space.

        Returns:
            numpy.ndarray: Discretized bins.
        """
        low = self.lower_lim
        high = self.upper_lim
        if dico:
            tmp = [np.linspace(l, h, b) for l, h, b in
                         zip(low, high, self.n_bins)]
            return {key : val for key, val in zip(self.start_pos.keys(), tmp)}
        else:
            return [np.linspace(l, h, b) for l, h, b in
                         zip(low, high, self.n_bins)]


    def discretized_observation(self, dico = False, start = -1, end = None):
        """
        Discretizes the current position for observation.

        Returns:
            numpy.ndarray: Discretized position
        """
        val_bins = self.discretized_space()
        list_pos = []
        for i, key in zip(range(len(self.start_pos.keys())), self.start_pos.keys()):
            dist = val_bins[i] - self.select_states(start,end)[key][:, np.newaxis]
            index = [np.argmin(array) for array in np.abs(dist)]
            list_pos.append([val_bins[i][val] for val in index])
        if dico:
            return {key : list_pos[i]
                    for key, i in zip(self.start_pos.keys(),
                                      range(len(self.start_pos.keys()))
                                      )
                    }
        else:
            if len(index) == 1:
                #return only array
                return np.array(list_pos).reshape(len(index), len(self.start_pos.keys()))[0]
            else:
                # return array of array
                return np.array(list_pos).reshape(len(index), len(self.start_pos.keys()))

    def state_for_q_table(self, start = -1, end = None) -> tuple:
        # get coordinate without trigger variables
        labels = set(self.start_pos.keys()) - set(self.trigger_variables)
        obs = self.discretized_observation(dico = True, start = start, end = end)
        return tuple(obs[key] for key in labels)

    def move_agent(self, action_key : str,
        trigger_variable : str,
        temporary_state : dict = None):
        """
        """
        if temporary_state is None:
            temporary_state = self.last_state()
        if isinstance(action_key, str):
            temporary_state["action"] = np.array([self.actions[trigger_variable][action_key]])
        else:
            temporary_state["action"] = np.array([self.actions[trigger_variable][str(action_key)]])
        return resolve_equations(
            copy.deepcopy(temporary_state),
            self.action_to_take[trigger_variable]
        )[trigger_variable]

    def step(self, actions : list[str],
        trigger_variables : list[str] = None):
        """
        """
        if trigger_variables is None:
            trigger_variables = self.trigger_variables
        solv_eq = {}
        new_states = {}
        rewards = {}
        done = []
        problem = []
        info = []
        # Evaluate new environment variables
        for action_key, trigger_variable in zip(actions, trigger_variables):
            # move agent according to action
            temporary_state = self.last_state()
            # new trigger value
            temporary_state[trigger_variable] = self.move_agent(
                action_key,
                trigger_variable
            )
            # Evaluate new environment variables
            solv_eq = resolve_equations(temporary_state, self.json["equations_variables"])
            for key in set(solv_eq.keys()) & set(temporary_state.keys()):
                temporary_state[key] = solv_eq[key]
            # Add new current position keys to use the same ones in the initial values field
            self.uppdate_state_variables(temporary_state)
            self.current_pos = np.array([
                temporary_state[tmpkey.replace('$', '')]
                for tmpkey in self.json["initial_values"].keys()
            ]).flatten()
            new_states[trigger_variable] = copy.deepcopy(self)
            # compute reward for trigger_variable
            tmp = self.compute_reward_for_agents()
            for var in trigger_variables:
                self.rewards[var] = np.append(
                    self.rewards[var],
                    tmp[var])
            rewards[trigger_variable] = self.select_rewards(start = -1)
            if any(self.upper_lim < self.current_pos) or any(self.lower_lim > self.current_pos):
                info.append("new position is out of bound")
                done.append(True)
                problem.append(True)
            else:
                info.append("new position")
                done.append(False)
                problem.append(False)
            # store new state only if we are in mono agent
            if len(trigger_variables) > 1:
                self.delete_last_state()
        return new_states, rewards, done, problem, info

    def check_variables_and_equations(self, delimeter = "$"):
        print("equations variables")
        debug_resolve_equations(self.last_state(), self.json["equations_variables"],  delimeter)
        print("\nEverything is good :)")
