import numpy as np
import json
import os
import sys
from sympy import sympify
import re
import copy
from text2equation import resolve_equations, debug_resolve_equations

class Environment():

    def compute_reward_for_agents(self, start = -1, end = None, agent_variables = None):
        if agent_variables is None:
            agent_variables = self.agent_variables
        return {trigger_var : resolve_equations(
            copy.deepcopy(self.select_states(start, end)),
            self.json["equations_rewards"])[trigger_var]
            for trigger_var in agent_variables }

    def check_input(self, syst_dic):
        # check limit number of field
        if not (np.array([len(val) for key, val in syst_dic['limit'].items()]).flatten() == 3).all():
            print("Error ! Expect 3 filed for limit. [minimum, maximum, number_bins]")
            sys.exit()

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
        self.check_input(syst_dic)
        self.json = syst_dic
        self.states_variables = syst_dic["states_variables"]
        self.agent_variables = syst_dic["agent_variables"]
        self.variable_names = tuple([key.replace(delimiter, '') for key in syst_dic["initial_values"].keys()])
        self.action_to_take = syst_dic["action_to_take"]
        initial_system = {tmpkey.replace(delimiter, ''): value for tmpkey, value in syst_dic["initial_values"].items()}
        for key, value in initial_system.items():
            setattr(self, key, np.array(value))
        # reward for each agents
        self.rewards = self.compute_reward_for_agents()
        self.start_pos = initial_system
        self.current_pos = np.array([np.round(value, 6) for tmpkey, value in self.start_pos.items()]).flatten()
        self.action_space = {tmpkey.replace(delimiter, '') : len(value) for tmpkey, value in syst_dic["n_action"].items()}
        self.actions = {tmpkey.replace(delimiter, '') : value for tmpkey, value in syst_dic["n_action"].items()}
        # Define the observation space based on your state variables
        self.lower_lim = np.array([list(val)[0] for key, val in syst_dic['limit'].items()]).flatten()
        self.upper_lim = np.array([list(val)[1] for key, val in syst_dic['limit'].items()]).flatten()
        if 'n_bins' in syst_dic.keys():
            self.n_bins = np.array([list(val)[2] for key, val in syst_dic['limit'].items()]).flatten()
        else:
            # use upper and lower limit to discretize space with 1 unit step
            self.n_bins = self.upper_lim - self.lower_lim + 1

    def reset(self):
        for key, value in self.select_states(0,1).items():
            setattr(self, key, np.array(value))
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
                colnames = self.variable_names
            elif isinstance(self.variable_names, tuple):
                colnames = list(self.variable_names)
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
                colnames = self.variable_names
            elif isinstance(self.variable_names, tuple):
                colnames = list(self.variable_names)
        state = {}
        for key in colnames:
            # intiate state
            # state[key] = np.array([self.__dict__[key][-1]])
        # return stateHah
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
                colnames = self.variable_names
            elif isinstance(self.variable_names, tuple):
                colnames = list(self.variable_names)
        state = {}
        for key in colnames:
            if key not in list(self.variable_names):
                continue
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
                if key not in list(self.rewards.keys()):
                    continue
                if start is None and end is None:
                    state[key] = self.rewards[key]
                elif start is None and end is not None:
                    state[key] = self.rewards[key][ : end]
                elif start is not None and end is None:
                    state[key] = self.rewards[key][start : ]
                else:
                    state[key] = self.rewards[key][start : end]
            return state

    def uppdate_variables(self, new_state, colnames = None):
        if colnames is None:
            if isinstance(self.variable_names, list):
                colnames = self.variable_names
            elif isinstance(self.variable_names, tuple):
                colnames = list(self.variable_names)
        # uppdate state variable
        for attr_name in colnames:
            setattr(self, attr_name,
                    np.append(
                        getattr(self, attr_name),
                        new_state[attr_name]
                    )
            )

    def delete_last_states(self, colnames = None, end_index : int = -1):
        """Remove the last visited states and last rewards from the system."""
        if colnames is None:
            if isinstance(self.variable_names, list):
                colnames = self.variable_names
            elif isinstance(self.variable_names, tuple):
                colnames = list(self.variable_names)
        for attr_name in colnames:
            current_value = getattr(self, attr_name)
            setattr(self, attr_name, current_value[:end_index])
        setattr(self, "rewards", {key: values[:end_index] for key, values in self.rewards.items()})

    def discretized_space(self, dico = False):
        """
        Discretizes the system space.

        Returns:
            numpy.ndarray: Discretized bins.
        """
        low = self.lower_lim
        high = self.upper_lim
        if dico:
            tmp = [np.linspace(float(l), float(h), int(b)) for l, h, b in
                         zip(low, high, self.n_bins)]
            return {key : val for key, val in zip(self.start_pos.keys(), tmp)}
        else:
            return [np.linspace(float(l), float(h), int(b)) for l, h, b in
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
            list_pos.append([np.round(val_bins[i][val], 6) for val in index])
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
        coord = []
        labels = self.states_variables
        obs = self.discretized_observation(dico = True, start = start, end = end)
        # return tuple(obs[key] for key in labels)
        return tuple([elmnt for key, elmnt in obs.items() if key in labels])

    def move_agent(self,
                    action_key : str,
                    trigger_var : str,
                    temporary_state : dict = None,
                    equation : dict[str] = None):
        if temporary_state is None:
            temporary_state = self.last_state()
        if equation is None:
            equation = self.action_to_take[trigger_var]
        # Move agent and assign its new value according to action_to_take
        if isinstance(action_key, str):
            temporary_state["action"] = np.array([self.actions[trigger_var][action_key]])
        else:
            temporary_state["action"] = np.array([self.actions[trigger_var][str(action_key)]])
        temporary_state[trigger_var] = resolve_equations(
            temporary_state,
            equation
        )[trigger_var]
        del temporary_state["action"]
        return temporary_state

    def step(self, actions, agent_variables = None, method : str = "centralized"):
        """
        Perform an environment step for multiple agents with different trigger variables and actions.

        Args:
            actions (list): List of chosen action keys for each agent. If the action space is discrete,
                        provide the action index. If the action space is continuous, provide the action key.
            agent_variables (list): List of trigger variables for each agent.

        Returns:
            tuple: A tuple containing the following:
                - dict: New BioreactorGym instances for each agent, with updated positions.
                - dict: Rewards for each agent for the current step.
                - list: Flags indicating whether the episode is done for each agent.
                - list: Flags indicating whether there was a problem with the step for each agent.
                - list: Additional information messages for each agent.

        Note:
            - If the action space is continuous, provide the action key as a string.
            - The 'problem' list indicates if there was an issue with the step, e.g., new position out of bounds.
            - The 'info' list provides additional information messages for each agent.
        """
        temporary_states = self.last_state()
        rewards = {}
        done = []
        problem = []
        info = []
        if agent_variables is None:
            agent_variables = self.agent_variables
        # each agent are moved before compute new state
        # rewards are computed after update state
        if method == "centralized" :
            for trigger_var, action_key in zip(agent_variables, actions):
                temporary_state = self.move_agent(action_key,
                                                trigger_var,
                                                temporary_states,
                                                self.action_to_take[trigger_var])
            # Evaluate new environment variables
            solv_eq = resolve_equations(
                copy.deepcopy(temporary_state),
                self.json["equations_variables"]
            )
            for key in set(solv_eq.keys()) & set(temporary_state.keys()):
                temporary_state[key] = solv_eq[key]
            rewards = {trigger_var : resolve_equations(
                copy.deepcopy(temporary_state), self.json["equations_rewards"])[trigger_var]
                for trigger_var in agent_variables }
        # Move a agent compute the new state and repeat process for next agent
        else:
            for trigger_var, action_key in zip(agent_variables, actions):
                temporary_state = self.move_agent(action_key,
                                                trigger_var,
                                                temporary_states,
                                                self.action_to_take[trigger_var])
                # Evaluate new environment variables
                solv_eq = resolve_equations(
                    copy.deepcopy(temporary_state),
                    self.json["equations_variables"]
                )
                for key in set(solv_eq.keys()) & set(temporary_state.keys()):
                    temporary_state[key] = solv_eq[key]
                rewards[trigger_var] = resolve_equations(
                    copy.deepcopy(temporary_state),
                    self.json["equations_rewards"])[trigger_var]
        # Add new current position keys to use the same ones in the initial values field
        self.uppdate_variables(temporary_state)
        # update rewards
        for key in self.rewards.keys():
            self.rewards[key] = np.append(self.rewards[key], rewards[key])
        self.current_pos = np.array([
            temporary_state[tmpkey.replace('$', '')]
            for tmpkey in self.json["initial_values"].keys()
        ]).flatten()
        if any(self.upper_lim < self.current_pos) or any(self.lower_lim > self.current_pos):
            info.append("new position is out of bound")
            done.append(True)
            problem.append(True)
        else:
            info.append("new position")
            done.append(False)
            problem.append(False)
        return self.last_state(), rewards, done, problem, info

    def check_variables_and_equations(self, delimeter = "$"):
        print("equations variables")
        debug_resolve_equations(self.last_state(), self.json["equations_variables"],  delimeter)
        print("\nEverything is good :)")
