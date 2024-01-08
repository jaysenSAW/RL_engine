import numpy as np
from collections import defaultdict
from agent import Environment
import pandas as pd
import copy
import matplotlib.pyplot as plt
import sys
from itertools import product
import random
import os

class QLearningTrainer:
    def global_q_tables(self, env: Environment = None):
        """
        Generate a global Q-table structure for a multi-agent environment.

        This function extracts information about variables and actions from the provided Environment environment
        and creates an empty Pandas DataFrame with a MultiIndex representing the Q-table structure.

        Args:
            env (Environment): The environment.

        Returns:
            pd.DataFrame: An empty Pandas DataFrame with a MultiIndex representing the Q-table structure.
                        Rows are indexed by variables, columns are indexed by actions.
        """
        list_variables = []
        list_actions = []
        if env is None:
            keys = self.env.actions.keys()
            # actions = self.env.actions[key].keys()
            for key in keys:
                for action in self.env.actions[key].keys():
                    list_variables.append(key)
                    list_actions.append(action)
            # Create MultiIndex for the DataFrame columns
            col = pd.MultiIndex.from_arrays([list_variables, list_actions])
            # Create an empty DataFrame with the MultiIndex columns
            # Index value is state variables
            return pd.DataFrame(index=[str(self.env.state_for_q_table())], columns=col, dtype=object).fillna(0)
        else:
            keys = env.actions.keys()
            # actions = env.actions[key].keys()
            # Iterate through the environment's action space to extract variables and actions
            for key in keys:
                for action in env.actions[key].keys():
                    list_variables.append(key)
                    list_actions.append(action)
            # Create MultiIndex for the DataFrame columns
            col = pd.MultiIndex.from_arrays([list_variables, list_actions])
            # Create an empty DataFrame with the MultiIndex columns
            # Index value is state variables
            return pd.DataFrame(index=[str(env.state_for_q_table())], columns=col, dtype=object).fillna(0)

    def __init__(self, env: Environment, num_episodes: int = 50, learning_rate: float = 0.1, discount_factor: float = 0.99,
                 exploration_prob: list[float] = [0.2, 1], run_limit: int = 1000, decrease_prob_exp: float = 0.05,
                 convergence_criterion = 0.001, decay_type : str = "linear", verbose: bool = False):
        self.env = env
        if isinstance(num_episodes, int):
            self.num_episodes = num_episodes
        else:
            print(" Expect integer for num_episodes")
            self.num_episodes = int(num_episodes)
        if isinstance(learning_rate, float):
            self.learning_rate = learning_rate
        else:
            self.learning_rate = float(learning_rate)
        if isinstance(discount_factor, float):
            self.discount_factor = discount_factor
        else:
            self.discount_factor = float(discount_factor)
        self.exploration_prob = exploration_prob
        if isinstance(self.exploration_prob, list):
            if len(self.exploration_prob) == 2:
                self.min_prob = np.min(self.exploration_prob)
                self.max_prob = np.max(self.exploration_prob)
            elif len(self.exploration_prob) == 1:
                print("exploration_prob argument as 1 element. Minimum probabily is set to 0")
                self.min_prob = 0
                self.max_prob = self.exploration_prob
            else:
                print("expect 2 elements for exploration_prob !!! Use default value")
                self.min_prob = 0.2
                self.max_prob = 1.0
                sys.exit
        elif isinstance(self.exploration_prob, int) or isinstance(self.exploration_prob, float):
            print("exploration_prob argument as 1 element. Minimum probabily is set to 0")
            self.min_prob = 0
            self.max_prob = self.exploration_prob
        self.decrease_prob_exp = decrease_prob_exp
        self.decay_type = decay_type
        self.convergence_criterion = convergence_criterion
        self.run_limit = run_limit
        self.verbose = verbose
        self.monitor_iter = []
        self.states_for_all_episodes = []
        self.q_table_for_all_episodes = []
        self.q_table = self.global_q_tables(env)
        self.q_table_old = self.global_q_tables(env)

    def reinitialize(self):
        self.env.reset()
        self.states_for_all_episodes = []
        self.q_table_for_all_episodes = []
        self.q_table = self.global_q_tables(self.env)
        self.q_table_old = self.global_q_tables(self.env)

    def choose_action(self,
                      q_table : pd.DataFrame,
                      action_space : int,
                      exploration_prob: float):
        """
        Choose an action based on the current Q-values using epsilon-greedy exploration strategy.

        Args:
            q_table (pd.DataFrame): Q-values for state-action pairs.
            action_space (int): Number of possible actions.
            exploration_prob (float): Probability that our agent will explore the environment rather than exploit it (exploration).

        Returns:
            int: Chosen action index.
        """
        if np.random.uniform(0, 1) < exploration_prob:
            return np.random.choice(action_space)
        else:
            return np.argmax(q_table)

    def call_choose_action(self,
                        states : tuple,
                        proba : float) -> list:
        # action_spaces = self.env.action_space
        return [self.choose_action(
            self.q_table.loc[ [str(states)], [key] ],
            self.env.action_space[key],
            proba) for key in self.env.action_space.keys()]

    def reset_envrionement_multi_agent(self):
        num_agents = len(self.env.action_space.keys())
        states = {key : self.env.state_for_q_table() for key, _ in zip(self.env.action_space.keys(), range(num_agents))}
        # states, done, current_iter, iter_out_of_bound
        return states, [False] * len(self.env.action_space.keys()), 0, 0

    def update_q_values(self,
                        trigger_variable : str,
                        state: str,
                        next_state : str,
                        action : str,
                        reward : float):
        """
        Update Q-values based on the Q-learning update rule.
        https://deeplizard.com/learn/video/mo96Nqlo1L8

        Args:
            q_table (defaultdict): A defaultdict containing Q-values for state-action pairs.
            state (tuple): Current state representation for the Q-table.
            action (list): List of actions taken by all agents.
            reward (float): Immediate reward.
            next_state (tuple): Next state representation for the Q-table.
            discount_factor (float): Discount factor for future rewards.
            learning_rate (float): Learning rate for updating Q-values.
        """
        # q_table = q_table[trigger_variable]
        current_q_value = self.q_table.loc[
                        [state],
                        ([trigger_variable], [action])
                    ].to_numpy().flatten()[-1]
        # next_max_q_value = q_table.loc[next_state].max()
        next_max_q_value = self.q_table.loc[next_state, trigger_variable].max()
        if np.isnan(current_q_value):
            current_q_value = 0
        if np.isnan(next_max_q_value):
            next_max_q_value = 0
        updated_q = self.learning_rate * (reward + self.discount_factor * next_max_q_value)
        self.q_table.loc[
                        [state],
                        ([trigger_variable], [action])
                    ] = (1- self.learning_rate) * current_q_value + updated_q

    def global_q_tables(self, env: Environment = None):
        """
        Generate a global Q-table structure for a multi-agent environment.

        This function extracts information about variables and actions from the provided Environment environment
        and creates an empty Pandas DataFrame with a MultiIndex representing the Q-table structure.

        Args:
            env (Environment): The Environment environment.

        Returns:
            pd.DataFrame: An empty Pandas DataFrame with a MultiIndex representing the Q-table structure.
                        Rows are indexed by variables, columns are indexed by actions.
        """
        list_variables = []
        list_actions = []
        if env is None:
            keys = self.env.actions.keys()
            # actions = self.env.actions[key].keys()
            for key in keys:
                for action in self.env.actions[key].keys():
                    list_variables.append(key)
                    list_actions.append(action)
            # Create MultiIndex for the DataFrame columns
            col = pd.MultiIndex.from_arrays([list_variables, list_actions])
            # Create an empty DataFrame with the MultiIndex columns
            # Index value is state variables
            return pd.DataFrame(index=[str(self.env.state_for_q_table())], columns=col).fillna(0)
        else:
            keys = self.env.actions.keys()
            # actions = env.actions[key].keys()
            # Iterate through the environment's action space to extract variables and actions
            for key in keys:
                for action in self.env.actions[key].keys():
                    list_variables.append(key)
                    list_actions.append(action)
            # Create MultiIndex for the DataFrame columns
            col = pd.MultiIndex.from_arrays([list_variables, list_actions])
            # Create an empty DataFrame with the MultiIndex columns
            # Index value is state variables
            return pd.DataFrame(index=[str(self.env.state_for_q_table())], columns=col).fillna(0)

    def check_bound(self,
                    problem : list,
                    next_env : dict,
                    action_spaces : dict,
                    rewards : dict,
                    done : list,
                    info : list,
                    actions : list):
        # get indice
        for i in np.where(problem)[0]:
            tmp_problem = [problem[i]]
            # action leds system to be out of bound
            key = list(next_env.keys())[i]
            action = 0
            out_bound_env = copy.copy(next_env[key])
            # delete last state because is out of bound
            out_bound_env.delete_last_state()
            while (action < action_spaces[key]) and (any(tmp_problem) is True):
                # do all possible action
                tmp_env, tmp_rewards, tmp_done, tmp_problem, tmp_info = out_bound_env.step([action], [key])
                if tmp_problem[0]:
                    # system is still out of bound remove last attempt
                    out_bound_env.delete_last_state()
                    # new attempt
                else:
                    # change input values with correct state
                    next_env[key] = tmp_env[key]
                    rewards[key] = tmp_rewards[key]
                    problem[i] = tmp_problem[0]
                    done[i] = tmp_done[0]
                    info[i] = tmp_info[0]
                    actions[i] = action
                # new attempt
                action += 1
        return next_env, rewards, done, problem, info, actions

    def control_loop(self,
                    current_iter : int,
                    iter_out_of_bound : int) -> bool:
        """_summary_

        Args:
            q_table (defaultdict): updated Q-table
            old_q_table (defaultdict): previous Q-table
            current_iter (int): current iteration's number
            run_limit (int): Maximum number of iteration steps allowed

        Returns:
            bool: stop or not loop
        """
        # keys = q_table.keys() & old_q_table.keys()
        if len(set(self.q_table.index) - set(self.q_table_old.index)) == 0:
            diff_q = np.abs(np.sum([self.q_table - self.q_table_old]))
            if diff_q <= self.convergence_criterion and current_iter >= np.min([100,
                                                                int(self.run_limit/1.5)]):
                return True
        if current_iter > self.run_limit:
            print(f"Episode did not converged. You should try to increase run_limit")
            print("stop_criterion ", current_iter)
            print("diff_q ", diff_q)
            return True
        if iter_out_of_bound > 20:
            print(f"Episode did not converged. agent is out of bound during the last 20th iterations")
            print("stop_criterion ", current_iter)
            print("diff_q ", diff_q)
            return True
        return False

    def iterate_all_possibility(self):
        # Generate all possible combinations
        all_combinations = list(product(*self.env.actions.values()))
        list_actions = []
        # Print the result
        for combination in all_combinations:
            next_env, rewards, done, problem, info = self.env.step(np.array(combination).flatten(), self.env.trigger_variables)
            self.env.delete_last_state()
            if any(problem):
                continue
            else:
                list_actions.append(np.array(combination).flatten())
        if len(list_actions) == 0:
            return [True], None
        else:
            # retrun new action action to take
            return [False], random.choice(list_actions)

    def training_q_learning(self, proba : float = 0.2) -> int:
        """
        """
        # Reset environment and get initial state for each agent (coordinate without trigger variable)
        self.env.reset()
        states = self.env.state_for_q_table()
        done = [False]
        current_iter = 0
        iter_out_of_bound = 0
        # states, done, current_iter, iter_out_of_bound = self.reset_envrionement_multi_agent()
        while not any(done):
            states = self.env.state_for_q_table()
            # Choose actions for each agent based on the global policy
            # proba = np.max([self.min_prob, self.max_prob])
            actions = self.call_choose_action(states, proba)
            # compute new states according to agents's action
            next_env, rewards, done, problem, info = self.env.step(actions, self.env.trigger_variables)
            if any(problem):
                self.env.delete_last_state()
                #try to escape bound limit
                done, actions = self.iterate_all_possibility()
                if any(done):
                    print(done)
                    print("No action possible. Stop episode at {0}th iterations".format(current_iter))
                    continue
                next_env, rewards, done, problem, info = self.env.step(actions, self.env.trigger_variables)
                current_iter  += 1
            # Update Q-values based on the Q-learning update rule.
            #   env.action_space.keys() -> dict_keys(['X', 'B', 'D'])
            #   actions -> [0, 1, 1]
            for trigger_variable, action in zip(self.env.action_space.keys() , actions):
                new_row = self.global_q_tables(next_env)
                # check if next states is present in dataFrame
                if not new_row.index.to_list()[0] in self.q_table.index.to_list():
                    self.q_table = pd.concat([self.q_table, new_row])
                self.update_q_values(trigger_variable,
                    str(states),
                    str(self.env.state_for_q_table()),
                    str(action),
                    sum(rewards[trigger_variable]))
            done = [self.control_loop(current_iter, iter_out_of_bound)]
            current_iter  += 1
            iter_out_of_bound = 0
            self.q_table_old = self.q_table.copy()
        return current_iter

    def give_epsilon(self,
                     tau : float = None,
                     min_prob : float = None,
                     max_prob : float = None,
                     decay_type : str = None,
                     iter : int = None) -> np.ndarray:
        if tau is None:
            tau = self.decrease_prob_exp
        if min_prob is None:
            min_prob = self.min_prob
        if max_prob is None:
            max_prob = self.max_prob
        if decay_type is None:
            decay_type = self.decay_type
        if iter is None:
            iter = self.num_episodes
        if decay_type == "exponential":
            return [ round( np.max([min_prob, elmt]), 3)
                    for elmt in np.exp( -tau * np.arange(iter)) ]
        elif decay_type == "linear":
            return [ round(
                        np.max([min_prob, max_prob - tau * i]),
                        3)
                    for i in range(iter)]
        else:
            print("Error decay_type argument is unkown")
            return None

    def q_learning(self) -> None:
        """
        Perform Q-learning algorithm to estimate Q-values and learn an optimal policy.

        Args:
            env (Environment): The Environment environment.
            num_episodes (int): Number of episodes for training.
            learning_rate (float): Learning rate for updating Q-values.
            discount_factor (float): Discount factor for future rewards.
            exploration_prob (list): Highest and lowest probability of selecting a random action (exploration).
            run_limit (int, optional): Maximum number of iteration steps allowed. Defaults to 100.
            decrease_prob_exp (float, optional): decrease rate for exploration_prob

        Returns:
            defaultdict: A defaultdict containing learned Q-values for state-action pairs.
        """
        # monitor_iters = [[] for _ in range(len(env.action_space))]
        self.q_table = self.global_q_tables()
        self.q_table_old = self.global_q_tables()
        proba = self.give_epsilon()
        for episode in range(self.num_episodes):
            print("Episode {0}/{1}".format(episode+1, self.num_episodes))
            print("exploration_prob : {0:.3f}".format(proba[episode]))
            current_iter = self.training_q_learning(proba[episode])
            self.states_for_all_episodes.append(pd.DataFrame(self.env.all_states()))
            self.q_table_for_all_episodes.append(self.q_table.copy().replace(0, np.nan))
            # self.all_episodes.append(self.q_table.copy())
            ###############################################
            # self.max_prob -= self.decrease_prob_exp
            print("end while loop iteration : ", current_iter)
            self.monitor_iter.append(current_iter)
            # check if we can stop training early
            # look last 6 episodes and check
            if len(self.monitor_iter) > 6:
                tmp = np.sum(np.array(self.monitor_iter[-6:-1]) == np.array(self.monitor_iter[-5:]))
                if tmp == 5 and current_iter < self.run_limit -1:
                    print("Look like nothing to learn anymore, stop training")
                    break
        #             return q_table, self.monitor_iter
        #replace 0 by nan
        self.q_table.replace(0, np.nan)
        # return q_table, self.monitor_iter, self.all_episodes

    def plot_convergence(self, display_prob = True):
        # resize plot as function of iteration number
        plt.figure(figsize=(np.max([10, int(3 * len(self.monitor_iter)/10) ]), 6))
        x  = np.arange(1, len(self.monitor_iter)+1)
        plt.plot(x, self.monitor_iter, 'b-o')
        if display_prob:
            labels = self.give_epsilon(iter = len(self.monitor_iter) )
            plt.xticks(x, labels, rotation=45)
            plt.xlabel("Probability to choose random action")
        else:
            plt.xlabel("# Episode")
        plt.axhline(self.run_limit, color = 'r', linestyle = 'dotted', label = "maximum criterion iteration")
        plt.axhline(np.min([100, int(self.run_limit/1.5)]), color = 'r', linestyle = 'dotted', label = "minimum criterion iteration")
        plt.ylabel("Number of iterations")
        plt.title("Convergence")
        plt.legend()
