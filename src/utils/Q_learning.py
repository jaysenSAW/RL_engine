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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
        if env is None:
            arrays = [list(self.env.actions[key].keys()) for key in self.env.actions.keys()]
            # Generate all possible combinations
            combinations = list(product(*arrays))
            # column is the combinason of all agent's key values
            col = pd.MultiIndex.from_arrays([["_".join(self.env.agent_variables)]* len(combinations), combinations])
            return pd.DataFrame(index=[str(self.env.state_for_q_table())], columns=col).fillna(0)
        else:
            arrays = [list(env.actions[key].keys()) for key in env.actions.keys()]
            # Generate all possible combinations
            combinations = list(product(*arrays))
            col = pd.MultiIndex.from_arrays([["_".join(env.agent_variables)]* len(combinations), combinations])
            return pd.DataFrame(index=[str(env.state_for_q_table())], columns=col).fillna(0)

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
                print("exploration_prob argument as 1 element. Minimum probabily is set to 0.1")
                self.min_prob = 0.1
                self.max_prob = self.exploration_prob
            else:
                print("expect 2 elements for exploration_prob !!! Use default value")
                self.min_prob = 0.1
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
        if verbose is True:
            self.monitor_action = pd.DataFrame(columns = self.env.state_variables + ["proba", "action"])
        else:
            self.monitor_action = None
        self.monitor_iter = []
        self.states_for_all_episodes = []
        self.q_table_for_all_episodes = []
        self.q_table = self.global_q_tables(env)
        self.q_table_old = self.global_q_tables(env)
        self.loss_train = None
        self.loss_episodes = []

    def get_epsilon(self,
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
    
    def plot_convergence(self, start : int = 0, end : int = None) -> None:
        """
        """
        if end is None:
            end = len(self.loss_episodes)
        plt.plot(np.arange(start, end), self.loss_episodes[start, end])
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title("Loss evolution")
        plt.show()

    def reinitialize(self):
        self.env.reset()
        self.states_for_all_episodes = []
        self.q_table_for_all_episodes = []
        self.q_table = self.global_q_tables(self.env)
        self.q_table_old = self.global_q_tables(self.env)

    def call_choose_action(self,
                        states : tuple,
                        proba : float) -> list:
        """
        Choose an action based on the current Q-values using epsilon-greedy exploration strategy.
        """
        # action_spaces = self.env.action_space
        if np.random.uniform(0, 1) < proba:
            return [str(np.random.choice(self.env.action_space[key]))  for key in self.env.action_space.keys()]
        elif self.q_table.loc[[str(states)]].replace(0, np.nan).isna().sum(axis=1).to_list()[0] == self.q_table.shape[1]:
            # if no value exist then choose random action
            return [str(np.random.choice(self.env.action_space[key]))  for key in self.env.action_space.keys()]
        else:
            col = self.q_table.replace(0, np.nan).loc[str(states)].argmax()
            return list(self.q_table.loc[ [str(states)]].columns[col][1])

    # def reset_envrionement_multi_agent(self):
    #     num_agents = len(self.env.action_space.keys())
    #     states = {key : self.env.state_for_q_table() for key, _ in zip(self.env.action_space.keys(), range(num_agents))}
    #     # states, done, current_iter, iter_out_of_bound
    #     return states, [False] * len(self.env.action_space.keys()), 0, 0

    def update_q_values(self,
                        current_state : str,
                        next_state : str,
                        actions : list,
                        reward : np.ndarray) -> None:
        """
        Update Q-values table (pandas dataFrame) based on the Q-learning update rule.
        https://deeplizard.com/learn/video/mo96Nqlo1L8

        Args:
            trigger_variable (str): agent name
            state (tuple): Current state representation for the Q-table.
            next_state (tuple): Next state representation for the Q-table.
            action (list): List of actions taken by agent.
            reward (float): Immediate reward.
        """
        # 1st term
        level0_col = self.q_table.columns[0][0]
        current_q_value = self.q_table.loc[
            [current_state],
            [(level0_col, tuple(actions))]
        ].to_numpy().flatten()[-1]
        # 2nd term
        # get maximum value for next state
        next_max_q_value = self.q_table.loc[str(next_state)].max()
        # replace NaN values by zero to avoid error
        if np.isnan(current_q_value):
            current_q_value = 0
        if np.isnan(next_max_q_value):
            next_max_q_value = 0
        if next_max_q_value == 0:
            # save a temporary Qlearning objet
            tmp = copy.deepcopy(self)
            col = [action[1] for action in self.q_table.loc[[next_state]].columns]
            indice = self.q_table.loc[str(next_state)].argmax()
            # compute next state and its reward value
            _, next_rewards, done, problem, info = tmp.env.step(col[indice])
            next_max_q_value = sum(next_rewards.values())
        # 3rd term
        updated_q = self.learning_rate * (reward + self.discount_factor * next_max_q_value)
        # update q_table
        self.q_table.loc[
            [current_state],
            [(level0_col, tuple(actions)) ]
        ] = (1 - self.learning_rate) * current_q_value + updated_q

    def control_loop(self,
                    current_iter : int) -> bool:
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
            self.loss_train = np.abs(np.sum([self.q_table - self.q_table_old])) 
            if self.loss_train <= self.convergence_criterion and current_iter >= np.min([100, 
                                                                int(self.run_limit/1.5)]):
                return True
        if current_iter > self.run_limit:
            print(f"Episode did not converged. You should try to increase run_limit")
            print("stop_criterion ", current_iter)
            self.loss_train = np.abs(np.sum([self.q_table - self.q_table_old])) 
            print("diff_q ", self.loss_train)
            return True
        self.loss_train = np.nan
        return False

    def iterate_all_possibility(self) -> list[list[bool], np.ndarray]:
        """
        Generate all possible combinations
        """
        all_combinations = list(product(*self.env.actions.values()))
        list_actions = []
        # Print the result
        for combination in all_combinations:
            next_env, rewards, done, problem, info = self.env.step(
                np.array(combination).flatten(), 
                self.env.agent_variables
                )
            self.env.delete_last_states()
            if any(problem):
                continue
            else:
                list_actions.append(np.array(combination).flatten())
        if len(list_actions) == 0:
            return [True], None
        else:
            # retrun new action action to take
            return [False], random.choice(list_actions)

    def correct_last_attempt(self, actions : list[str], current_iter : int = 0) -> list[dict[float], float, bool, bool, str]:
        # boundaries limit. Attempt to correct it
        self.env.delete_last_states()
        done, new_actions  = self.iterate_all_possibility()
        if any(done):
            print("No action possible. Stop episode at {0}th iterations".format(current_iter))
            print("Save last state even is out of bound")
            return self.env.step(list(actions), self.env.agent_variables)
        else:
            actions = new_actions
            return self.env.step(actions, self.env.agent_variables)
    
    def record_actions(self, proba : float, actions : tuple) -> None:
        """
        save all actions into a dataFrame row
        """
        tmp = pd.DataFrame({var : self.env.discretized_observation(True)[var] for var in self.env.state_variables})
        tmp["proba"] = proba
        tmp["action"] = str(actions)
        self.monitor_action = pd.concat(
            [self.monitor_action, tmp], 
            axis = 0,
            ignore_index = True)
        
    def training_q_learning(self, proba : float = 0.1) -> int:
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
            #states = self.env.state_for_q_table()
            # Choose actions for each agent based on the global policy
            actions = self.call_choose_action(self.env.state_for_q_table(), proba)
            # compute new states according to agents's action
            next_env, rewards, done, problem, info = self.env.step(actions, self.env.agent_variables)
            if any(problem):
                # trouble with boundaries limit
                _, rewards, done, problem, info = self.correct_last_attempt(actions, current_iter)
            # check if next states is present in dataFrame
            if not self.global_q_tables().index.to_list()[0] in self.q_table.index.to_list():
                # add new empty line into data Frame before to compute score
                self.q_table = pd.concat([self.q_table, self.global_q_tables()])
            self.update_q_values(
                str(self.env.state_for_q_table(start = -2, end = -1)),
                str(self.env.state_for_q_table()),
                actions,
                sum(rewards.values())
            )
            # save all actions done during the episode
            if self.monitor_action is not None:
                self.record_actions(proba, actions)
            if any(done):
                # compute loss
                if len(set(self.q_table.index) - set(self.q_table_old.index)) == 0:
                    self.loss_train = np.abs(np.sum([self.q_table - self.q_table_old]))
                    print("diff Qtable_N - Qtable_N-1 ", np.abs(np.sum([self.q_table - self.q_table_old])))
                else:
                    print("Q table still growths")
                self.loss_train = np.sum([np.sum(val) for val in self.env.rewards.values()])
                print("Loss : {0}".format(self.loss_train))
                self.q_table_old = self.q_table.copy()
                continue
            done = [self.control_loop(current_iter)]
            current_iter  += 1
            self.q_table_old = self.q_table.copy()
        return current_iter

    def q_learning(self) -> None:
        """
        Perform Q-learning algorithm to estimate Q-values and learn an optimal policy.

        """
        proba = self.get_epsilon()
        for episode in range(self.num_episodes):
            print("Episode {0}/{1}".format(episode+1, self.num_episodes))
            print("exploration_prob : {0:.3f}".format(proba[episode]))
            # save information 
            current_iter = self.training_q_learning(proba[episode])
            # NEXT UPDATE SAVE Q8TALBE ACCORDING A FREQUENCY
            self.states_for_all_episodes.append(
                pd.concat(
                    [
                        pd.DataFrame(self.env.all_states()),
                        pd.DataFrame(self.env.rewards).add_prefix('reward_')
                    ], axis = 1)
            )
            self.q_table_for_all_episodes.append(self.q_table.copy().replace(self.q_table.values.min() - 100, np.nan))
            # self.all_episodes.append(self.q_table.copy())
            ###############################################
            # self.max_prob -= self.decrease_prob_exp
            print("end while loop iteration : ", current_iter)
            self.loss_episodes.append(self.loss_train)
            self.monitor_iter.append(current_iter)
            # check if we can stop training early
            # look last 6 episodes and check
            if len(self.loss_episodes) > 7:
                # tmp = np.sum(np.array(self.monitor_iter[-6:-1]) == np.array(self.monitor_iter[-5:]))  
                # if tmp == 5 and current_iter < self.run_limit -1:
                if all(
                    np.abs(
                        np.array(self.loss_episodes[-6:-1]) - np.array(self.loss_episodes[-5:])
                        ) < self.convergence_criterion):
                    print("Look like nothing to learn anymore, stop training")
                    break
        #replace 0 by nan
        self.q_table.replace(self.q_table.values.min() - 100, np.nan)

    
    def pivot_index_dataframe(self, convert_state_variables = True, convert :str = "float", episode : int = -1) -> pd.DataFrame:
        """Convert q_table index to columns and return a data frame.
        Args:
            episode (int, optional): q_table to use. Defaults to -1.
            convert_state_variables (True, optional): by default state_variables have string type. 
            Convert columns type (by default is float) 

        Returns:
            pd.DataFrame:
        """
        tmp = copy.deepcopy(self.q_table_for_all_episodes[episode].reset_index())
        tmp["index"] = tmp["index"].str.replace("\[|\]|\(|\)", "", regex= True)
        tmp = tmp["index"].str.split(",",expand=True)
        tmp.columns = self.env.state_variables
        if convert_state_variables:
            tmp = tmp.astype({var : convert for var in self.env.state_variables} )            
        return pd.concat(
            [tmp, 
             self.q_table_for_all_episodes[-1].reset_index(drop = True)],
            axis = 1)
