"""
agent.py
========
Defines the Environment class, which is the single source of truth for the
rocket landing simulation.

Design philosophy
-----------------
Rather than hard-coding the rocket physics, every equation (state transitions,
reward functions, action mappings) is read from a JSON configuration file and
compiled at runtime into a Python module called `next_state_compute.py`.
This means the environment can be fully reconfigured — different physics,
different reward weights, different action spaces — without touching this file.

State representation
--------------------
Every variable (position, speed, fuel, …) is stored as a 1-D NumPy array
that grows by one element each timestep.  This gives the full episode history
for free: index -1 is always the current value, index 0 is the initial value.

Key concepts
------------
- states_variables : physics variables used as RL state (pos_x, pos_y, …)
- agent_variables  : actuator variables chosen by the agents (booster, alpha)
- variable_names   : every variable tracked in the simulation (superset)
- start_pos        : initial values, used by reset() to restart an episode
- current_pos      : flat array of the latest state + action values
- rewards          : dict {agent_var: reward_history_array}, grows each step
"""

import numpy as np
import json
import os
import sys
import re
import copy


class Environment():
    """Rocket landing simulation environment driven by a JSON config.

    The environment compiles physics and reward equations from the JSON into
    `next_state_compute.py` at initialisation time, then imports and calls
    those functions on every `step()`.  This keeps the simulation logic
    fully data-driven and easy to reconfigure from the Streamlit UI.

    Attributes
    ----------
    json              : dict   Full JSON config as loaded.
    states_variables  : list   RL state variable names (used to build Q-table index).
    agent_variables   : list   Actuator variable names (one per agent/policy).
    variable_names    : tuple  All tracked variable names (state + agent + auxiliary).
    action_to_take    : dict   Mapping from agent variable to action assignment rule.
    rewards           : dict   {agent_var: np.ndarray} reward history per step.
    start_pos         : dict   Initial values for state + agent variables.
    current_pos       : np.ndarray  Flat array of latest state + action values.
    action_space      : dict   {agent_var: n_discrete_actions}.
    actions           : dict   {agent_var: {action_key: action_value}}.
    lower_lim         : np.ndarray  Per-variable lower bounds (from config "limit").
    upper_lim         : np.ndarray  Per-variable upper bounds (from config "limit").
    n_bins            : np.ndarray  Per-variable number of discretisation bins.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Config validation (optional, call with check_model=True)
    # ──────────────────────────────────────────────────────────────────────────

    def check_input(self):
        """Validate the JSON config and run a dry-run of all equations.

        Checks performed
        ----------------
        1. Variable ordering: states_variables, initial_values, and limit must
           list variables in the same order so that array indexing is consistent.
        2. Limit field completeness: every entry in "limit" must have exactly
           3 fields [min, max, n_bins].
        3. Boundary sanity: the initial state must lie within [lower, upper] for
           every variable.
        4. Equation execution: both compute_equations_variables and
           compute_equations_rewards are called once with the current state to
           catch any syntax or name errors in the compiled module.

        This method is not called automatically; pass check_model=True to
        __init__ to run it at startup, which is useful during development.
        """
        import os
        sys.path.insert(1, os.getcwd())
        from next_state_compute import compute_equations_variables, compute_equations_rewards

        # --- 1. Check that states_variables order matches initial_values and limit ---
        print("check order names for states_variables")
        for i in range(len(self.states_variables)):
            if not self.states_variables[i] == list(self.all_states().keys())[i] == list(self.json["limit"].keys())[i]:
                print("Warning : State variable order are not the  (not a fatal error):")
                print("state_variables : {0}, initial_value: {1}, limit: {2}".format(
                    self.states_variables[i], 
                    list(self.all_states().keys())[i],
                    list(self.json["limit"].keys())[i]
                ))

        # --- 2. Check that agent_variables order is consistent ---
        print("\nCheck order names for agent_variables")
        for i in range(len(self.agent_variables)):
            if not self.agent_variables[i] == list(self.all_states().keys())[len(self.states_variables) + i] == list(self.json["limit"].keys())[len(self.states_variables) + i]:
                print("Warning : agent variable order are not the same:")
                print("Warning : agent_variables : {0}, initial_value: {1}, limit: {2}".format(
                    self.agent_variables[i], 
                    list(self.all_states().keys())[len(self.states_variables) + i],
                    list(self.json["limit"].keys())[len(self.states_variables) + i]
                ))

        # --- 3. Check that every limit entry has exactly 3 fields ---
        print("\ncheck limit number of field")
        if not (np.array([len(val) for key, val in self.json['limit'].items()]).flatten() == 3).all():
            print("Error ! Expect 3 filed for limit. [minimum, maximum, number_bins]")
            sys.exit()

        # --- 4. Check that the initial state lies within the declared bounds ---
        print("\nCheck limit boundaries for initial state")
        tmp = [self.json["limit"][variable][1] < self.last_state()[variable] 
                       for variable in self.json["limit"].keys()]
        if any(tmp):
            print("Error in upper limit value given")
            print([list(self.all_states().keys())[i] for i in np.where(tmp == False)[0]] )
        tmp = [self.json["limit"][variable][0] > self.last_state()[variable] 
                       for variable in self.json["limit"].keys()]
        if any(tmp):
            print("Error in lower limit value given")
            print([list(self.all_states().keys())[i] for i in np.where(tmp == False)[0]] )

        # --- 5. Dry-run both equation functions to catch compilation errors ---
        print("\nSolve equations present in equations_variables field")
        _ = compute_equations_variables(copy.deepcopy(self.last_state()))
        print("\nSolve equations present in equations_rewards field")
        _ = compute_equations_rewards(copy.deepcopy(self.last_state()))
        print("\nEverything is good :)")

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self, json_file, delimiter="$", check_model=False):
        """Initialise the environment from a JSON config file or dictionary.

        The constructor does three things:
          1. Parses the JSON and stores every initial value as a NumPy array
             instance attribute (e.g. self.pos_x, self.pos_y, …).
          2. Compiles all physics/reward/action equations from the JSON into
             `next_state_compute.py` and imports it.
          3. Builds derived attributes: action space, discretisation bins,
             lower/upper limits, start position snapshot.

        Parameters
        ----------
        json_file : str or dict
            Path to a JSON config file, or a config dictionary directly
            (as produced by the Streamlit UI).
        delimiter : str
            Character used to mark variable placeholders inside equation
            strings, e.g. "$pos_x$". Defaults to "$".
        check_model : bool
            If True, run check_input() after loading to validate the config.
        """

        # ── Inner helper: replace $var$ placeholders with Python dict lookups ──
        def replace2dico(equation, state: dict, delimiter: str = "$") -> str:
            """Translate a config equation string into executable Python code.

            The JSON stores equations like:
                "$speed_x$ = ($F$ / (5 + $weight_rocket$)) * $booster$"

            This function replaces each $name$ token with either:
              - `state["name"]`  if "name" is a key in the current state dict
                                 (i.e. a live simulation variable)
              - `name`           if it is a local/temporary variable that will
                                 be defined earlier in the same generated function

            Parameters
            ----------
            equation  : str   Raw equation string from the JSON.
            state     : dict  Current state dictionary (used to know which
                              names are state keys vs local temporaries).
            delimiter : str   Placeholder delimiter character.

            Returns
            -------
            str  Equation string with all $name$ tokens replaced, ready to
                 be written into the generated Python module.
            """
            # Collect the index of every delimiter occurrence in the string
            index_delimiter = [
                i for i in range(len(equation)) 
                if equation.startswith(delimiter, i)
            ]
            # Delimiters must come in pairs: $open$ ... $close$
            if len(index_delimiter) % 2 != 0:
                print("Error number of delimit is odd : " + equation)
                sys.exit()
            # Iteratively replace each $name$ token from left to right
            while len(index_delimiter) > 0:
                # Extract the full $name$ token (including delimiters)
                pattern = equation[index_delimiter[0]: index_delimiter[1] + 1]
                # Strip delimiters to get the bare variable name
                if pattern[1:-1] in state.keys():
                    # Known state key → translate to dict lookup
                    equation = equation.replace(
                        pattern,
                        "state[\"" + pattern[1:-1] + "\"]")
                else:
                    # Temporary / intermediate variable → keep the bare name
                    equation = equation.replace(
                        pattern,
                        pattern[1:-1])
                # Recompute delimiter positions after the replacement
                index_delimiter = [i for i in range(len(equation)) if equation.startswith(delimiter, i)]
            return equation

        # ── Inner helper: build the full next_state_compute.py source code ────
        def compile_equation(json_file: dict, last_state: dict, delimiter: str = "$") -> str:
            """Generate the source code of `next_state_compute.py` as a string.

            The generated module contains three functions:

            compute_equations_variables(state) -> dict
                Applies the physics update equations in order (acceleration,
                new velocities, new positions, fuel consumption, etc.) and
                writes results back into `state`.  All $var$ references become
                `state["var"]` dict accesses.

            compute_equations_rewards(state) -> dict
                Computes the reward value for each agent variable using the
                current state after the physics update.  Same substitution
                logic as above.

            compute_action(state, action, trigger_var) -> dict
                Applies the discrete action value for a given agent variable
                (e.g. sets state["booster"] = 2.0 for full thrust).

            Parameters
            ----------
            json_file  : dict  Parsed JSON config containing equation dicts.
            last_state : dict  Current state dict, used by replace2dico to
                               distinguish state keys from local temporaries.
            delimiter  : str   Placeholder delimiter character.

            Returns
            -------
            str  Full Python source code ready to be written to a .py file.
            """
            # --- compute_equations_variables: physics update step ---
            tmp = "\nimport numpy as np\nimport json\n\ndef compute_equations_variables(state) -> dict:\n"
            for key, var in json_file['equations_variables'].items():
                # Each line: translated_lhs = np.array([ translated_rhs ]).flatten()
                # Wrapping in np.array([...]).flatten() normalises scalar vs
                # array outputs so all state values are consistently 1-D arrays.
                tmp += "\t" + replace2dico(key, last_state, delimiter) + " = np.array([ " + replace2dico(var, last_state, delimiter) + " ]).flatten()\n"
            tmp += "\treturn state\n"

            # --- compute_equations_rewards: reward computation ---
            tmp += "\n\ndef compute_equations_rewards(state) -> dict:\n"
            for key, var in json_file['equations_rewards'].items():
                tmp += "\t" + replace2dico(key, last_state, delimiter) + " = np.array([ " + replace2dico(var, last_state, delimiter) + " ]).flatten()\n"
            tmp += "\treturn state\n"

            # --- compute_action: apply a discrete action for one agent ---
            tmp += "\n\ndef compute_action(state : dict, action : float, trigger_var : str) -> dict:\n"
            for trigger_var in json_file['action_to_take'].keys():
                # One if-branch per agent variable so the function knows which
                # state key to overwrite with the chosen action value.
                tmp += '\tif trigger_var == \"' + trigger_var + '\":\n'
                for key, var in json_file['action_to_take'][trigger_var].items():
                    tmp += "\t\t" + replace2dico(key, last_state, delimiter) + " = np.array([ " + replace2dico(var, last_state, delimiter) + " ]).flatten()\n"
            tmp += "\treturn state\n"
            return tmp

        # ── Inner helper: write the generated source to disk and import it ────
        def save_function_to_file(json_file: dict, last_state: dict, filename='next_state_compute.py', delimiter="$") -> None:
            """Write the generated next_state_compute.py to the working directory.

            Writing to a .py file (rather than using exec/eval) lets Python
            import it as a normal module, which is faster and easier to debug —
            you can open next_state_compute.py and inspect the exact equations
            that will be executed during the simulation.

            Parameters
            ----------
            json_file  : dict  Parsed JSON config.
            last_state : dict  Current state snapshot for placeholder resolution.
            filename   : str   Output filename. Defaults to 'next_state_compute.py'.
            delimiter  : str   Placeholder delimiter character.
            """
            function_string = compile_equation(json_file, last_state, delimiter=delimiter)
            with open(filename, 'w') as file:
                file.write(function_string)

        # ── Load JSON: accept either a file path or a dict (from Streamlit UI) ──
        if isinstance(json_file, str):
            with open(json_file, 'r') as config_file:
                syst_dic = json.load(config_file)
        elif isinstance(json_file, dict):
            # The Streamlit UI passes the config as a Python dict directly
            syst_dic = json_file
        else:
            print("expect JSON file or a dictionary")

        # ── Store the full config for later reference (e.g. limit checks) ───
        self.json = syst_dic

        # ── Extract the variable name lists ─────────────────────────────────
        # states_variables: the subset of variables that form the RL state
        #   (used to build Q-table row keys and DQN input vectors)
        self.states_variables = syst_dic["states_variables"]

        # agent_variables: the actuators each RL agent controls
        #   (one reward signal is computed per agent variable)
        self.agent_variables = syst_dic["agent_variables"]

        # variable_names: every tracked variable (state + agent + auxiliaries
        #   like acceleration, fuel, future positions, …)
        self.variable_names = tuple([key.replace(delimiter, '') for key in syst_dic["initial_values"].keys()])

        # action_to_take: describes how to write an action value into state
        #   e.g. {"booster": {"$booster$": "$action$"}}
        self.action_to_take = syst_dic["action_to_take"]

        # ── Initialise each variable as a NumPy array instance attribute ────
        # strip the delimiter from key names so "$pos_x$" → self.pos_x
        initial_system = {tmpkey.replace(delimiter, ''): value for tmpkey, value in syst_dic["initial_values"].items()}
        for key, value in initial_system.items():
            # Store as a NumPy array so that np.append() grows the history
            # automatically on each step without special-casing scalars.
            setattr(self, key, np.array(value))

        # ── Compile and write the physics module, then import it ─────────────
        # This must happen after all initial attributes are set because
        # last_state() is called inside save_function_to_file to resolve
        # which names are state keys vs temporary variables.
        save_function_to_file(self.json, self.last_state(), filename='next_state_compute.py', delimiter=delimiter)
        from next_state_compute import compute_equations_variables, compute_equations_rewards, compute_action

        # ── Optional config validation (for development / debugging) ─────────
        if check_model:
            self.check_input()

        # ── Compute initial rewards for each agent using the compiled module ─
        # This runs the reward equations once at t=0 so self.rewards is already
        # populated before the first step() call.
        self.rewards = {
            agent_var: compute_equations_rewards(copy.deepcopy(self.last_state()))[agent_var]
            for agent_var in self.agent_variables
        }

        # ── Snapshot the start position (state + agent variables only) ───────
        # reset() will copy these values back to restore the initial condition
        # at the beginning of every new training episode.
        self.start_pos = {
            key: list(initial_system[key])
            for key in self.states_variables + self.agent_variables
            if key in initial_system.keys()
        }

        # ── Flat array of current state + action values ──────────────────────
        # Kept for compatibility with gym-style interfaces that expect a
        # single vector rather than a dictionary.
        self.current_pos = np.array([
            np.round(initial_system[key], 6)
            for key in self.states_variables + self.agent_variables
            if key in initial_system
        ]).flatten()

        # ── Action space metadata ─────────────────────────────────────────────
        # action_space: {agent_var: number_of_discrete_actions}
        self.action_space = {tmpkey.replace(delimiter, ''): len(value) for tmpkey, value in syst_dic["n_action"].items()}
        # actions: {agent_var: {action_key_str: action_float_value}}
        #   e.g. {"booster": {"0": 0.0, "1": 1.0, "2": 2.0}}
        self.actions = {tmpkey.replace(delimiter, ''): value for tmpkey, value in syst_dic["n_action"].items()}

        # ── Discretisation bounds (used by discretized_space / observation) ──
        # lower_lim / upper_lim are parallel arrays aligned to
        # states_variables + agent_variables, in that order.
        self.lower_lim = np.array([
            list(syst_dic['limit'][key])[0]
            for key in self.states_variables + self.agent_variables
            if key in syst_dic['limit']
        ]).flatten()
        self.upper_lim = np.array([
            list(syst_dic['limit'][key])[1]
            for key in self.states_variables + self.agent_variables
            if key in syst_dic['limit']
        ]).flatten()

        # ── Number of bins per variable for tabular Q-learning ───────────────
        # Each variable's continuous range is divided into n_bins equally-spaced
        # values; the nearest bin centre is used as the Q-table key.
        if all([len(item) == 3 for item in syst_dic['limit'].values()]):
            # The config explicitly specifies n_bins as the third limit field
            self.n_bins = np.array([
                list(val)[2]
                for key, val in syst_dic['limit'].items()
                if key in self.states_variables + self.agent_variables
            ]).flatten()
        else:
            # Fallback: use one bin per integer unit of the range
            print("number of bins was not specified for all variable.")
            print("Use formula: \"upper_lim - lower_lim + 1\" to discretize space")
            self.n_bins = self.upper_lim - self.lower_lim + 1

    # ──────────────────────────────────────────────────────────────────────────
    # Episode lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset the environment to the initial state for a new episode.

        Restores all instance attributes to their t=0 values (the first
        element of each variable's history array) and recomputes the initial
        rewards.  Called at the start of every training episode.
        """
        from next_state_compute import compute_equations_rewards
        # Restore each variable to its initial value (index 0:1 → 1-element array)
        for key, value in self.select_states(0, 1).items():
            setattr(self, key, np.array(value))
        # Recompute rewards at t=0 so self.rewards is a fresh 1-element array
        self.rewards = {
            agent_var: compute_equations_rewards(copy.deepcopy(self.last_state()))[agent_var]
            for agent_var in self.agent_variables
        }
        # Also reset the flat current_pos snapshot
        self.current_pos = copy.deepcopy(self.start_pos)

    # ──────────────────────────────────────────────────────────────────────────
    # State accessors
    # ──────────────────────────────────────────────────────────────────────────

    def all_states(self, colnames=None):
        """Return the full history array for each tracked variable.

        This is the main data-export method: after an episode, calling
        all_states() returns a dict where each value is the complete
        time-series array for that variable, ready to be wrapped in a DataFrame.

        Parameters
        ----------
        colnames : list or None
            Variable names to include. Defaults to all variable_names.

        Returns
        -------
        dict  {variable_name: np.ndarray of shape (n_timesteps,)}
        """
        if colnames is None:
            colnames = list(self.variable_names) if isinstance(self.variable_names, (list, tuple)) else self.variable_names
        state = {}
        for key in colnames:
            state[key] = self.__dict__[key]
        return state

    def last_state(self, colnames=None):
        """Return a snapshot of the most recent (current) state.

        Each value in the returned dict is a 1-element NumPy array containing
        the last value of that variable.  Using 1-element arrays (rather than
        scalars) keeps everything consistent with the batch-array convention
        used by the compiled equations in next_state_compute.py.

        Parameters
        ----------
        colnames : list or None
            Variable names to include. Defaults to all variable_names.

        Returns
        -------
        dict  {variable_name: np.ndarray of shape (1,)}
        """
        if colnames is None:
            colnames = list(self.variable_names) if isinstance(self.variable_names, (list, tuple)) else self.variable_names
        state = {}
        for key in colnames:
            if isinstance(self.__dict__[key], np.ndarray):
                # Take only the last element, but keep it wrapped in an array
                state[key] = np.array([self.__dict__[key][-1]])
            else:
                state[key] = self.__dict__[key]
        return state

    def select_states(self, start: int = None, end: int = None, colnames=None):
        """Return a slice of the state history arrays.

        Useful for extracting a sub-trajectory (e.g. the initial state for
        reset, or the last two steps for Q-learning update comparisons).

        Parameters
        ----------
        start : int or None  First index of the slice (inclusive). None = beginning.
        end   : int or None  Last index of the slice (exclusive). None = end.
        colnames : list or None  Variables to include. Defaults to all variable_names.

        Returns
        -------
        dict  {variable_name: np.ndarray of the requested slice}
        """
        if colnames is None:
            colnames = list(self.variable_names) if isinstance(self.variable_names, (list, tuple)) else self.variable_names
        state = {}
        for key in colnames:
            if key not in list(self.variable_names):
                continue
            # Apply the requested slice to each variable's history array
            if start is None and end is None:
                state[key] = self.__dict__[key]
            elif start is None and end is not None:
                state[key] = self.__dict__[key][:end]
            elif start is not None and end is None:
                state[key] = self.__dict__[key][start:]
            else:
                state[key] = self.__dict__[key][start:end]
        return state

    def select_rewards(self, start: int = None, end: int = None, colnames=None):
        """Return a slice of the reward history arrays.

        Mirrors select_states() but operates on self.rewards rather than
        the state variable arrays.

        Parameters
        ----------
        start    : int or None  First index of the slice (inclusive).
        end      : int or None  Last index of the slice (exclusive).
        colnames : list or None  Agent variable names to include.
                                 Defaults to all reward keys.

        Returns
        -------
        dict  {agent_var: np.ndarray of the requested reward slice}
        """
        if colnames is None:
            colnames = self.rewards.keys()
        state = {}
        for key in colnames:
            if key not in list(self.rewards.keys()):
                continue
            if start is None and end is None:
                state[key] = self.rewards[key]
            elif start is None and end is not None:
                state[key] = self.rewards[key][:end]
            elif start is not None and end is None:
                state[key] = self.rewards[key][start:]
            else:
                state[key] = self.rewards[key][start:end]
        return state

    # ──────────────────────────────────────────────────────────────────────────
    # State mutation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def uppdate_variables(self, new_state, colnames=None):
        """Append new timestep values to each variable's history array.

        Called at the end of step() after the physics equations have computed
        the next state.  Each variable's attribute grows by one element per
        call, building up the full episode trajectory.

        Parameters
        ----------
        new_state : dict  {variable_name: new_value} for the new timestep.
        colnames  : list or None  Variables to update. Defaults to all variable_names.
        """
        if colnames is None:
            colnames = list(self.variable_names) if isinstance(self.variable_names, (list, tuple)) else self.variable_names
        for attr_name in colnames:
            # np.append returns a new array; setattr replaces the attribute
            setattr(self, attr_name,
                    np.append(getattr(self, attr_name), new_state[attr_name]))

    def delete_last_states(self, colnames=None, end_index: int = -1):
        """Roll back the last appended state (and its corresponding reward).

        Used by the Q-learning trainer when a proposed action leads to an
        out-of-bounds state: the bad step is deleted and the trainer can try
        a different action without polluting the episode history.

        Parameters
        ----------
        colnames  : list or None  Variables to roll back. Defaults to all variable_names.
        end_index : int           Slice end for the truncation (default -1 removes
                                  only the last element).
        """
        if colnames is None:
            colnames = list(self.variable_names) if isinstance(self.variable_names, (list, tuple)) else self.variable_names
        for attr_name in colnames:
            current_value = getattr(self, attr_name)
            setattr(self, attr_name, current_value[:end_index])
        # Also roll back the reward entry that was appended for the same step
        setattr(self, "rewards", {key: values[:end_index] for key, values in self.rewards.items()})

    # ──────────────────────────────────────────────────────────────────────────
    # Discretisation (for tabular Q-learning)
    # ──────────────────────────────────────────────────────────────────────────

    def discretized_space(self, dico=False):
        """Build the bin grid for the entire state + action space.

        Creates n_bins equally-spaced values between lower_lim and upper_lim
        for each variable.  This grid defines the Q-table's possible index
        values: continuous positions are snapped to the nearest bin centre.

        Parameters
        ----------
        dico : bool
            If True, return a dict keyed by variable name.
            If False (default), return a plain list of arrays.

        Returns
        -------
        list[np.ndarray] or dict{str: np.ndarray}
            One bin-centre array per variable.
        """
        low  = self.lower_lim
        high = self.upper_lim
        if dico:
            tmp = [np.linspace(float(l), float(h), int(b)) for l, h, b in zip(low, high, self.n_bins)]
            return {key: val for key, val in zip(self.start_pos.keys(), tmp)}
        else:
            return [np.linspace(float(l), float(h), int(b)) for l, h, b in zip(low, high, self.n_bins)]

    def discretized_observation(self, dico=False, start=-1, end=None):
        """Snap a slice of the state history to the nearest bin centres.

        For each variable, finds the bin centre closest to the actual value
        at each timestep in [start:end].  The result is the discretised
        representation used as Q-table row keys.

        Parameters
        ----------
        dico  : bool   If True, return a dict; if False, return a 2-D array.
        start : int    Start index of the state history slice. Default -1 (current).
        end   : int or None  End index of the slice (exclusive). Default None (to end).

        Returns
        -------
        dict or np.ndarray
            Discretised values of shape (n_timesteps, n_variables) or a dict
            of lists when dico=True.
        """
        val_bins = self.discretized_space()
        list_pos = []
        for i, key in zip(range(len(self.start_pos.keys())), self.start_pos.keys()):
            # Distance of each timestep value to every bin centre: shape (n_steps, n_bins)
            dist  = val_bins[i] - self.select_states(start, end)[key][:, np.newaxis]
            # Index of the closest bin for each timestep
            index = [np.argmin(array) for array in np.abs(dist)]
            list_pos.append([np.round(val_bins[i][val], 6) for val in index])
        if dico:
            return {
                key: list_pos[i]
                for key, i in zip(self.start_pos.keys(), range(len(self.start_pos.keys())))
            }
        else:
            if len(index) == 1:
                # Single timestep → return a 1-D vector
                return np.array(list_pos).reshape(len(index), len(self.start_pos.keys()))[0]
            else:
                # Multiple timesteps → return a 2-D array (n_steps, n_vars)
                return np.array(list_pos).reshape(len(index), len(self.start_pos.keys()))

    def state_for_q_table(self, start=-1, end=None) -> tuple:
        """Return the current discretised state as a hashable tuple.

        This tuple is used as the row index into the Q-table DataFrame.
        Only states_variables are included (not agent_variables) because
        the action is the column, not part of the row key.

        Parameters
        ----------
        start : int       Start index of the state history slice. Default -1 (current).
        end   : int/None  End index of the slice. Default None.

        Returns
        -------
        tuple  One discretised float per state variable, e.g. (75.0, 175.0, 0.0, …)
        """
        labels = self.states_variables
        obs    = self.discretized_observation(dico=True, start=start, end=end)
        # Keep only the state variables (not agent/actuator variables) and
        # take the first (and typically only) element of each bin-centre list
        return tuple([elment[0] for key, elment in obs.items() if key in labels])

    # ──────────────────────────────────────────────────────────────────────────
    # Action application
    # ──────────────────────────────────────────────────────────────────────────

    def move_agent(self, trigger_var: str, action_key: str, temporary_state: dict = None):
        """Apply one agent's chosen action to a temporary state snapshot.

        Calls compute_action() from the compiled module, which writes the
        action's numeric value into the appropriate state key (e.g. sets
        state["booster"] = 2.0).  The result is a modified copy of the
        state dict that still needs physics integration before becoming the
        true next state.

        Parameters
        ----------
        trigger_var     : str   The agent variable being set (e.g. "booster").
        action_key      : str   The discrete action key (e.g. "2" for full thrust).
        temporary_state : dict  State snapshot to modify. Defaults to last_state().

        Returns
        -------
        dict  The updated temporary state with the action value written in.
        """
        import os
        sys.path.insert(1, os.getcwd())
        from next_state_compute import compute_action
        if temporary_state is None:
            temporary_state = self.last_state()
        # Look up the numeric value for this action key and apply it
        if isinstance(action_key, str):
            return compute_action(temporary_state, self.actions[trigger_var][action_key], trigger_var)
        else:
            return compute_action(temporary_state, self.actions[trigger_var][str(action_key)], trigger_var)

    # ──────────────────────────────────────────────────────────────────────────
    # Main simulation step
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, actions: list, agent_variables=None, method: str = "centralized"):
        """Advance the simulation by one timestep.

        Applies the chosen joint action, integrates the physics equations,
        computes per-agent rewards, appends everything to the history arrays,
        and checks termination conditions.

        Two execution modes are supported:

        "centralized" (default)
            All agents act simultaneously: every agent writes its action into
            the temporary state first, then the physics equations are solved
            once on the combined result.  This is the standard multi-agent
            setting where agents act in parallel.

        "sequential"
            Agents act one at a time: each agent's action is applied, the
            physics are integrated, and rewards computed before moving to
            the next agent.  Useful for debugging or asymmetric agent designs,
            but not used in the default training loop.

        Parameters
        ----------
        actions         : list  Discrete action keys for each agent, aligned
                                with agent_variables (e.g. ["2", "0"] for
                                booster=full, alpha=straight).
        agent_variables : list or None
                                Agent variable names. Defaults to self.agent_variables.
        method          : str   "centralized" or "sequential" (see above).

        Returns
        -------
        tuple of (last_state, rewards, done, problem, info)
            last_state : dict   Current state snapshot after the step.
            rewards    : dict   {agent_var: reward_value} for this step.
            done       : list   [bool] — True if the episode should end.
            problem    : list   [bool] — True if the step hit a bounds violation.
            info       : list   [str]  — Human-readable status message.
        """
        import os
        sys.path.insert(1, os.getcwd())
        from next_state_compute import compute_equations_variables, compute_equations_rewards

        # Work on a copy of the last state to avoid modifying history mid-step
        temporary_state = self.last_state()
        rewards = {}
        done    = []
        problem = []
        info    = []

        if agent_variables is None:
            agent_variables = self.agent_variables

        # ── Apply actions and integrate physics ─────────────────────────────
        if method == "centralized":
            # Phase 1: all agents write their actions into the temp state
            for trigger_var, action_key in zip(agent_variables, actions):
                temporary_state = self.move_agent(trigger_var, action_key, temporary_state)

            # Phase 2: run the physics update equations once for the joint action
            solv_eq = compute_equations_variables(copy.deepcopy(temporary_state))
            # Merge updated values back (only keys that exist in both dicts)
            for key in set(solv_eq.keys()) & set(temporary_state.keys()):
                temporary_state[key] = solv_eq[key]

            # Phase 3: compute one reward per agent from the updated state
            rewards = {
                trigger_var: compute_equations_rewards(copy.deepcopy(temporary_state))[trigger_var]
                for trigger_var in agent_variables
            }

        else:
            # Sequential mode: each agent acts, physics run, reward computed, repeat
            for trigger_var, action_key in zip(agent_variables, actions):
                temporary_state = self.move_agent(
                    trigger_var, action_key,
                    copy.deepcopy(temporary_state),
                    self.action_to_take[trigger_var]
                )
                solv_eq = compute_equations_variables(copy.deepcopy(temporary_state))
                for key in set(solv_eq.keys()) & set(temporary_state.keys()):
                    temporary_state[key] = solv_eq[key]
                rewards[trigger_var] = compute_equations_rewards(
                    copy.deepcopy(temporary_state))[trigger_var]

        # ── Persist the new state and rewards into the history arrays ────────
        self.uppdate_variables(temporary_state)
        for key in self.rewards.keys():
            self.rewards[key] = np.append(self.rewards[key], rewards[key])

        # Update the flat current_pos snapshot (state + agent variables only)
        self.current_pos = np.array(
            list(self.last_state(colnames=self.states_variables + self.agent_variables).values())
        ).reshape(-1)

        # ── Bounds check ─────────────────────────────────────────────────────
        # Compare the new state against the [min, max] limits from config.
        # If any variable is outside its bounds the episode is terminated
        # with problem=True so the trainer can penalise / roll back.
        check_upper = [
            self.json["limit"][variable][1] < self.last_state()[variable]
            for variable in self.json["limit"].keys()
        ]
        check_lower = [
            self.json["limit"][variable][0] > self.last_state()[variable]
            for variable in self.json["limit"].keys()
        ]

        # ── Goal check ───────────────────────────────────────────────────────
        # If the config defines stop_episode conditions, check whether all of
        # them are satisfied simultaneously.  This is the successful landing
        # condition (pos_x in landing zone, pos_y near ground, low speeds, …).
        if "stop_episode" in self.json.keys():
            stop_episode = [
                # Single-value condition: use np.isclose for floating-point safety
                np.isclose(self.last_state()[key][0], value[0])
                if len(value) == 1
                # Range condition: check [min, max] inclusively
                else self.last_state()[key][0] >= value[0] and self.last_state()[key][0] <= value[1]
                for key, value in self.json["stop_episode"].items()
            ]
            if all(stop_episode):
                # All conditions met → successful landing
                print("stop episode because agent reach goal")
                info.append("Reach goal")
                done.append(True)
                problem.append(False)
                return self.last_state(), rewards, done, problem, info

        # ── Out-of-bounds termination ─────────────────────────────────────────
        if any(check_upper) or any(check_lower):
            info.append("new position is out of bound")
            done.append(True)
            problem.append(True)
        else:
            # Normal step, episode continues
            info.append("new position")
            done.append(False)
            problem.append(False)

        return self.last_state(), rewards, done, problem, info
