# <span style="color:orange">Author</span>

* [Jaysen SAW](https://www.linkedin.com/in/jaysen-sawmynaden-a5409367/)

# <span style="color:orange">How to install it</span>
`python version >= 3.10`

`pip install -r requierement.txt`

# <span style="color:orange">Example usage</span>

# <span style="color:orange">Reinforcement learning</span>

Reinforcement learning (RL) is a machine learning method that models events as Markov processes. This means that the current state of the system depends only on its state at t-1.
The idea of RL is to learn the actions to take in order to achieve a given objective. For this, our system will be modified by agents, who will perform actions at each time step. These actions are evaluated and scored relative to a target to be reached. After several training episodes, our algorithm will have learned the actions that maximize the reward to achieve the desired objective.
The code implements the Q-learning algorithm. The key element is the Bellman equation serves as a foundation for many RL algorithms. It iteratively updates Q-values based on experiences collected during training. By iteratively improving Q-values using the Bellman equation, RL algorithms can learn effective policies for sequential decision-making tasks

<img src="notebook\frames\rocket_trajectory.gif"/>


## <span style="color:orange">Bellman equation</span>

The Bellman equation is a fundamental concept in reinforcement learning (RL) that helps in understanding how to compute the value of being in a particular state and taking a particular action. The Bellman equation expresses the relationship between the value of a state-action pair and the values of the subsequent state-action pairs that can be reached from it:

$Q(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} Q(s', a')]$

Where:
- $Q(s, a)$ is the value of taking action $a$ in state $s$, which represents the expected cumulative reward obtained by starting in state $s$, taking action $a$, and then following the optimal policy thereafter.
- $p(s', r | s, a)$ is the transition probability function, representing the probability of transitioning to state $s'$ and receiving reward $r$ when taking action $a$ in state $s$.
- $r$ is the immediate reward obtained after taking action $a$ in state $s$.
- $\gamma$ is the discount factor, representing the importance of future rewards relative to immediate rewards.
- $max_{a'} Q(s', a')$ represents the maximum value of taking any action in the subsequent state $s'$.

Bellman equation states that the value of being in a state-action pair is equal to the immediate reward obtained plus the discounted value of the best possible action that can be taken from the subsequent state.

---

## Getting Started

### Prerequisites

```bash
pip install streamlit numpy pandas plotly torch
```

### Run the App

```bash
streamlit run app.py
```

### Typical Workflow

1. **Configure** — Set the rocket's start position, target landing zone, and reward weights in the **Environment** tab.
2. **Explore** — Visualise reward components in the **Reward Explorer** to verify the reward landscape makes sense.
3. **Train** — Select Tabular Q-Learning or DQN, set hyperparameters, and click **Start Training**.
4. **Analyse** — Inspect trajectories and convergence curves in the **Results** tab.
5. **Export** — Download the JSON config for reproducibility.

---

## Configuration (JSON)

The entire simulation is driven by a JSON config. You can edit it in the UI or upload a custom file. Key fields:

| Field | Description |
|-------|-------------|
| `states_variables` | State variables used to build the Q-table index |
| `agent_variables` | Controlled actuators (`booster`, `alpha`) |
| `initial_values` | Starting values for all simulation variables |
| `limit` | `[min, max, n_bins]` for each variable |
| `n_action` | Discrete action values per agent |
| `equations_variables` | Physics update equations (compiled at runtime) |
| `equations_rewards` | Reward computation equations |
| `stop_episode` | Conditions that define a successful landing |

---

## Episode Termination

An episode ends when **all** of the following conditions are met simultaneously:

| Variable | Target Range |
|----------|-------------|
| `pos_x` | 135 – 145 |
| `pos_y` | 0 – 5 |
| `acceleration_y` | −2 to +2 |
| `speed_x` | −10 to +10 |
| `speed_y` | −10 to +10 |

An episode also terminates if the rocket exits the defined state-space bounds.

---


## <span style="color:orange">Programmatic Usage</span>

You can also use the trainers directly outside of the Streamlit app:

```python
import json
from agent import Environment
from Q_learning import QLearningTrainer
from deep_q_learning import DQNTrainer

# Load config
with open("rocket_config.json") as f:
    config = json.load(f)

env = Environment(config)

# --- Tabular Q-Learning ---
trainer = QLearningTrainer(env, num_episodes=200, learning_rate=0.1, discount_factor=0.99)
trainer.q_learning()

# --- Deep Q-Network ---
trainer = DQNTrainer(env, num_episodes=200, hidden_sizes=[128, 128], lr=1e-3)
trainer.dqn_learning()

# Inspect results
import pandas as pd
last_episode = trainer.states_for_all_episodes[-1]
print(last_episode[["pos_x", "pos_y", "speed_x", "speed_y"]].tail())
```


# <span style="color:orange">Bibliography</span>

* [Reinforcement learning: An introduction 2020](http://incompleteideas.net/book/RLbook2020.pdf)
* [Learn the essentials of Reinforcement Learning!](https://towardsdatascience.com/reinforcement-learning-101-e24b50e1d292)
* [wikipedia Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)

# <span style="color:orange">Special thanks</span>

Thanks for the long talks and suggestion about this code [Fadi N](https://github.com/fadinammour)

