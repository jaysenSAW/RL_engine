# <span style="color:orange">Author</span>

* [Jaysen SAW](https://www.linkedin.com/in/jaysen-sawmynaden-a5409367/)

# <span style="color:orange">How to install it</span>

`pip install -r requierement.txt`

# <span style="color:orange">Example usage</span>

# <span style="color:orange">Reinforcement learning</span>

Reinforcement learning (RL) is a machine learning method that models events as Markov processes. This means that the current state of the system depends only on its state at t-1.
The idea of RL is to learn the actions to take in order to achieve a given objective. For this, our system will be modified by agents, who will perform actions at each time step. These actions are evaluated and scored relative to a target to be reached. After several training episodes, our algorithm will have learned the actions that maximize the reward to achieve the desired objective.
The code implements the Q-learning algorithm. The key element is the Bellman equation serves as a foundation for many RL algorithms. It iteratively updates Q-values based on experiences collected during training. By iteratively improving Q-values using the Bellman equation, RL algorithms can learn effective policies for sequential decision-making tasks

<img src="notebook\frames\rocket_trajectory.gif"/>


## <span style="color:orange">Bellman equation</span>

The Bellman equation is a fundamental concept in reinforcement learning (RL) that helps in understanding how to compute the value of being in a particular state and taking a particular action. The Bellman equation expresses the relationship between the value of a state-action pair and the values of the subsequent state-action pairs that can be reached from it:

$ Q(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} Q(s', a')] $

Where:
- $Q(s, a)$ is the value of taking action $a$ in state $s$, which represents the expected cumulative reward obtained by starting in state $s$, taking action $a$, and then following the optimal policy thereafter.
- $p(s', r | s, a)$ is the transition probability function, representing the probability of transitioning to state $s'$ and receiving reward $r$ when taking action $a$ in state $s$.
- $r$ is the immediate reward obtained after taking action $a$ in state $s$.
- $\gamma$ is the discount factor, representing the importance of future rewards relative to immediate rewards.
- $max_{a'} Q(s', a')$ represents the maximum value of taking any action in the subsequent state $s'$.

Bellman equation states that the value of being in a state-action pair is equal to the immediate reward obtained plus the discounted value of the best possible action that can be taken from the subsequent state.


# <span style="color:orange">Bibliography</span>

* [Reinforcement learning: An introduction 2020](http://incompleteideas.net/book/RLbook2020.pdf)
* [Learn the essentials of Reinforcement Learning!](https://towardsdatascience.com/reinforcement-learning-101-e24b50e1d292)
* [wikipedia Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)

# <span style="color:orange">Special thanks</span>

Thanks for the long talks and suggestion about this code [Fadi N](https://github.com/fadinammour)

