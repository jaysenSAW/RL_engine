"""
deep_q_learning.py  (PyTorch edition)
======================================
Deep Q-Network (DQN) trainer using PyTorch.

Architecture
------------
The Q-function  Q(s, a)  is approximated by a fully-connected neural network:

    Input layer  : raw continuous state vector  (n_states neurons)
    Hidden layers: configurable depth and width  (ReLU activations)
    Output layer : one Q-value per joint action  (n_actions neurons, linear)

PyTorch advantages over the NumPy implementation
-------------------------------------------------
  - Autograd: no manual backward() / gradient math required. PyTorch tracks
    every operation on tensors and computes exact gradients automatically via
    loss.backward().
  - torch.optim.Adam: battle-tested, fused C++ implementation of Adam — same
    algorithm as before but faster and numerically more stable.
  - GPU support: moving .to(device) makes the entire forward/backward pass run
    on CUDA with zero code changes.
  - torch.no_grad(): inference (action selection, Q-value heatmaps) skips
    gradient bookkeeping entirely, saving memory and time.

Key DQN components implemented
--------------------------------
  1. QNetwork (nn.Module)         — MLP with configurable hidden layers
  2. ReplayBuffer                 — ring buffer, (s, a, r, s', done) tuples
  3. DQNTrainer                   — episode loop, Double-DQN update, target sync
"""

import copy
import random
from collections import deque
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import Environment


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Q-Network  (nn.Module)
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """Fully-connected Q-network: state vector → Q-value per joint action.

    Parameters
    ----------
    n_states : int
        Dimensionality of the continuous state input.
    n_actions : int
        Number of discrete joint actions (output neurons).
    hidden_sizes : list[int]
        Width of each hidden layer, e.g. [128, 128].

    Architecture
    ------------
    Linear(n_states → h0) → ReLU
    Linear(h0 → h1)       → ReLU
    ...
    Linear(hN → n_actions)          ← linear output, no activation

    Weight initialisation
    ---------------------
    He (Kaiming) uniform init on every Linear layer.  For a ReLU network this
    keeps the variance of activations approximately constant across depth,
    preventing vanishing / exploding gradients at initialisation.
    """

    def __init__(self, n_states: int, n_actions: int, hidden_sizes: list = None):
        super().__init__()

        hidden_sizes = hidden_sizes or [128, 128]
        sizes = [n_states] + hidden_sizes + [n_actions]

        layers = []
        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])
            # He (Kaiming) uniform initialisation for weights
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            # Add ReLU after every layer except the last (output is linear)
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, n_states)  →  q : (batch, n_actions)"""
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Experience Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-capacity ring buffer of (s, a, r, s', done) transitions.

    Why replay?
    -----------
    Consecutive environment steps are highly correlated: s_{t+1} is directly
    caused by s_t.  Training a neural network on correlated data produces
    biased gradient estimates and causes catastrophic forgetting of earlier
    experiences.  Storing transitions and drawing *random* mini-batches:
      - Breaks temporal correlations
      - Allows each transition to be reused many times (sample efficiency)
      - Stabilises the loss landscape

    Parameters
    ----------
    capacity : int
        Max transitions stored.  Oldest are silently overwritten.
    device : torch.device
        Tensors returned by sample() are placed on this device.
    """

    def __init__(self, capacity: int = 10_000, device: torch.device = None):
        self.buffer = deque(maxlen=capacity)
        self.device = device or torch.device("cpu")

    def push(self, state, action_idx: int, reward: float,
             next_state, done: bool) -> None:
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action_idx),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        """Return a batch of tensors on self.device."""
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)

        states      = torch.tensor(np.array(s),  dtype=torch.float32, device=self.device)
        actions     = torch.tensor(a,             dtype=torch.long,    device=self.device)
        rewards     = torch.tensor(r,             dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(ns),  dtype=torch.float32, device=self.device)
        dones       = torch.tensor(d,             dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DQN Trainer
# ══════════════════════════════════════════════════════════════════════════════

class DQNTrainer:
    """PyTorch Deep Q-Network trainer, drop-in replacement for QLearningTrainer.

    Key differences from tabular QLearningTrainer
    ----------------------------------------------
    * Continuous state input   — raw float vector, no discretisation / bins.
    * Neural network Q-function — generalises to unseen states.
    * Experience replay buffer  — random mini-batch sampling, no temporal bias.
    * Target network            — frozen copy synced every target_update steps,
                                  provides stable Bellman targets.
    * Double DQN update         — online net selects next action, target net
                                  evaluates it; reduces Q-value over-estimation.
    * GPU-ready                 — moves to CUDA automatically if available.

    Parameters
    ----------
    env : Environment
    num_episodes : int
    hidden_sizes : list[int]
        Hidden layer widths, e.g. [128, 128].
    lr : float
        Adam learning rate.
    discount_factor : float
        γ in the Bellman equation.
    batch_size : int
        Mini-batch size sampled from replay buffer each gradient step.
    buffer_capacity : int
        Max transitions in replay buffer.
    target_update : int
        Hard-copy online → target network every N gradient steps.
    exploration_prob : list[float, float]
        [min_epsilon, max_epsilon].
    decrease_prob_exp : float
        Epsilon decay rate.
    decay_type : str
        'exponential' or 'linear'.
    run_limit : int
        Max steps per episode before forced termination.
    convergence_criterion : float
        Early-stop threshold: stop if last-6-episode loss change < this value.
    seed : int
    """

    def __init__(
        self,
        env: Environment,
        num_episodes: int            = 200,
        hidden_sizes: list           = None,
        lr: float                    = 1e-3,
        discount_factor: float       = 0.99,
        batch_size: int              = 64,
        buffer_capacity: int         = 10_000,
        target_update: int           = 50,
        exploration_prob             = None,
        decrease_prob_exp: float     = 0.005,
        decay_type: str              = "exponential",
        run_limit: int               = 500,
        convergence_criterion: float = 0.5,
        seed: int                    = 42,
    ):
        # ── reproducibility ───────────────────────────────────────────────────
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.env                   = env
        self.num_episodes          = int(num_episodes)
        self.lr                    = float(lr)
        self.discount_factor       = float(discount_factor)
        self.batch_size            = int(batch_size)
        self.target_update         = int(target_update)
        self.run_limit             = int(run_limit)
        self.convergence_criterion = float(convergence_criterion)
        self.decay_type            = decay_type
        self.decrease_prob_exp     = float(decrease_prob_exp)
        self.seed                  = seed
        # Number of gradient updates to perform per environment step.
        # A ratio > 1 improves sample efficiency: the same transition is used
        # to train the network multiple times from different random batches,
        # which accelerates learning without extra environment interactions.
        self.gradient_steps        = 4

        # ── device: CUDA if available, else CPU ───────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── epsilon bounds ────────────────────────────────────────────────────
        if exploration_prob is None:
            exploration_prob = [0.05, 1.0]
        if isinstance(exploration_prob, list) and len(exploration_prob) == 2:
            self.min_prob = float(min(exploration_prob))
            self.max_prob = float(max(exploration_prob))
        else:
            self.min_prob = 0.05
            self.max_prob = 1.0

        # ── action space ──────────────────────────────────────────────────────
        # Enumerate every joint action combination and assign a stable integer
        # index so the network output neuron i always corresponds to the same
        # (booster_action, alpha_action) pair.
        all_keys             = list(env.actions.keys())
        all_vals             = [list(env.actions[k].keys()) for k in all_keys]
        self.joint_actions   = list(product(*all_vals))  # [(b0,a0),(b0,a1),...]
        self.n_actions       = len(self.joint_actions)
        self.agent_keys      = all_keys

        # ── state vector definition ───────────────────────────────────────────
        # Concatenate states_variables and agent_variables in the same order
        # the environment stores them.  This vector is fed directly to the
        # network without discretisation.
        self.state_keys = env.states_variables + env.agent_variables
        self.n_states   = len(self.state_keys)

        # ── networks ──────────────────────────────────────────────────────────
        hidden = hidden_sizes or [128, 128]

        # Online network: updated every gradient step
        self.online_net = QNetwork(self.n_states, self.n_actions, hidden).to(self.device)

        # Target network: frozen copy, periodically synced from online_net.
        # Using a separate network prevents the Bellman targets from "chasing"
        # the predictions — both sides of the loss would otherwise shift
        # together, causing oscillations or divergence.
        self.target_net = QNetwork(self.n_states, self.n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()   # target net is never trained directly

        # ── optimiser ─────────────────────────────────────────────────────────
        # Adam: adaptive per-parameter learning rates.  Far more robust than
        # SGD for RL where gradient magnitudes vary widely across states.
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)

        # ── replay buffer ─────────────────────────────────────────────────────
        self.replay = ReplayBuffer(buffer_capacity, device=self.device)

        # ── state normalisation bounds (must come after env + state_keys set) ─
        self._build_norm_bounds()

        # ── logging (mirrors QLearningTrainer attributes for app.py compat) ───
        self.loss_episodes            : list = []
        self.monitor_iter             : list = []
        self.states_for_all_episodes  : list = []
        self.q_table_for_all_episodes : list = []   # None entries; kept for API compat
        self.train_sucess_episode     : dict = {}
        self.loss_train                      = None
        self._grad_steps                     = 0    # total gradient update counter

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_norm_bounds(self) -> None:
        """Cache per-feature min/max from the environment limits for normalisation.

        Each state feature lives on a very different scale
        (e.g. pos_y ∈ [0, 250], angle ∈ [−0.8, 0.8], weight ∈ [0, 305]).
        Feeding raw values to a neural network makes learning slow and
        numerically unstable because gradient magnitudes differ by orders of
        magnitude across features.  Min-max normalisation maps every feature
        to [0, 1] so all inputs contribute equally at initialisation.
        """
        lim = self.env.json["limit"]
        lo, hi = [], []
        for k in self.state_keys:
            if k in lim:
                lo.append(float(lim[k][0]))
                hi.append(float(lim[k][1]))
            else:
                lo.append(0.0)
                hi.append(1.0)
        self._norm_lo    = np.array(lo, dtype=np.float32)
        self._norm_range = np.array(hi, dtype=np.float32) - self._norm_lo
        # Avoid division by zero for degenerate features
        self._norm_range = np.where(self._norm_range == 0, 1.0, self._norm_range)

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        """Map raw state vector into [0, 1] using precomputed bounds."""
        return (raw - self._norm_lo) / self._norm_range

    def _state_vector(self) -> np.ndarray:
        """Read env.last_state() into a normalised 1-D float32 numpy array.

        Raw physics values (pos_x ~75-200, pos_y ~0-250, weight ~5-305)
        span very different scales. We normalise to [0, 1] using the limits
        defined in the environment config before feeding to the network.
        """
        s = self.env.last_state()
        raw = np.array([float(s[k][-1]) for k in self.state_keys], dtype=np.float32)
        return self._normalise(raw)

    def get_epsilon(self) -> list:
        """Epsilon schedule over all episodes — mirrors QLearningTrainer API."""
        n = self.num_episodes
        if self.decay_type == "exponential":
            return [round(max(self.min_prob,
                              float(np.exp(-self.decrease_prob_exp * i))), 4)
                    for i in range(n)]
        else:
            return [round(max(self.min_prob,
                              self.max_prob - self.decrease_prob_exp * i), 4)
                    for i in range(n)]

    @torch.no_grad()
    def _choose_action(self, state: np.ndarray, epsilon: float) -> int:
        """ε-greedy action selection.

        @torch.no_grad() disables gradient tracking for this inference step —
        no backward pass is needed here, so skipping autograd saves memory and
        is ~2× faster than a normal forward pass.

        With prob ε  → random joint action (exploration)
        With prob 1-ε → argmax Q(s, ·) from online network (exploitation)
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)

        s_tensor = torch.tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)   # (1, n_states)
        q_vals = self.online_net(s_tensor)                          # (1, n_actions)
        return int(q_vals.argmax(dim=1).item())

    def _learn(self) -> float:
        """Sample one mini-batch and perform one Double-DQN gradient step.

        Double DQN update
        -----------------
        Standard DQN computes:
            target = r + γ · max_a Q_target(s', a)

        This leads to over-estimation because the max operator is applied to
        noisy Q-values.  Double DQN separates selection from evaluation:
            a* = argmax_a  Q_online(s', a)    ← online net selects best action
            target = r + γ · Q_target(s', a*) ← target net evaluates it

        Loss
        ----
        Huber loss (smooth_l1) between Q_online(s, a) and the Double-DQN
        target.  Compared to MSE, Huber loss is quadratic for small errors
        (smooth gradients) and linear for large errors (robust to outliers /
        early-training instability).

        Gradient clipping
        -----------------
        clip_grad_norm_(parameters, max_norm=10) rescales the entire gradient
        vector if its L2 norm exceeds 10.  This prevents parameter explosions
        in the early episodes when Q-values are poorly calibrated.
        """
        if len(self.replay) < self.batch_size:
            return float("nan")

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        # ── predicted Q-values for the actions actually taken ─────────────────
        # online_net(states) → (B, n_actions)
        # .gather(1, actions.unsqueeze(1)) → (B, 1) → squeeze → (B,)
        q_pred = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Double-DQN targets ────────────────────────────────────────────────
        with torch.no_grad():
            # Online net selects the best next action
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)  # (B,1)
            # Target net evaluates that action's value
            q_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)  # (B,)
            targets = rewards + self.discount_factor * q_next * (1.0 - dones)

        # ── Huber loss ────────────────────────────────────────────────────────
        loss = F.smooth_l1_loss(q_pred, targets)

        # ── backprop + gradient clipping + Adam step ──────────────────────────
        self.optimizer.zero_grad()   # clear gradients from previous step
        loss.backward()              # autograd computes all gradients
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self._grad_steps += 1

        # ── hard sync online → target every target_update steps ───────────────
        if self._grad_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # ── episode loop ──────────────────────────────────────────────────────────

    def training_dqn(self, epsilon: float, episode: int = None) -> int:
        """Run one full episode with ε-greedy exploration.

        Returns
        -------
        int : steps taken in this episode.
        """
        self.env.reset()
        state     = self._state_vector()
        done      = [False]
        step      = 0
        ep_losses = []

        self.online_net.train()   # enable dropout / batch-norm if used (future-proof)

        while not any(done):
            action_idx  = self._choose_action(state, epsilon)
            action_keys = list(self.joint_actions[action_idx])

            _, rewards, done, problem, info = self.env.step(
                action_keys, self.env.agent_variables
            )

            if any(problem):
                reward_val = -10.0
                done       = [True]
            else:
                reward_val = float(sum(float(np.squeeze(v)) for v in rewards.values()))

            next_state = self._state_vector()
            self.replay.push(state, action_idx, reward_val, next_state, any(done))
            state = next_state

            for _ in range(self.gradient_steps):
                loss_val = self._learn()
                if not np.isnan(loss_val):
                    ep_losses.append(loss_val)

            # Record successful landing
            if info and info[-1] == "Reach goal":
                self.train_sucess_episode[episode] = {
                    "epsilon": epsilon,
                    "states":  copy.deepcopy(self.env.all_states()),
                    "rewards": copy.deepcopy(self.env.rewards),
                }
                break

            step += 1
            if step >= self.run_limit:
                break

        self.loss_train = float(np.mean(ep_losses)) if ep_losses else float("nan")
        return step

    def dqn_learning(self) -> None:
        """Full training loop over all episodes with early stopping."""
        epsilons = self.get_epsilon()

        for episode in range(self.num_episodes):
            eps   = epsilons[episode]
            steps = self.training_dqn(eps, episode)

            ep_df = pd.concat([
                pd.DataFrame(self.env.all_states()),
                pd.DataFrame(self.env.rewards).add_prefix("reward_"),
            ], axis=1)
            self.states_for_all_episodes.append(ep_df)
            self.q_table_for_all_episodes.append(None)
            self.loss_episodes.append(self.loss_train)
            self.monitor_iter.append(steps)

            ls = f"{self.loss_train:.4f}" if not np.isnan(self.loss_train) else "—"
            print(f"Episode {episode+1:>4}/{self.num_episodes} | "
                  f"ε={eps:.3f} | steps={steps} | loss={ls} | "
                  f"buf={len(self.replay)} | grad_steps={self._grad_steps}")

            if len(self.loss_episodes) > 7:
                recent = [l for l in self.loss_episodes[-6:]
                          if l is not None and not np.isnan(l)]
                if len(recent) == 6 and all(
                    abs(a - b) < self.convergence_criterion
                    for a, b in zip(recent, recent[1:])
                ):
                    print("Early stopping: loss converged.")
                    break

    # ── inference helpers (used by app.py Results tab) ────────────────────────

    @torch.no_grad()
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all joint actions given a raw (unnormalised) state vector."""
        self.online_net.eval()
        normed = self._normalise(np.array(state, dtype=np.float32))
        s = torch.tensor(normed, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        return self.online_net(s).squeeze(0).cpu().numpy()

    def get_network_summary(self) -> list[dict]:
        """Return layer-by-layer architecture info for the Results panel."""
        rows = []
        param_count = 0
        for i, module in enumerate(self.online_net.net):
            if isinstance(module, nn.Linear):
                p = module.weight.numel() + module.bias.numel()
                param_count += p
                rows.append({
                    "Layer": len(rows) + 1,
                    "Type": "Linear",
                    "In":   module.in_features,
                    "Out":  module.out_features,
                    "Parameters": p,
                    "Activation": "ReLU" if i < len(list(self.online_net.net)) - 1 else "Linear (output)",
                })
        return rows