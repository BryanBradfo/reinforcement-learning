# Reinforcement Learning (RL): Load Balancing, MDPs, and Multi-Armed Bandits

![Reinforcement learning wallpaper](rf.gif)

This repository contains implementations and explorations of various Reinforcement Learning (RL) concepts, focusing on a load balancing problem modeled as a Markov Decision Process (MDP), along with supplementary exercises on dynamic programming and multi-armed bandits.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Core Project: Load Balancing (BRYAN\_CHEN-Project.ipynb)](#core-project-load-balancing-bryan_chen-projectipynb)
    *   [1. MDP (Model-Based)](#1-mdp-model-based)
        *   [1.1. Policy Evaluation](#11-policy-evaluation)
        *   [1.2. Optimal Control](#12-optimal-control)
        *   [1.3. One-Step Policy Improvement](#13-one-step-policy-improvement)
    *   [2. Tabular Model-Free Control](#2-tabular-model-free-control)
        *   [2.1. Policy Evaluation (TD(0))](#21-policy-evaluation-td0)
        *   [2.2. Optimal Control (Q-Learning)](#22-optimal-control-q-learning)
3.  [Supplementary Exercises](#supplementary-exercises)
    *   [practical_session: Dynamic Programming and Value Iteration](#practical_session-dynamic-programming-and-value-iteration)
        *   [tp1.py: Airline Ticket Pricing](#tp1py-airline-ticket-pricing)
        *   [tp2.py, tp2\_policy.py, tp2\_test.py: Simple 2-State MDP](#tp2py-tp2_policypy-tp2_testpy-simple-2-state-mdp)
4.  [Setup and Dependencies](#setup-and-dependencies)
5.  [How to Run](#how-to-run)
6.  [Key Parameters and Visualizations](#key-parameters-and-visualizations)

## Project Overview

This project explores fundamental Reinforcement Learning algorithms. The main focus is on a load balancing problem where a dispatcher decides which of two servers to send an incoming job to, aiming to minimize the total number of jobs in the system. This problem is tackled using both model-based (Value Iteration, Policy Evaluation) and model-free (TD(0), Q-Learning) approaches.

Additionally, the repository includes:
*   Exercises on dynamic programming (airline ticket pricing).
*   Value iteration for a simple 2-state MDP, analyzing the impact of the discount factor (`gamma`).
*   An extensive study of multi-armed bandit problems, comparing various strategies like epsilon-greedy, UCB, Bayesian Beta Prior (Thompson Sampling), and Gradient Bandit, along with parameter tuning.

## Core Project: Load Balancing (BRYAN_CHEN-Project.ipynb)

This Jupyter Notebook investigates a load balancing scenario with two servers.
*   **State**: $(Q_1, Q_2)$, where $Q_i$ is the number of jobs in server $i$. Max queue size is 20.
*   **Action**: Dispatch a new job to server 1 ($a_1$) or server 2 ($a_2$).
*   **Cost**: $Q_1 + Q_2$ in every time slot.
*   **Dynamics**:
    *   New job arrival probability: $\lambda = 0.3$.
    *   Departure probability from server 1: $\mu_1 = 0.2$ (if $Q_1 > 0$).
    *   Departure probability from server 2: $\mu_2 = 0.4$ (if $Q_2 > 0$).
    *   Only one event (arrival or departure from one server) per time slot.
    *   If $Q_i=20$, no new jobs arrive at server $i$.
*   **Discount Factor**: $\gamma = 0.99$.

### 1. MDP (Model-Based)

#### 1.1. Policy Evaluation
*   **Policy**: Random policy (dispatch to server 1 or 2 with probability 0.5).
*   **Method**: Iterative Policy Evaluation to calculate the value function $V(Q_1, Q_2)$.
*   **Bellman Equation**:
    $$V_{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_{\pi}(s')]$$
    (The notebook implements a version where reward is accrued based on current state,
    $$R(s) = -(Q_1+Q_2)$$).
*   **Output**: Heatmap of the calculated value function.

#### 1.2. Optimal Control
*   **Goal**: Find the optimal policy $\pi^*$.
*   **Method**: Value Iteration Algorithm.
*   **Bellman Optimality Equation**:
    $V^*(s) = \max_{a} \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$
*   **Outputs**:
    *   Heatmap of the optimal value function $V^*$.
    *   Heatmap showing the optimal action for each state $(Q_1, Q_2)$.
    *   Quiver plot overlaying optimal actions on the optimal value function heatmap.
*   **Comparison**: The optimal policy's expected value is shown to be better than the random policy's. $\mathbb{E}[V_{\text{optimal}}] > \mathbb{E}[V_{\text{random}}]$.

#### 1.3. One-Step Policy Improvement
*   Performs a one-step policy improvement on the random policy.
*   The value function resulting from this improved policy is calculated.
*   **Comparison**: Shows that the one-step improved policy performs better than the initial random policy, and its value function is closer to the optimal one. This is because it makes locally optimal decisions based on the current value estimates.

### 2. Tabular Model-Free Control

#### 2.1. Policy Evaluation (TD(0))
*   **Policy**: Random policy (dispatch with probability 0.5).
*   **Method**: TD(0) for policy evaluation.
    $V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$
*   **Learning Rate ($\alpha_n$)**: Explores $\alpha_n = 1/n$, constant $\alpha=1$, and $\alpha_n = 1/\sqrt[4]{n_{sa}}$ (where $n_{sa}$ is the visit count to state-action pair or state).
*   **Simulation**: An environment simulation function `simulate_env` is used.
*   **Outputs**:
    *   Heatmap of the value function learned via TD(0).
    *   Comparisons with value functions from Section 1 (random policy, one-step improved, optimal).
*   **Observations**: The choice of $\alpha_n$ affects convergence and final value estimates. Constant alpha can lead to instability or slow convergence if too large/small. $1/n$ satisfies Robbins-Monro conditions but can be slow. $1/\sqrt[m]{n}$ variations are explored for potentially better empirical performance.

#### 2.2. Optimal Control (Q-Learning)
*   **Goal**: Find the optimal Q-function $Q^*(s,a)$.
*   **Method**: Q-Learning algorithm.
    $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)]$
*   **Exploration**: Epsilon-greedy strategy ($\epsilon = 0.1$).
*   **Learning Rate ($\alpha_n$)**: Explores $\alpha_n = 1/n_{sa}$ (visit count for state-action pair), constant $\alpha=1$, and $\alpha_n = 1/n_{sa}^{1.5}$.
*   **Outputs**:
    *   Heatmaps for $Q(s, a_1)$ and $Q(s, a_2)$.
    *   Heatmap for $\max_a Q(s,a)$ (which is $$V^*(s)$$).
    *   Quiver plot showing the optimal action derived from $Q^*$ for each state.
*   **Observations**: Similar to TD(0), the choice of learning rate schedule impacts performance. Q-learning directly learns the optimal action-value function without a model of the environment.

## Supplementary Exercises

### practical_session: Dynamic Programming and Value Iteration

#### tp1.py: Airline Ticket Pricing
*   **Problem**: Determine the optimal pricing strategy (action 1 or 2 with different success probabilities and revenues) for selling airline tickets over time.
*   **State**: `(temps, sieges)` - time remaining, number of seats available.
*   **Method**: Value Iteration (dynamic programming).
*   **Output**: Heatmap of the optimal action (price level) for each state. Matrix of value functions.

#### tp2.py, tp2_policy.py, tp2_test.py: Simple 2-State MDP
*   **Problem**: A simple 2-state (0, 1) MDP with defined rewards for transitions and actions.
*   **`tp2.py`**: Implements Value Iteration to find the optimal value function and policy for different discount factors (`gamma`).
    *   **Outputs**: Plots of $V(0)$, $V(1)$, optimal action for state 0, and optimal action for state 1 as a function of `gamma`.
*   **`tp2_policy.py` & `tp2_test.py`**: Appear to be alternative implementations or explorations, possibly involving policy iteration or direct matrix solutions for the Bellman equations.

## Setup and Dependencies

This project uses Python 3. The main dependencies are:
*   `numpy`
*   `matplotlib`
*   `jupyter` (for running .ipynb files)

You can install them using pip:
```bash
pip install numpy matplotlib jupyterlab
```

## How to Run

1. Jupyter Notebooks (.ipynb files):
- Navigate to the directory containing the notebook (project/).
- Launch Jupyter Lab or Jupyter Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

- Open the desired .ipynb file and run the cells sequentially.

2. Python Scripts (.py files in practical_session/):

Navigate to the practical_session/ directory.
Run the scripts from the command line:

```bash
python tp1.py
python tp2.py
```


## Key Parameters and Visualizations

*   **Load Balancing Project**:
    *   Parameters: $\lambda, \mu_1, \mu_2, \gamma$, stopping criterion $\delta$ (for iterative methods), learning rate $\alpha_n$, exploration rate $\epsilon$ (for Q-learning).
    *   Visualizations: Heatmaps of value functions, optimal policies, quiver plots of actions.
*   **practical_session Exercises**:
    *   Parameters: `gamma` (discount factor).
    *   Visualizations: Heatmap of optimal actions (ticket pricing), plots of value functions and actions vs. `gamma` (2-state MDP).
*   **Multi-Armed Bandits**:
    *   Parameters: `epsilon` (epsilon-greedy), `c` (UCB), `alpha` (gradient bandit), `q_init` (optimistic initialization), number of arms, reward probabilities `p`.
    *   Visualizations: Plots of mean reward, cumulative reward, and percentage of best arm pulls over time, comparing different algorithms and parameter settings.
