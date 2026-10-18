## Theory: Passive Adaptive Dynamic Programming (Passive ADP)

Passive Adaptive Dynamic Programming (ADP) is a reinforcement learning technique in which an agent follows a fixed policy and learns the environment model from its experiences. It estimates transition probabilities and rewards, then calculates the utility of each state using the Bellman equation.

### 1. State Transition Model

The agent records the current state, next state, and reward received after each action.

Each experience is represented as:

`(Current State, Next State, Reward)`

### 2. Transition Probability Estimation

The transition probability represents the probability of moving from state `s` to state `s'` while following the fixed policy.

**Formula:**

P(s' | s) = N(s, s') / N(s)

Where:
- `N(s, s')` = Number of observed transitions from state `s` to state `s'`.
- `N(s)` = Total number of observed transitions from state `s`.

### 3. Reward Model Estimation

The reward model estimates the expected immediate reward received in a particular state by calculating the average of the observed rewards.

**Formula:**

R(s) = Total Rewards Observed in State s / Number of Observations in State s

### 4. Bellman Equation

The Bellman equation is used to calculate the utility of each state based on its immediate reward and the expected utility of its possible next states.

**Formula:**

U(s) = R(s) + γ Σ P(s' | s) U(s')

Where:
- `U(s)` = Utility of the current state.
- `R(s)` = Expected immediate reward.
- `γ` = Discount factor, where 0 ≤ γ < 1.
- `P(s' | s)` = Probability of transitioning to the next state.
- `U(s')` = Utility of the next state.
- `Σ` = Sum over all possible next states.

### 5. Utility Estimation

Initially, all state utilities are set to zero. The program repeatedly applies the Bellman equation using the learned transition probabilities and reward model until the utility values converge or a maximum number of iterations is reached.

### 6. Working Procedure

1. Initialize the fixed policy and environment states.
2. Record state transitions and observed rewards.
3. Count the transitions between states.
4. Estimate transition probabilities from the recorded experiences.
5. Estimate the reward model using average observed rewards.
6. Initialize state utilities.
7. Apply the Bellman equation iteratively.
8. Stop when the utility estimates converge.
9. Display the learned model and final state utilities.

### Conclusion

Passive Adaptive Dynamic Programming learns the environment model from observed experiences and estimates state utilities under a fixed policy. It combines transition probability estimation, reward estimation, and the Bellman equation to evaluate the expected long-term rewards of the states.
