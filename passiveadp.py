
from collections import defaultdict

# Passive Adaptive Dynamic Programming (ADP)

# Fixed policy followed by the robot
policy = {
    "A": "Move_Right",
    "B": "Move_Down",
    "C": "Move_Right",
    "D": "Move_Down"
}

# Experience format: (current_state, next_state, reward)
# These observations are collected while following the fixed policy.
experiences = [
    ("A", "B", -1),
    ("A", "B", -1),
    ("A", "C", -1),
    ("A", "B", -1),
    ("B", "C", -1),
    ("B", "D", -1),
    ("B", "C", -1),
    ("C", "D", 10),
    ("C", "D", 10),
    ("C", "D", 10),
    ("D", "D", 0)
]

states = list(policy.keys())
terminal_states = {"D"}
gamma = 0.9

# Step 1: Record state transitions and rewards
transition_counts = defaultdict(lambda: defaultdict(int))
reward_sum = defaultdict(float)
reward_count = defaultdict(int)

for state, next_state, reward in experiences:
    transition_counts[state][next_state] += 1
    reward_sum[state] += reward
    reward_count[state] += 1

print("Recorded Experiences:")
for experience in experiences:
    print(experience)

# Step 2: Estimate transition probabilities
transition_prob = {}

for state in states:
    total = sum(transition_counts[state].values())

    if total > 0:
        transition_prob[state] = {
            next_state: count / total
            for next_state, count
            in transition_counts[state].items()
        }
    else:
        transition_prob[state] = {}

print("\nTransition Probabilities:")
for state, transitions in transition_prob.items():
    for next_state, probability in transitions.items():
        print(f"P({next_state}|{state}) = {probability:.2f}")

# Step 3: Estimate the reward model
reward_model = {}

for state in states:
    if reward_count[state] > 0:
        reward_model[state] = (
            reward_sum[state] / reward_count[state]
        )
    else:
        reward_model[state] = 0.0

print("\nEstimated Rewards:")
for state in states:
    print(f"R({state}) = {reward_model[state]:.2f}")

# Step 4: Calculate utilities using Bellman equation
utilities = {state: 0.0 for state in states}

# Terminal state utility is fixed at zero.
for state in terminal_states:
    utilities[state] = 0.0

tolerance = 1e-6
max_iterations = 10000

for iteration in range(max_iterations):
    new_utilities = utilities.copy()

    for state in states:
        if state in terminal_states:
            continue

        expected_next_utility = sum(
            probability * utilities[next_state]
            for next_state, probability
            in transition_prob[state].items()
        )

        new_utilities[state] = (
            reward_model[state]
            + gamma * expected_next_utility
        )

    difference = max(
        abs(new_utilities[s] - utilities[s])
        for s in states
    )

    utilities = new_utilities

    if difference < tolerance:
        break

print("\nState Utilities:")
for state in states:
    print(f"U({state}) = {utilities[state]:.4f}")

print("\nIterations:", iteration + 1)
