from __future__ import division
import numpy as np
import cv2
import gym


def euclidean(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


def expected_reward(scene, action, memory, k):
    # Check if memory empty
    if memory.length == 0:
        return 0

    # Use subspace of memories that took this action
    subspace = memory.memory[memory.memory[:, -2] == action]

    # Check if subspace empty
    if subspace.size == 0:
        return 0

    # Initialize probabilities
    reward_outcomes = []
    weights = []

    # Similarity of each memory to the scene
    for mem in subspace:
        # similarity = np.math.exp(-777*euclidean(scene, mem[:-2]))
        # similarity = 1 - (euclidean(scene, mem[:-2]) / 1000) ** 2
        distance = euclidean(scene, mem[:-2])

        reward_outcomes.append(mem[-1])
        weights.append(distance)

    weights = np.array(weights)

    nearest_neighbors = weights.argsort()[:k]

    expected = 0

    for i in nearest_neighbors:
        expected += reward_outcomes[i]

    return expected / k

    # max_dist = weights.max()
    #
    # # Convert distances to similarities
    # # weights = 1 - (weights / (max_dist + 1)) ** similarity_discount
    # weights = similarity_discount ** weights
    #
    # # Scale similarities such that they can be used as a probability distribution i.e. between [0, 1] and sum to 1
    # weights = np.divide(weights, np.sum(weights))
    #
    # # Compute expectation
    # expected = 0
    # for i in range(len(weights)):
    #     expected += weights[i] * reward_outcomes[i]

    # Return expected reward
    # return expected


class Memory:
    length = 0

    def __init__(self, memory_size, memory_horizon, reward_horizon=1):
        self.memory_size = memory_size
        self.memory_horizon = memory_horizon
        self.reward_horizon = reward_horizon

        self.memory = np.zeros((1, self.memory_size))

    def Update(self, scene, action, reward, reward_discount):
        # Insert scene into memory
        self.memory = np.insert(self.memory, 0, np.append(scene, [action, reward]), axis=0)

        # Remove initial zeros
        if self.length == 0:
            self.memory = np.delete(self.memory, -1, 0)

        # Manage memory size
        if self.length >= self.memory_horizon:
            self.memory = np.delete(self.memory, -1, 0)
        else:
            self.length += 1

        # Add discounted rewards
        for i in range(1, min(self.reward_horizon, self.length)):
            # self.memory[i, -1] += reward * (1 - (i / self.reward_horizon)**reward_discount)
            # self.memory[i, -1] += reward * (np.math.exp(-i))
            self.memory[i, -1] += reward * (reward_discount ** i)

    # Merge and clear
    def Merge(self, m):
        if self.length == 0:
            self.memory = m.memory
        else:
            self.memory = np.concatenate((m.memory, self.memory), axis=0)

        self.length += m.length

        # Delete memories beyond horizon
        if self.length > self.memory_horizon:
            self.memory = self.memory[:-(self.length - self.memory_horizon)]

        return np.zeros((1, self.memory_size))


class Agent:
    def __init__(self, model, global_memory, actions, reward_horizon, reward_discount, k, memory_horizon, memory_size):
        self.global_memory = global_memory
        self.actions = actions
        self.reward_horizon = reward_horizon
        self.reward_discount = reward_discount
        self.k = k
        self.memory_horizon = memory_horizon
        self.memory_size = memory_size

        self.model = model
        self.local_memory = Memory(self.memory_size, self.memory_horizon, self.memory_horizon)

    def Model(self, state):
        return self.model.Update(state)

    def Act(self, scene, decisiveness):
        # Initialize probabilities
        weights = []

        # Get expected reward for each action
        for action in self.actions:
            weights.append(expected_reward(scene, action, self.global_memory, self.k))

        weights = np.array(weights)

        # Shift probabilities such that they are positive
        if weights.min() <= 0:
            weights += weights.min() + 1

        # Increase likelihood of better actions
        weights = weights ** decisiveness

        # Scale probabilities such that they are between [0, 1]
        weights = np.divide(weights, np.sum(weights))

        # Return an action probabilistically
        return np.random.choice(self.actions, p=weights)

    def Learn(self, scene, action, reward):
        self.local_memory.Update(scene, action, reward, self.reward_discount)

    def Finish(self):
        self.local_memory.memory = self.global_memory.Merge(self.local_memory)
        self.local_memory.length = 0
        # self.model.Finish()


# Lower dimension projection of state
class Projection:
    def __init__(self, dimension, size):
        self.dimension = dimension
        self.size = size

    def Update(self, state):
        # Greyscale
        state = np.dot(state[..., :3], [0.299, 0.587, 0.114])

        # Image resize
        state = cv2.resize(state, dsize=self.size)

        # Lower dimension projection
        state = state.flatten()
        projection = np.random.RandomState().randn(self.dimension, len(state)).astype(np.float32)

        return np.dot(projection, state)


# Main method
if __name__ == "__main__":
    # Environment
    env = gym.make('CartPole-v0')
    action_space = [0, 1]
    state_space = 4

    # Global memory
    hippocampus = Memory(state_space + 2, 100000)

    # Agent
    agent = Agent(Projection(state_space, (80, 80)), hippocampus, action_space, 1000, 0.999, 7, 1000, state_space + 2)

    for run_through in range(2000):
        # Initialize environment
        s = env.reset()

        rewards = []

        for t in range(10000):
            # Display environment
            # env.render()

            # Get scene from model
            # sc = agent.Model(s)
            sc = s

            # Get action
            a = agent.Act(sc, min(run_through**0.8 / 10, 7))

            # Execute action
            s, r, done, info = env.step(a)

            rewards.append(r)

            # Learn from the reward
            agent.Learn(sc, a, r)

            # Break at end of run-through
            if done:
                print("Run-through {} finished after {} timesteps with reward {}".format(run_through + 1, t + 1,
                                                                                         sum(rewards)))
                break

        # Dump run-through's memories into global memory
        agent.Finish()
