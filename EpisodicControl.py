from __future__ import division
import numpy as np
import cv2

def euclidean(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


def expected_reward(scene, action, memory, k):
    # Use subspace of emmories that took this action
    subspace = memory.memory[memory.memory[:, -2] == action]

    # Initialize probabilities
    weights = []

    # Get similarity of each memory to the scene
    for mem in subspace:
        similarity = np.math.exp(-euclidean(scene, mem[:-3]))
        weights.append(similarity)

    # Scale probabilities such that they are between [0, 1]
    weights /= sum(weights)

    # Compute expectation
    expected = 0
    for i in range(len(weights)):
        expected += weights[i] * subspace[i, -1]

    # Return expected reward
    return expected


class Memory:
    memory = np.array([])
    length = 0

    def __init__(self, memory_horizon, reward_horizon, discount):
        self.memory_horizon = memory_horizon
        self.reward_horizon = reward_horizon
        self.discount = discount

    def Update(self, scene, action, reward):
        # Insert scene into memory
        self.memory = np.insert(self.memory, 0, np.append(scene, [action, reward]), axis=0)

        # Manage memory size
        if self.length >= self.memory_horizon:
            np.delete(self.memory, -1, 0)
        else:
            self.length += 1

        # Add discounted rewards
        for i in range(1, self.reward_horizon):
            self.memory[i, -1] += reward * (1 - (i / self.reward_horizon)**self.discount)

    # Merge and clear
    def Merge(self, m):
        np.concatenate((m, self.memory), axis=0)
        return np.array([])


class Agent:
    def __init__(self, global_memory, actions, reward_horizon, discount, memory_horizon, k):
        self.global_memory = global_memory
        self.actions = actions
        self.reward_horizon = reward_horizon
        self.discount = discount
        self.memory_horizon = memory_horizon
        self.k = k

        self.local_memory = Memory(self.memory_horizon, self.memory_horizon, self.discount)
        self.model = Projection(33, (100, 100))

    def Model(self, state):
        return self.model.Update(state)

    def Act(self, scene):
        # Initialize probabilities
        weights = []

        # Get expected reward for each action
        for action in self.actions:
            weights.append(expected_reward(scene, action, self.global_memory, self.k))

        # Shift probabilities such that they are positive
        if min(weights) <= 0:
            weights += min(weights) + 1

        # Scale probabilities such that they are between [0, 1]
        weights /= sum(weights)

        # Return an action probabilistically
        return np.random.choice(self.actions, p=weights)

    def Learn(self, scene, action, reward):
        self.local_memory.Update(scene, action, reward)

    def Finish(self):
        self.local_memory = self.global_memory.Merge(self.local_memory)
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
    print()
