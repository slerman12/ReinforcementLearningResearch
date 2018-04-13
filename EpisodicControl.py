from __future__ import division
import numpy as np
import cv2
import gym
from sklearn import random_projection
from sklearn.neighbors import KNeighborsRegressor
from skimage.measure import regionprops
from skimage.segmentation import slic, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt


# def euclidean(x, y):
#     return np.linalg.norm(x - y)
#
#
# def expected_reward(scene, action, memory, k):
#     # Check if memory empty
#     if memory.length == 0:
#         return 0
#
#     # Use subspace of memories that took this action
#     subspace = memory.memory[memory.memory[:, -2] == action]
#
#     # Check if subspace empty
#     if subspace.size == 0:
#         return 0
#
#     # Initialize probabilities
#     reward_outcomes = []
#     weights = []
#
#     # Similarity of each memory to the scene
#     for mem in subspace:
#         # similarity = np.math.exp(-777*euclidean(scene, mem[:-2]))
#         # similarity = 1 - (euclidean(scene, mem[:-2]) / 1000) ** 2
#         distance = euclidean(scene, mem[:-2])
#
#         reward_outcomes.append(mem[-1])
#         weights.append(distance)
#
#     weights = np.array(weights)
#
#     nearest_neighbors = weights.argsort()[:k]
#
#     nearest_neighbors = memory.knn[action].kneighbors([scene], return_distance=False)
#
#     expected = 0
#
#     # for i in nearest_neighbors:
#     #     expected += reward_outcomes[i]
#
#     for nn in nearest_neighbors[0]:
#         expected += memory.memory[nn, -1]
#
#     return expected / k
#
#     max_dist = weights.max()
#
#     # Convert distances to similarities
#     # weights = 1 - (weights / (max_dist + 1)) ** similarity_discount
#     weights = similarity_discount ** weights
#
#     # Scale similarities such that they can be used as a probability distribution i.e. between [0, 1] and sum to 1
#     weights = np.divide(weights, np.sum(weights))
#
#     # Compute expectation
#     expected = 0
#     for i in range(len(weights)):
#         expected += weights[i] * reward_outcomes[i]
#
#     # Return expected reward
#     return expected


class Memory:
    length = 0
    knn = {}

    def __init__(self, memory_size, memory_horizon, reward_horizon=1):
        self.memory_size = memory_size
        self.memory_horizon = memory_horizon
        self.reward_horizon = reward_horizon

        self.memory = np.zeros((1, self.memory_size))

    def Update(self, scene, action, reward, reward_discount, expected):
        # Insert scene into memory
        self.memory = np.insert(self.memory, 0, np.append(scene, [action, reward, expected, 0]), axis=0)

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
            self.memory[i, REWARD_INDEX] += reward * (reward_discount ** i)

    def Advantage(self):
        self.memory[:, ADVANTAGE_INDEX] = np.absolute(self.memory[:, REWARD_INDEX] - self.memory[:, EXPECTED_INDEX])

    # Merge and clear
    def Merge(self, m, k, actions):
        duplicates = []
        # If duplicate, use max reward
        for mem in m.memory:
            try:
                duplicate = np.argwhere(np.equal(self.memory[:, :ACTION_INDEX + 1], mem[:ACTION_INDEX + 1]).all(1))[0]

                if self.memory[duplicate, REWARD_INDEX] > mem[REWARD_INDEX]:
                    mem[REWARD_INDEX] = self.memory[duplicate, REWARD_INDEX]

                duplicates.append(duplicate)
            except IndexError:
                pass

        if len(duplicates) > 0:
            self.memory = np.delete(self.memory, duplicates, axis=0)

        # Merge memories
        if self.length == 0:
            self.memory = m.memory
        else:
            self.memory = np.concatenate((m.memory, self.memory), axis=0)

        # Update memory length
        self.length += m.length - len(duplicates)

        # Delete memories beyond horizon
        if self.length > self.memory_horizon:
            self.memory = self.memory[:-(self.length - self.memory_horizon)]
            self.length = self.memory_horizon

        # Custom weight s.t. duplicate state decides ("distance" parameter does that too but weighs inversely otherwise)
        def duplicate_weights(dist):
            for i, point_dist in enumerate(dist):
                if 0. in point_dist:
                    dist[i] = point_dist == 0.
                else:
                    dist[i] = 1.
            return dist

        for action in actions:
            # self.knn[action] = NearestNeighbors(n_neighbors=k)
            # self.knn[action].fit(self.memory[self.memory[:, -2] == action, :-2])
            subspace = self.memory[self.memory[:, ACTION_INDEX] == action]
            subspace_size = subspace.shape[0]
            if subspace_size == 0:
                subspace = np.zeros((1, self.memory_size))
            self.knn[action] = KNeighborsRegressor(n_neighbors=min(k, subspace_size), weights=duplicate_weights)
            self.knn[action].fit(subspace[:, :-NUM_ATTRIBUTES], subspace[:, [REWARD_INDEX, ADVANTAGE_INDEX]])

        return np.zeros((1, self.memory_size))


class Agent:
    def __init__(self, model, global_memory, actions, memory_horizon, reward_horizon, reward_discount, k):
        self.global_memory = global_memory
        self.actions = actions
        self.memory_horizon = memory_horizon
        self.reward_horizon = reward_horizon
        self.reward_discount = reward_discount
        self.k = k

        self.model = model
        self.local_memory = Memory(self.model.state_space + NUM_ATTRIBUTES, self.memory_horizon, self.reward_horizon)

    def Model(self, state):
        return self.model.Update(state)

    def Act(self, scene, decisiveness):
        # Initialize probabilities
        expected = []
        advantage = []

        # Get expected reward for each action
        for action in self.actions:
            # weights.append(expected_reward(scene, action, self.global_memory, self.k))
            exp, adv = self.global_memory.knn[action].predict([scene])[0] if self.global_memory.length > 0 else (0, 0)
            expected.append(exp)
            advantage.append(adv)

        weights = np.array(expected)

        # Shift probabilities such that they are positive
        if weights.min() <= 0:
            weights += weights.min() + 1

        # Increase likelihood of better actions
        weights = weights ** decisiveness

        # Scale probabilities such that they are between [0, 1]
        weights = np.divide(weights, np.sum(weights))

        # Choose an action probabilistically
        i = np.random.choice(np.arange(self.actions.size), p=weights)

        return self.actions[i], expected[i]

    def Learn(self, scene, action, reward, expected):
        self.local_memory.Update(scene, action, reward, self.reward_discount, expected)

    def Finish(self):
        self.local_memory.Advantage()
        self.local_memory.memory = self.global_memory.Merge(self.local_memory, self.k, self.actions)
        self.local_memory.length = 0
        # self.model.Finish()


# Lower dimension projection of state
class Projection:
    def __init__(self, dimension, size=None):
        self.dimension = dimension
        self.size = size
        self.projection = None

    def Update(self, state):
        # Greyscale
        state = np.dot(state[..., :3], [0.299, 0.587, 0.114])

        # Image resize
        if self.size is not None:
            state = cv2.resize(state, dsize=self.size)

        # Lower dimension projection
        state = state.flatten()

        if self.projection is None:
            self.projection = np.random.RandomState().randn(self.dimension, len(state)).astype(np.float32)

        return np.dot(self.projection, state)


# Lower dimension projection of state
class RandomProjection:
    def __init__(self, memory_capacity, state_size, dimension, size=None):
        self.memory_capacity = memory_capacity
        self.dimension = dimension
        self.size = size

        self.projection = random_projection.GaussianRandomProjection(dimension).fit(np.zeros((self.memory_capacity, state_size)))
        self.state_space = self.dimension

    def Update(self, state):
        # Greyscale
        state = np.dot(state[..., :3], [0.299, 0.587, 0.114])

        # Image resize
        if self.size is not None:
            state = cv2.resize(state, dsize=self.size)

        # Lower dimension projection
        state = state.flatten()

        return self.projection.transform(state)


# SLIC image segmentation
class SLIC:
    state = None
    segments = None

    def __init__(self, object_capacity):
        self.object_capacity = object_capacity

    def Update(self, state, num_segments=3, sigma=0, compactness=0.01, max_size_factor=1000, min_size_factor=0.0001,
               convert2lab=True):
        # Force image to float
        self.state = img_as_float(state)

        # Segmentation
        self.segments = slic(state, n_segments=num_segments, sigma=sigma, compactness=compactness,
                             convert2lab=convert2lab, min_size_factor=min_size_factor, max_size_factor=max_size_factor)

        props = regionprops(self.segments)

        num_objects = len(props)

        scene = np.zeros(self.object_capacity * 3)

        for obj in range(min(self.object_capacity, num_objects)):
            scene[obj * 3] = props[obj].area
            scene[obj * 3 + 1] = props[obj].centroid[0]
            scene[obj * 3 + 1] = props[obj].centroid[1]

        return scene

    def Show(self):
        if self.state is not None and self.segments is not None:
            # Show segments
            figure = plt.figure("Segments")
            ax = figure.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(self.state, self.segments))
            plt.axis("off")

            # Plot
            plt.show()


# Felsenszwalb’s efficient graph based image segmentation
class Felsenszwalb:
    state = None
    segments = None

    def __init__(self, object_capacity, property_capacity):
        self.object_capacity = object_capacity
        self.property_capacity = property_capacity
        if self.object_capacity is not None and self.property_capacity is not None:
            self.state_space = self.object_capacity * self.property_capacity

    def Update(self, state, scale=3.0, sigma=0.1, min_size=1):
        # Segmentation
        self.segments = felzenszwalb(state, scale=scale, sigma=sigma, min_size=min_size)

        props = regionprops(self.segments)

        num_objects = len(props)

        scene = np.zeros(self.object_capacity * 3)

        for obj in range(min(self.object_capacity, num_objects)):
            scene[obj * 3] = props[obj].area
            scene[obj * 3 + 1] = props[obj].centroid[0]
            scene[obj * 3 + 2] = props[obj].centroid[1]

        return scene

    def Show(self):
        if self.state is not None and self.segments is not None:
            # Show segments
            figure = plt.figure("Segments")
            ax = figure.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(self.state, self.segments))
            plt.axis("off")

            # Plot
            plt.show()


# Main method
if __name__ == "__main__":
    NUM_ATTRIBUTES = 4
    ACTION_INDEX = -4
    REWARD_INDEX = -3
    EXPECTED_INDEX = -2
    ADVANTAGE_INDEX = -1

    # Environment
    env = gym.make('CartPole-v0')
    action_space = np.arange(env.action_space.n)
    objects = None
    properties = None
    state_space = env.observation_space.shape[0]

    # # Environment
    # env = gym.make('Breakout-v0')
    # action_space = np.arange(env.action_space.n)
    # objects = 25
    # properties = 3
    # state_space = env.observation_space.shape[0] * env.observation_space.shape[1]

    # Visual model
    occipital = Felsenszwalb(objects, properties)
    occipital.state_space = state_space

    # Global memory
    memory_limit = 100000
    hippocampus = Memory(occipital.state_space + NUM_ATTRIBUTES, memory_limit)

    # Agent
    agent = Agent(occipital, hippocampus, action_space, 1000, 1000, reward_discount=1, k=11)

    for run_through in range(100000):
        # Initialize environment
        s = env.reset()

        rewards = []

        for t in range(10000):
            # Display environment
            # env.render()

            # Get scene from model
            # sc = agent.Model(s)
            # sc = np.array([round(elem, 10) for elem in s])
            sc = s

            # Get action and expected reward
            a, e = agent.Act(sc, min(run_through ** 0.8 / 10, 7))

            # Execute action
            s, r, done, info = env.step(a)

            rewards.append(r)

            # Learn from the reward
            agent.Learn(sc, a, r, e)

            # Break at end of run-through
            if done:
                print("Run-through {} finished after {} timesteps with reward {}".format(run_through + 1, t + 1,
                                                                                         sum(rewards)))
                break

        # Dump run-through's memories into global memory
        agent.Finish()
