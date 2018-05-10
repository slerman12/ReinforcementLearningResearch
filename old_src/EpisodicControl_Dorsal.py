from __future__ import division
import numpy as np
import cv2
import gym
import time
import copy
from sklearn import random_projection
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
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
    remove = np.array([])

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

    def Advantage(self, advantage_cutoff=None):
        self.memory[:, ADVANTAGE_INDEX] = self.memory[:, REWARD_INDEX] - self.memory[:, EXPECTED_INDEX]
        if advantage_cutoff is not None:
            self.memory = self.memory[np.abs(self.memory[:, ADVANTAGE_INDEX]) > advantage_cutoff]
            self.length = self.memory.shape[0]

    def Queue_For_Removal(self, index):
        if index not in self.remove:
            self.remove = np.append(self.remove, index)

    def Unique(self):
        _, unique = np.unique(self.memory[:, :ACTION_INDEX + 1], axis=0, return_index=True)
        unique_size = unique.size
        if unique_size < self.length:
            M = self.memory[unique]

            # Need to make this way more efficient by only iterating duplicates
            for mem in self.memory:
                dup = np.argwhere(np.equal(M[:, :ACTION_INDEX + 1], mem[:ACTION_INDEX + 1]).all(1))[0]
                if M[dup, REWARD_INDEX] < mem[REWARD_INDEX]:
                    M[dup, REWARD_INDEX] = mem[REWARD_INDEX]

            self.memory = M
            self.length = unique.size

    def Reset(self):
        self.memory = np.zeros((1, self.memory_size))
        self.length = 1
        self.remove = np.array([])

    # Merge and clear
    def Merge(self, m, k, actions):
        # If duplicate, use max reward
        # for i in m.remove:
        #     i = int(i)
        #     print("BLA")
        #     print(i)
        #     print(np.argwhere(np.equal(m.memory[:, :ACTION_INDEX + 1], self.memory[i, :ACTION_INDEX + 1]).all(1)))
        #
        #     duplicate = np.argwhere(np.equal(m.memory[:, :ACTION_INDEX + 1], self.memory[i, :ACTION_INDEX + 1]).all(1))[0]
        #
        #     if self.memory[i, REWARD_INDEX] > m.memory[duplicate, REWARD_INDEX]:
        #         m.memory[duplicate, REWARD_INDEX] = self.memory[i, REWARD_INDEX]
        duplicates = []

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
            self.memory = self.memory[:-(self.length - self.memory_horizon), :]
            self.length = self.memory_horizon

        # Custom weight s.t. duplicate state decides ("distance" parameter does that too but weighs inversely otherwise)
        def duplicate_weights(dist):
            for i, point_dist in enumerate(dist):
                if 0. in point_dist:
                    dist[i] = point_dist == 0.
                else:
                    dist[i] = 1.
            return dist

        # def advantage_distance(x, y):
        #     return np.linalg.norm(x[:-NUM_ATTRIBUTES] - y[:-NUM_ATTRIBUTES])

        for action in actions:
            # self.knn[action] = NearestNeighbors(n_neighbors=k)
            # self.knn[action].fit(self.memory[self.memory[:, -2] == action, :-2])
            subspace = self.memory[self.memory[:, ACTION_INDEX] == action]
            subspace_size = subspace.shape[0]
            if subspace_size == 0:
                subspace = np.zeros((1, self.memory_size))
                subspace_size = 1
            self.knn[action] = KNeighborsRegressor(n_neighbors=min(k, subspace_size), weights=duplicate_weights)
            self.knn[action].fit(subspace[:, :-NUM_ATTRIBUTES], subspace[:, REWARD_INDEX])


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

    def Model(self, state, r):
        return self.model.Update(state, r)

    def Act(self, scene, decisiveness, important=None, relation=None):
        # Initialize probabilities
        expected = []
        # advantage = []

        # # Duplicate
        # prev_actions = []
        # indices = []

        # Custom weight s.t. duplicate state decides ("distance" parameter does that too but weighs inversely otherwise)
        def duplicate_weights(dist):
            for i, point_dist in enumerate(dist):
                if 0. in point_dist:
                    dist[i] = point_dist == 0.
                else:
                    dist[i] = 1.
            return dist

        def relate(x, y, imp, rel):
            objects_x = x.reshape((-1, 5))
            objects_y = y.reshape((-1, 5))

            x_neighbors = NearestNeighbors(n_neighbors=1)
            x_neighbors.fit(objects_x)

            x_imp_dist, x_imp_ind = x_neighbors.kneighbors(imp)
            x_imp_ind = x_imp_ind[0][0]

            x_imp = objects_x[x_imp_ind]

            x_rel_dist, x_rel_ind = x_neighbors.kneighbors(rel) # assret does not equal x_imp_ind
            x_rel_ind = x_rel_ind[0][0]

            x_rel = objects_x[x_rel_ind]

            y_neighbors = NearestNeighbors(n_neighbors=1)
            y_neighbors.fit(objects_y)

            y_imp_dist, y_imp_ind = y_neighbors.kneighbors(imp)
            y_imp_ind = y_imp_ind[0][0]

            y_imp = objects_y[y_imp_ind]

            y_rel_dist, y_rel_ind = y_neighbors.kneighbors(rel)
            y_rel_ind = y_rel_ind[0][0]

            y_rel = objects_y[y_rel_ind]

            return np.linalg.norm(np.array([x_imp, x_rel]) - np.array([y_imp, y_rel])) * (x_imp_dist + x_rel_dist +
                                                                                          y_imp_dist + y_rel_dist)

        # Get expected reward for each action
        for action in self.actions:
            # weights.append(expected_reward(scene, action, self.global_memory, self.k))
            if important is None:
                exp = self.global_memory.knn[action].predict([scene])[0] if self.global_memory.length > 0 else 0
                expected.append(exp)
            else:
                subspace = self.global_memory.memory[self.global_memory.memory[:, ACTION_INDEX] == action]
                subspace_size = subspace.shape[0]
                if subspace_size == 0:
                    subspace = np.zeros((1, self.global_memory.memory_size))
                    subspace_size = 1

                relatenn = {action: KNeighborsRegressor(n_neighbors=min(self.k, subspace_size), weights=duplicate_weights,
                                                        metric=relate, metric_params={"imp": important, "rel": relation})}

                relatenn[action].fit(subspace[:, :-NUM_ATTRIBUTES], subspace[:, REWARD_INDEX])

                exp = relatenn[action].predict([scene])[0] if self.global_memory.length > 0 else 0
                expected.append(exp)
            # advantage.append(adv)


            # exp = 0
            # advantage_count = 0
            #
            # subspace = self.global_memory.memory[self.global_memory.memory[:, ACTION_INDEX] == action]
            #
            # if self.global_memory.length > 0 and subspace.size > 0:
            #     dist, ind = self.global_memory.knn[action].kneighbors([scene])
            #     dist = dist[0]
            #     ind = ind[0]
            #
            #     if 0 in dist:
            #         index = ind[np.argwhere(dist == 0)][0][0]
            #         indices.append(index)
            #         prev_actions.append(subspace[index, ACTION_INDEX])
            #         exp = subspace[index, REWARD_INDEX]
            #     else:
            #         dist = 1 / dist
            #         dist = dist / np.sum(dist)
            #
            #         for ix, i in enumerate(ind):
            #             adv = np.abs(subspace[i, ADVANTAGE_INDEX])
            #             advantage_count += adv
            #             exp += subspace[i, REWARD_INDEX]
            #         exp = exp / self.k
            #
            # expected.append(exp)

        weights = np.array(expected)

        # Shift probabilities such that they are positive
        if weights.min() <= 0:
            weights -= weights.min() - 1

        # Increase likelihood of better actions
        weights = weights ** decisiveness

        # Scale probabilities such that they are between [0, 1]
        weights = np.divide(weights, np.sum(weights))

        # act = np.where(weights == weights.max())[0][0]
        # weights = [1 - epsilon if i == act else epsilon / (self.actions.size - 1) for i, w in np.ndenumerate(weights)]

        # Choose an action probabilistically
        i = np.random.choice(np.arange(self.actions.size), p=weights)

        # The reason this doesn't work is because index indexes an action subspace/buffer -- use memory space per action
        # for ii in range(len(prev_actions)):
        #     if prev_actions[ii] == self.actions[i]:
        #         self.local_memory.Queue_For_Removal(indices[ii])

        return self.actions[i], expected[i]

    def Learn(self, scene, action, reward, expected):
        self.local_memory.Update(scene, action, reward, self.reward_discount, expected)

    def Finish(self):
        self.local_memory.Advantage()
        # self.local_memory.Unique()
        self.global_memory.Merge(self.local_memory, self.k, self.actions)
        self.local_memory.Reset()
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

        self.projection = random_projection.GaussianRandomProjection(dimension).fit(
            np.zeros((self.memory_capacity, state_size)))
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
    important_object = None
    relation = None

    class Object:
        index = None
        area = None
        x = None
        y = None
        trajectory_x = 0
        trajectory_y = 0
        importance = 0
        previous = None

        def array(self):
            return np.ndarray((1, 5), buffer=np.array([self.area, self.x, self.y, self.trajectory_x, self.trajectory_y]))

    class Model:
        objects = []
        length = 0
        array = None
        previous = None

        def add(self, o):
            self.objects.append(o)
            assert self.objects == sorted(self.objects, key=lambda x: x.index)
            self.length += 1

            if self.array is None:
                self.array = np.ndarray((1, 5), buffer=np.array([o.area, o.x, o.y, o.trajectory_x, o.trajectory_y]))
            else:
                self.array = np.append(self.array, np.array([[o.area, o.x, o.y, o.trajectory_x, o.trajectory_y]]),
                                       axis=0)

        def forget(self):
            self.previous = None
            for o in self.objects:
                o.previous = None

        def finish(self):
            self.forget()
            self.previous = copy.deepcopy(self)
            self.objects = []
            self.length = 0
            self.array = None

        def scene(self, object_capacity, property_capacity):
            scene = np.zeros((object_capacity, property_capacity))

            for index in range(min(object_capacity, self.length)):
                scene[index, 0] = self.objects[index].area
                scene[index, 1] = self.objects[index].x
                scene[index, 2] = self.objects[index].y
                scene[index, 3] = self.objects[index].trajectory_x
                scene[index, 4] = self.objects[index].trajectory_y

            return scene.flatten()

        def prev_objects(self):
            likely_pairs = NearestNeighbors(n_neighbors=1)
            likely_pairs.fit(self.previous.array[:, :3])

            graph = likely_pairs.kneighbors_graph(self.array[:, :3], mode="distance").toarray()

            for _ in range(max(self.length, self.previous.length)):

                x, y = np.unravel_index(np.argmin(graph, axis=None), graph.shape)

                self.objects[x].previous = self.previous.objects[y]

                self.objects[x].trajectory_x = self.objects[x].x - self.objects[x].previous.x
                self.objects[x].trajectory_y = self.objects[x].y - self.objects[x].previous.y

                self.objects[x].importance = self.objects[x].previous.importance

                self.array[x, 3] = self.objects[x].trajectory_x
                self.array[x, 4] = self.objects[x].trajectory_y

                graph[x, :] = np.inf
                graph[:, y] = np.inf

                if np.isinf(graph).all():
                    break

                    # objects_gone = np.setdiff1d(np.arange(self.previous.array.shape[0]), self.pair_indices[:, 1])
                    # new_objects = np.setdiff1d(np.arange(self.array.shape[0]), self.pair_indices[:, 0])

    # Group relations based on importance points given for action/reward correlations with trajectory changes,
    # and physical proximity

    def __init__(self, object_capacity):
        self.object_capacity = object_capacity
        self.property_capacity = 5
        self.state = None
        self.segments = None

        self.time_of_consideration = 0

        if self.object_capacity is not None and self.property_capacity is not None:
            self.state_space = self.object_capacity * self.property_capacity

        self.model = self.Model()

    def Update(self, state, reward, scale=3.0, sigma=0.1, min_size=1):
        self.state = state

        # Greyscale
        self.state = np.dot(state[..., :3], [0.299, 0.587, 0.114])

        # Segmentation
        self.segments = felzenszwalb(state, scale=scale, sigma=sigma, min_size=min_size)

        props = regionprops(self.segments)

        # Can maybe speed up by only updating changed objects
        for obj in range(len(props)):
            o = self.Object()

            o.index = obj
            o.area = props[obj].area
            o.x = props[obj].centroid[0]
            o.y = props[obj].centroid[1]

            self.model.add(o)

        if self.model.previous is None:
            self.model.previous = self.model

        self.model.prev_objects()

        scene = self.model.scene(self.object_capacity, self.property_capacity)

        # if self.important_object is not None and self.relation is not None and reward > 0:

        if reward > 0:
            self.time_of_consideration = 10

        if self.time_of_consideration > 0:
            trajectory_changes = [o for o in self.model.objects
                                  if np.sign(o.trajectory_x) != np.sign(o.previous.trajectory_x)
                                  or np.sign(o.trajectory_y) != np.sign(o.previous.trajectory_y)]

            trajectory_changes.sort(key=lambda x: np.abs(x.trajectory_x - x.previous.trajectory_x) +
                                    np.abs(x.trajectory_y - x.previous.trajectory_y), reverse=True)

            for o in trajectory_changes[:min(5, len(trajectory_changes))]:
                o.importance += 1

            self.time_of_consideration = max(0, self.time_of_consideration - 1)

        self.important_object = max(self.model.objects, key=lambda x: x.importance)
        other_objects = [i for i in self.model.objects if i.index != self.important_object.index]

        self.relation = min(other_objects, key=lambda x: np.linalg.norm(
            np.array([self.important_object.x, self.important_object.y]) - np.array([x.x, x.y])))

        use_relation = False

        if self.relation.importance > 0:
            use_relation = True

        self.model.finish()

        return scene, use_relation, self.important_object.get_array(), self.relation.get_array()

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
    # env = gym.make('CartPole-v0')
    # action_space = np.arange(env.action_space.n)
    # objects = None
    # properties = None
    # state_space = env.observation_space.shape[0]

    # Environment
    env = gym.make('Breakout-v0')
    action_space = np.arange(env.action_space.n)
    objects = 22

    # Environment
    # env = gym.make('SpaceInvaders-v0')
    # action_space = np.arange(env.action_space.n)
    # objects = 160

    # Visual model
    occipital = Felsenszwalb(objects)
    # occipital.state_space = state_space

    # Global memory
    memory_limit = 20
    hippocampus = Memory(occipital.state_space + NUM_ATTRIBUTES, memory_limit)

    # Agent
    agent = Agent(occipital, hippocampus, action_space, 20, 20, reward_discount=0.999, k=7)

    epoch = 1
    epoch_rewards = []
    times = []

    sc = [0, 0]
    r = 0

    time_of_consid = 0

    for run_through in range(10000):
        # Initialize environment
        s = env.reset()

        rewards = []

        for t in range(1000):

            # Display environment
            # if run_through > 6:
            # env.render()

            if time_of_consid == 0:
                start = time.time()

                # Get scene from model
                sc = agent.Model(s, r)
                # sc = np.array([round(elem, 1) for elem in s])
                # sc = s

                end = time.time()
                times.append(end - start)
                # print("Model time: {}".format(end - start))

            if sc[1] or time_of_consid > 0:
                a, e = agent.Act(sc[0], min(run_through * 10, 100), important=sc[2], relation=sc[3])

                if np.abs(r - e) < 5 or r > 0:
                    time_of_consid += 5
                else:
                    time_of_consid = max(0, time_of_consid - 1)
            else:
                # start = time.time()

                # Get action and expected reward
                # a, e = (a, e) if t % 4 else agent.Act(sc, min(run_through * 20, 100))
                a, e = agent.Act(sc[0], min(run_through * 10, 100))

                # end = time.time()
                # print("Action time: {}".format(end - start))

            # Execute action
            s, r, done, info = env.step(a)

            rewards.append(r)

            # start = time.time()

            # Learn from the reward
            if r > 0:
                agent.Learn(sc[0], a, r, e)

            # end = time.time()
            # print("Learning time: {}".format(end - start))

            # Break at end of run-through
            if done:
                # print("Run-through {} finished after {} timesteps with reward {}".format(run_through + 1, t + 1,
                #                                                                          sum(rewards)))
                break

        epoch_rewards.append(sum(rewards))
        if not run_through % epoch:
            print("Last {} run-through reward average: {}".format(epoch, np.mean(epoch_rewards)))
            print("* {} memories stored".format(agent.global_memory.length))
            print("* Average modeling time: {}".format(np.mean(times)))
            epoch_rewards = []
            times = []

        # Dump run-through's memories into global memory
        agent.Finish()
