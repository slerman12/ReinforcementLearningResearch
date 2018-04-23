from __future__ import division
import numpy as np
import cv2
import gym
import time
import copy
import sys
from sklearn import random_projection
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from threading import Lock, Thread


# Display progress in console
class Progress:
    # Initialize progress measures
    progress_complete = 0.00
    progress_total = 0.00
    name = ""
    show = True

    def __init__(self, pc, pt, name, show):
        self.progress_complete = pc
        self.progress_total = pt
        self.name = name
        self.show = show
        if self.show:
            sys.stdout.write("\rProgress: {:.2%} [{}]".format(0, name))
            sys.stdout.flush()

    def update_progress(self):
        # Update progress
        self.progress_complete += 1.00
        if self.show:
            sys.stdout.write("\rProgress: {:.2%} [{}]".format(self.progress_complete / self.progress_total, self.name))
            sys.stdout.flush()
        if (self.progress_complete == self.progress_total) and self.show:
            print("")


class Memory:
    length = 0
    knn = {}
    remove = np.array([])

    def __init__(self, memory_size, memory_horizon):
        self.memory_size = memory_size
        self.memory_horizon = memory_horizon

        self.memory = np.zeros((1, self.memory_size))

    def Add(self, scene, action, reward, expected):
        attributes = np.zeros(NUM_ATTRIBUTES)
        attributes[ACTION_INDEX] = action
        attributes[REWARD_INDEX] = reward
        attributes[EXPECTED_INDEX] = expected

        # Insert scene into memory
        self.memory = np.insert(self.memory, 0, np.append(scene, attributes), axis=0)

        # Remove initial zeros
        if self.length == 0:
            self.memory = np.delete(self.memory, -1, 0)

        # Manage memory size
        if self.length >= self.memory_horizon:
            self.memory = np.delete(self.memory, -1, 0)
        else:
            self.length += 1

    def Reset(self):
        self.memory = np.zeros((1, self.memory_size))
        self.length = 1
        self.remove = np.array([])

    # Merge and clear
    def Merge(self, m, reward_discount):
        # Dynamic programming using Bellman equation to computes values (temporally discounted rewards) in linear time
        m.memory[0, VALUE_INDEX] = m.memory[0, REWARD_INDEX]
        for i in range(1, m.length):
            m.memory[i, VALUE_INDEX] = m.memory[i, REWARD_INDEX] + reward_discount * m.memory[i - 1, VALUE_INDEX]
            bellman_LHS = m.memory[i, EXPECTED_INDEX]
            bellman_RHS = m.memory[i, REWARD_INDEX] + reward_discount * m.memory[i - 1, EXPECTED_INDEX]
            m.memory[i, TD_ERROR_INDEX] = bellman_RHS - bellman_LHS

        duplicates = []

        # Begin lock
        gm_lock.acquire()

        # This is very slow -- huge bottleneck (can do some of the work while iterating to compute distances instead!)
        for mem in m.memory:
            try:
                # This in particular is likely the cause
                duplicate = np.argwhere(np.equal(self.memory[:, :ACTION_INDEX + 1], mem[:ACTION_INDEX + 1]).all(1))[0]

                if self.memory[duplicate, VALUE_INDEX] > mem[VALUE_INDEX]:
                    mem[REWARD_INDEX] = self.memory[duplicate, REWARD_INDEX]
                    mem[VALUE_INDEX] = self.memory[duplicate, VALUE_INDEX]

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

        # End lock
        gm_lock.release()

    # Merge and clear
    def Learn(self, k, actions):
        # Custom weight s.t. duplicate state decides ("distance" parameter does that too but weighs inversely otherwise)
        def duplicate_weights(dist):
            for i, point_dist in enumerate(dist):
                if 0. in point_dist:
                    dist[i] = point_dist == 0.
                else:
                    dist[i] = 1.
            return dist

        # This is  slow -- bottleneck
        for action in actions:
            subspace = self.memory[self.memory[:, ACTION_INDEX] == action]
            subspace_size = subspace.shape[0]
            if subspace_size == 0:
                subspace = np.zeros((1, self.memory_size))
                subspace_size = 1
            self.knn[action] = KNeighborsRegressor(n_neighbors=min(k, subspace_size), weights=duplicate_weights,
                                                   n_jobs=1)
            self.knn[action].fit(subspace[:, :-NUM_ATTRIBUTES], subspace[:, VALUE_INDEX])


class Agent:
    def __init__(self, model, global_memory, actions, local_memory_horizon, gamma, epsilon, k):
        self.global_memory = global_memory
        self.actions = actions
        self.local_memory_horizon = local_memory_horizon
        self.reward_discount = gamma
        self.epsilon = epsilon
        self.k = k

        self.model = model
        self.local_memory = Memory(self.model.state_space + NUM_ATTRIBUTES, self.local_memory_horizon)

    def Model(self, state):
        return self.model.Update(state)

    def Act(self, scene):
        # Initialize probabilities
        expected = []

        # Get expected reward for each action
        for action in self.actions:
            exp = self.global_memory.knn[action].predict([scene])[0] if self.global_memory.length > 0 else 0
            expected.append(exp)

        weights = np.array(expected)

        i = np.random.choice(np.arange(2), p=np.array([self.epsilon, 1 - self.epsilon]))
        i = np.random.choice(np.arange(self.actions.size)) if i == 0 else weights.argmax()

        return self.actions[i], expected[i]

    def Learn(self, scene, action, reward, expected):
        self.local_memory.Add(scene, action, reward, expected)

    def Finish_Merge(self):
        self.global_memory.Merge(self.local_memory, self.reward_discount)
        self.local_memory.Reset()

    def Finish_Learn(self):
        self.global_memory.Learn(self.k, self.actions)


# Lower dimension projection of state
class RandomProjection:
    def __init__(self, dimension, size=None, greyscale=False, flatten=False):
        self.dimension = dimension
        self.size = size
        self.greyscale = greyscale
        self.flatten = flatten

        self.projection = None
        self.state_space = self.dimension

    def Update(self, state):
        # Greyscale
        if self.greyscale:
            state = np.dot(state[..., :3], [0.299, 0.587, 0.114])

        # Image resize
        if self.size is not None:
            state = cv2.resize(state, dsize=self.size)

        # Lower dimension projection
        if self.flatten:
            state = state.flatten()

        if self.projection is None:
            self.projection = random_projection.GaussianRandomProjection(self.dimension).fit([state])

        return self.projection.transform([state])[0]


# Felsenszwalb's efficient graph based image segmentation w/ trajectories
class Felsenszwalb:
    class Object:
        index = None
        area = None
        x = None
        y = None
        trajectory_x = 0
        trajectory_y = 0
        previous = None

        def array(self):
            return np.ndarray((1, 5), buffer=np.array([self.area, self.x, self.y, self.trajectory_x, self.trajectory_y]))

        def __eq__(self, another):
            # Might be good to not include 'previous' attribute
            return self.__dict__ == another.__dict__

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

        # A bit slow
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

                self.array[x, 3] = self.objects[x].trajectory_x
                self.array[x, 4] = self.objects[x].trajectory_y

                graph[x, :] = np.inf
                graph[:, y] = np.inf

                if np.isinf(graph).all():
                    break

    def __init__(self, object_capacity):
        self.object_capacity = object_capacity
        self.property_capacity = 5
        self.state = None
        self.segments = None

        if self.object_capacity is not None and self.property_capacity is not None:
            self.state_space = self.object_capacity * self.property_capacity

        self.model = self.Model()

    def Update(self, state, scale=3.0, sigma=0.1, min_size=1):
        self.state = state

        # Greyscale
        self.state = np.dot(state[..., :3], [0.299, 0.587, 0.114])

        # Segmentation
        self.segments = felzenszwalb(state, scale=scale, sigma=sigma, min_size=min_size)

        props = regionprops(self.segments)

        # print(len(props))

        # Can maybe speed up by only updating changed objects
        for obj in range(len(props)):
            o = self.Object()

            o.index = obj
            o.area = round(props[obj].area)
            o.x = round(props[obj].centroid[0])
            o.y = round(props[obj].centroid[1])

            self.model.add(o)

        if self.model.previous is None:
            self.model.previous = self.model

        scene = self.model.scene(self.object_capacity, self.property_capacity)

        # self.model.finish()
        # return scene

        self.model.prev_objects()

        scene = self.model.scene(self.object_capacity, self.property_capacity)

        self.model.finish()

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

    def __eq__(self, another):
        # Might be good to not include 'previous' attribute
        return self.__dict__ == another.__dict__


def run_through_run():
    # Environment
    env = gym.make('CartPole-v0')

    # Visual model
    # occipital = Felsenszwalb(objects)
    occipital = RandomProjection(STATE_SPACE, None, True, True)
    # occipital.state_space = state_space

    # Agent
    agent = Agent(occipital, hippocampus, ACTION_SPACE, local_memory_horizon=1000, gamma=0.999, epsilon=1, k=K)

    # Initialize environment
    s = env.reset()

    for t in range(250):

        # Display environment
        # if run_through > 400:
        #     env.render()

        # start = time.time()

        # Get scene from model
        # sc = agent.Model(s)
        sc = s

        # end = time.time()
        # model_times += end - start

        # start = time.time()

        # Likelihood of picking a random action
        agent.epsilon = max(min(100000 / (run_through + 1) ** 3, 1), 0.001)
        a, e = agent.Act(scene=sc)

        # end = time.time()
        # act_times += end - start

        s, r, done, info = env.step(a)

        # rewards += r

        # start = time.time()

        # Learn from the reward
        agent.Learn(sc, a, r, e)

        # end = time.time()
        # learn_times += end - start

        # Break at end of run-through
        if done:
            break

    # Update temporally discounted rewards and dump run-through's memories into global memory
    agent.Finish_Merge()


# Main method
if __name__ == "__main__":
    NUM_ATTRIBUTES = 5
    ACTION_INDEX = -NUM_ATTRIBUTES
    REWARD_INDEX = -4
    VALUE_INDEX = -3
    EXPECTED_INDEX = -2
    TD_ERROR_INDEX = -1

    NUM_THREADS = 2

    K = 50
    STATE_SPACE = 64

    # Environment
    init_env = gym.make('CartPole-v0')
    ACTION_SPACE = np.arange(init_env.action_space.n)
    STATE_SPACE = init_env.observation_space.shape[0]

    # Environment
    # env = gym.make('Pong-v0')
    # action_space = np.arange(env.action_space.n)
    # objects = 12

    # Environment
    # env = gym.make('SpaceInvaders-v0')
    # action_space = np.arange(env.action_space.n)
    # objects = 160

    # Global memory
    global_memory_horizon = 1000000
    hippocampus = Memory(STATE_SPACE + NUM_ATTRIBUTES, global_memory_horizon)
    gm_lock = Lock()

    epoch = 100

    # Initialize metric variables for measuring performance
    # epoch_rewards = []
    # epoch_model_times = []
    # epoch_act_times = []
    # epoch_learn_times = []
    # epoch_finish_times = []
    epoch_run_through_times = []
    prog = None

    for run_through in range(10000):
        # rewards = 0
        # model_times = 0
        # act_times = 0
        # learn_times = 0

        run_through_start = time.time()

        threads = []
        for i in range(NUM_THREADS):
            threads.append(Thread(target=run_through_run))
            threads[-1].start()

        # start = time.time()

        # Join
        for thread in threads:
            thread.join()

        # Build kNN tree, etc.
        hippocampus.Learn(K, ACTION_SPACE)

        # end = time.time()
        # epoch_finish_times.append(end - start)
        #
        # epoch_rewards.append(rewards)
        # epoch_model_times.append(model_times)
        # epoch_act_times.append(act_times)
        # epoch_learn_times.append(learn_times)

        run_through_end = time.time()
        epoch_run_through_times.append(run_through_end - run_through_start)

        if prog is not None:
            prog.update_progress()

        if not run_through % epoch:
            if run_through > 0:
                # print("Epoch {}, last {} run-through reward average: {}".format(run_through / epoch, epoch, np.mean(epoch_rewards)))
                print("* {} memories stored".format(hippocampus.length))
                # print("* K is {}, r discount {}, exploration {}".format(agent.k, agent.reward_discount, agent.epsilon))
                # print("* Mean modeling time per run-through: {}".format(np.mean(epoch_model_times)))
                # print("* Mean acting time per run-through: {}".format(np.mean(epoch_act_times)))
                # print("* Mean learning time per run-through: {}".format(np.mean(epoch_learn_times)))
                # print("* Mean finishing time per run-through: {}".format(np.mean(epoch_finish_times)))
                print("* Mean run-through time: {}\n".format(np.mean(epoch_run_through_times)))
            # epoch_rewards = []
            # epoch_model_times = []
            # epoch_act_times = []
            # epoch_learn_times = []
            # epoch_finish_times = []
            epoch_run_through_times = []

            # Initiate progress
            prog = Progress(0, epoch, "Epoch", True)
