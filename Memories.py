from __future__ import division
import math
import numpy as np
from pyflann.index import FLANN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors.kd_tree import KDTree


class Traces:
    def __init__(self, capacity, attributes, memories, gamma):
        # Initialize dict of traces
        self.traces = {}

        # Arrays of memory traces for calculating values from rewards
        for attribute, dimensionality in attributes.items():
            if dimensionality > 1:
                self.traces[attribute] = np.zeros((capacity, dimensionality))
            else:
                self.traces[attribute] = np.zeros(capacity)

        # Indices of attributes (such as reward, value, etc.)
        self.attributes = attributes

        # Max number of memories
        self.capacity = capacity

        # Current number of memory traces
        self.length = 0

        # Memories to store into
        self.memories = memories

        # Reward discount
        self.gamma = gamma

    def add(self, trace):
        # Reward of trace
        reward = trace["reward"]

        # Initialize trace's value to reward
        trace["value"] = reward

        # Update values of existing traces except oldest
        # for i in range(1, self.length):
        for i in range(0, self.length):
            self.traces["value"][i] += self.gamma ** (self.length - i) * reward

        # If memory capacity has not been reached
        if self.length < self.capacity:
            # Update oldest trace
            # self.traces[0, value_index] += self.gamma ** self.length * reward

            # Add trace
            for attribute, dimensionality in self.attributes.items():
                self.traces[attribute][self.length] = trace[attribute]

            # Update length
            self.length += 1
        else:
            # Oldest trace
            memory = self.get_trace_by_index(0)

            # Expected value index
            # expected_index = self.attributes["expected"]

            # Update memory with off-policy prediction
            # memory[value_index] = self.gamma ** self.length * trace[expected_index]

            # Add memory to long term memory
            # TODO: Since I'm not updating the KD tree at every step, it is possible for a duplicate to enter memory
            self.memories[int(memory["action"])].store(memory)

            # Pop memory from traces and add
            for attribute, dimensionality in self.attributes.items():
                # Pop trace
                self.traces[attribute][:-1] = self.traces[attribute][1:]

                # Add trace
                self.traces[attribute][self.length - 1] = trace[attribute]

        # If terminal state
        if trace["terminal"]:
            # Dump all traces into memory
            for i in range(self.length):
                trace = self.get_trace_by_index(i)
                self.memories[int(trace["action"])].store(trace)

            # Reset traces
            self.reset()

    def get_trace_by_index(self, index):
        # Initialize memory
        trace = {}

        # Set memory
        for attribute, dimensionality in self.attributes.items():
            trace[attribute] = self.traces[attribute][index]

        # Return memory
        return trace

    def reset(self):
        # Reset internal states
        self.length = 0
        for attribute, dimensionality in self.attributes.items():
            if dimensionality > 1:
                self.traces[attribute] = np.zeros((self.capacity, dimensionality))
            else:
                self.traces[attribute] = np.zeros(self.capacity)


class Memories:
    def __init__(self, capacity, attributes):
        # Initialize memories
        self.memories = {}

        # Arrays of memory
        for attribute, dimensionality in attributes.items():
            if dimensionality > 1:
                self.memories[attribute] = np.zeros((capacity, dimensionality))
            else:
                self.memories[attribute] = np.zeros(capacity)

        # Memory attributes (such as state, reward, value, etc.) and their dimensionality
        self.attributes = attributes

        # Max number of memories
        self.capacity = capacity

        # Current number of memories
        self.length = 0

        # Consolidated memories
        self.tree = None

        # Number of duplicates
        self.num_duplicates = 0

    def store(self, memory, check_duplicate=False):
        # If memory capacity has not been reached
        if self.length < self.capacity:
            # Add memory
            for attribute, dimensionality in self.attributes.items():
                self.memories[attribute][self.length] = memory[attribute]

            # Update length
            self.length += 1
        else:
            # Replace least recently accessed memory with new memory
            least_recently_accessed = np.argmin(self.memories["time_accessed"][:self.length])
            for attribute, dimensionality in self.attributes.items():
                self.memories[attribute][least_recently_accessed] = memory[attribute]

    def retrieve(self, experience, k):
        # Retrieve k most similar memories #
        dist, ind = self.tree.query([experience], k=min(k, self.length))
        # ind, dist = self.tree.nn_index(experience.reshape(1, experience.shape[0]), min(k, self.length))

        # Return memories
        return dist[0], ind[0]

    def consolidate(self, short_term_memories):
        # If there are short term memories available
        if short_term_memories.length > 0:
            # Store short term memories
            for memory in range(short_term_memories.length):
                self.store(short_term_memories.get_memory_by_index(memory), True)

            # Empty out short term memories
            short_term_memories.reset()

            # Build tree of long term memories
            self.tree = KDTree(self.memories["state"][:self.length], leaf_size=400)
            # self.tree = KDTree(self.memories[:self.length, :-num_attributes], leaf_size=math.ceil(self.length / 250))
            # self.tree = FLANN()
            # self.tree.build_index(self.memories[:self.length, :-num_attributes])
            # self.tree = KNeighborsRegressor(n_neighbors=min(50, self.length))
            # self.tree.fit(self.memories[:self.length, :-num_attributes],
            #               self.memories[:self.length, self.attributes["value"]])

    def get_memory_by_index(self, index):
        # Initialize memory
        memory = {}

        # Set memory
        for attribute, dimensionality in self.attributes.items():
            memory[attribute] = self.memories[attribute][index]

        # Return memory
        return memory

    def reset(self):
        # Reset internal states
        self.tree = None
        self.length = 0
        for attribute, dimensionality in self.attributes.items():
            if dimensionality > 1:
                self.memories[attribute] = np.zeros((self.capacity, dimensionality))
            else:
                self.memories[attribute] = np.zeros(self.capacity)


class MFEC(Memories):
    def store(self, memory, check_duplicate=False):
        # Duplicate index (positive if exists)
        duplicate_index = int(memory["duplicate"]) if check_duplicate else -1

        # If not a duplicate
        if duplicate_index < 0:
            # If memory capacity has not been reached
            if self.length < self.capacity:
                # Add memory
                for attribute, dimensionality in self.attributes.items():
                    self.memories[attribute][self.length] = memory[attribute]

                # Update length
                self.length += 1
            else:
                # Replace least recently accessed memory with new memory
                least_recently_accessed = np.argmin(self.memories["time_accessed"][:self.length])
                for attribute, dimensionality in self.attributes.items():
                    self.memories[attribute][least_recently_accessed] = memory[attribute]
        else:
            # Increment duplicates counter
            self.num_duplicates += 1

            # Reconcile duplicate by using one with max value
            if self.memories["value"][duplicate_index] < memory["value"]:
                for attribute, dimensionality in self.attributes.items():
                    self.memories[attribute][duplicate_index] = memory[attribute]
