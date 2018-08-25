from __future__ import division
import math
import numpy as np
# from pyflann.index import FLANN
from copy import deepcopy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors.kd_tree import KDTree
import Brains
import tensorflow as tf


class Traces:
    def __init__(self, capacity, attributes, memories, reward_discount):
        # Initialize dict of traces
        self.traces = {}

        # Arrays of memory traces for calculating values from rewards TODO: add support for >2D memory
        for attribute, dimensionality in attributes.items():
            if dimensionality > 1:
                self.traces[attribute] = np.zeros((capacity, dimensionality))
            else:
                self.traces[attribute] = np.zeros(capacity)

        # Indices of attributes (such as scene, reward, value, etc.)
        self.attributes = attributes

        # Max number of memories
        self.capacity = capacity

        # Current number of memory traces
        self.length = 0

        # Memories to store into
        self.memories = memories

        # Reward discount
        self.reward_discount = reward_discount

    def add(self, trace):
        # Reward of trace
        reward = trace["reward"]

        # Initialize trace's value to reward
        trace["value"] = reward

        # Update values of existing traces except oldest
        # for i in range(1, self.length):
        for i in range(0, self.length):
            self.traces["value"][i] += self.reward_discount ** (self.length - i) * reward

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
    def __init__(self, capacity, attributes, tensorflow=True):
        # Initialize memories
        self.memories = {}

        # Arrays of memory
        for attribute, dimensionality in attributes.items():
            if dimensionality > 1:
                self.memories[attribute] = np.zeros((capacity, dimensionality))
            else:
                self.memories[attribute] = np.zeros(capacity)

        # Memory attributes (such as scene, state, reward, value, etc.) and their dimensionality
        self.attributes = attributes

        # Max number of memories
        self.capacity = capacity

        # Current number of memories
        self.length = 0

        # Consolidated memories
        self.tree = None

        # Whether the memory has been modified since consolidation
        self.modified = False

        # Number of duplicates
        self.num_duplicates = 0

        # Brain
        self.brain = None

        # If using TensorFlow
        self.tensorflow = tensorflow

    def start_brain(self, projection, name_scope="memory"):
        if self.tensorflow:
            with tf.name_scope(name_scope):
                # Parameters, placeholders, and components
                parameters = {}
                placeholders = {}
                components = {}

                # TODO allow projection to be None and pass in raw parameters (either here or through brains)

                # Parameters TODO change midstream dim to output dim in brains
                parameters.update({"input_dim": projection.parameters["midstream_dim"],
                                   "output_dim": projection.parameters["memory_embedding_dim"]})

                # Placeholders
                placeholders.update(projection.placeholders)

                # Components
                components.update({"inputs": projection.brain, "mask": projection.components["mask"]})

                # Embedding weights and bias
                embedding_weights = tf.get_variable("memory_embedding_weights",
                                                    [parameters["input_dim"], parameters["output_dim"]])
                embedding_bias = tf.get_variable("memory_embedding_bias", [parameters["output_dim"]])

                # Embedding for memory
                memory_embedding = tf.einsum('aij,jk->aik', components["inputs"], embedding_weights)
                memory_embedding += embedding_bias

                # Give mask the right dimensionality
                mask = tf.tile(components["mask"], [1, 1, parameters["output_dim"]])

                # Mask for canceling out padding in dynamic sequences
                memory_embedding *= mask

                # Components
                components.update({"embedding_weights": embedding_weights, "embedding_bias": embedding_bias,
                                   "memory_embedding": memory_embedding})

                # Brain
                self.brain = Brains.Brains(brain=memory_embedding, placeholders=placeholders, components=components)

    def represent(self, placeholders=None, partial_run_setup=None):
        # Get memory representation
        return self.brain.run(placeholders, partial_run_setup=partial_run_setup)

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

        # Modified
        self.modified = True

    def retrieve(self, scene, k):
        # Retrieve k most similar memories #
        dist, ind = self.tree.query([scene], k=min(k, self.length))
        # ind, dist = self.tree.nn_index(experience.reshape(1, experience.shape[0]), min(k, self.length))

        # Return memories
        return dist[0], ind[0]

    def consolidate(self, short_term_memories=None, leaf_size=400):
        # If there are short term memories available
        if short_term_memories is not None:
            # Store short term memories
            for memory in range(short_term_memories.length):
                self.store(short_term_memories.get_memory_by_index(memory), True)

            # Empty out short term memories
            short_term_memories.reset()

        # If memories in memory and modified, consolidate tree
        if self.length > 0 and self.modified:
            # Build tree of long term memories
            self.tree = KDTree(self.memories[list(self.attributes.keys())[0]][:self.length], leaf_size=leaf_size)
            # self.tree = KDTree(self.memories[:self.length, :-num_attributes], leaf_size=math.ceil(self.length / 250))
            # self.tree = FLANN()
            # self.tree.build_index(self.memories[:self.length, :-num_attributes])
            # self.tree = KNeighborsRegressor(n_neighbors=min(50, self.length))
            # self.tree.fit(self.memories[:self.length, :-num_attributes],
            #               self.memories[:self.length, self.attributes["value"]])

            # Modified
            self.modified = False

    def get_memory_by_index(self, index):
        # Initialize memory
        memory = {}

        # Set memory
        for attribute, dimensionality in self.attributes.items():
            memory[attribute] = self.memories[attribute][index]

        # Return memory
        return memory

    def reset(self, population=None):
        # Reset internal states
        self.tree = None
        self.length = 0

        # For each memory attribute
        for attribute, dimensionality in self.attributes.items():
            # If no population provided
            if population is None:
                # Use default empty memories
                if dimensionality > 1:
                    self.memories[attribute] = np.zeros((self.capacity, dimensionality))
                else:
                    self.memories[attribute] = np.zeros(self.capacity)
            else:
                # Otherwise, set memories to population and verify dimensions
                shape = population[attribute].shape
                assert shape[1] == dimensionality if dimensionality > 1 else len(shape) == 1
                self.memories[attribute] = population[attribute]
                if not (self.length == shape[0] or self.length == 0):
                    print(self.length, shape[0], attribute, dimensionality)
                assert self.length == shape[0] or self.length == 0
                self.length = shape[0]

        # Modified
        self.modified = True

    def adapt(self, capacity=None, attributes=None, tensorflow=None):
        # Bodies
        bodies = []

        # Genes
        genes = [self.capacity, self.attributes, self.tensorflow]

        # Mutations
        body_mutations = []
        gene_mutations = [capacity, attributes, tensorflow]

        # Default bodies
        for i, mutation in enumerate(body_mutations):
            if mutation is None and bodies[i] is not None:
                body_mutations[i] = bodies[i].adapt()

        # Default genes
        for i, mutation in enumerate(gene_mutations):
            if mutation is None:
                gene_mutations[i] = genes[i]

        # Return adapted agent
        return self.__class__(capacity=gene_mutations[0], attributes=gene_mutations[1], tensorflow=gene_mutations[2])


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

            # Modified
            self.modified = True
        else:
            # Increment duplicates counter
            self.num_duplicates += 1

            # Reconcile duplicate by using one with max value
            if self.memories["value"][duplicate_index] < memory["value"]:
                for attribute, dimensionality in self.attributes.items():
                    self.memories[attribute][duplicate_index] = memory[attribute]
