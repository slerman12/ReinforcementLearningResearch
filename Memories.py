from __future__ import division
import math
import random

import numpy as np
# from pyflann.index import FLANN
from copy import deepcopy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors.kd_tree import KDTree
import Brains
import tensorflow as tf


class Traces:
    def __init__(self, capacity, attributes, reward_discount, memories, experience_replay=None):
        # Initialize dict of traces
        self.traces = {}

        # Arrays of memory traces for calculating values from rewards
        for attribute, dimensionality in attributes.items():
            if isinstance(dimensionality, int):
                if dimensionality > 1:
                    self.traces[attribute] = np.zeros((capacity, dimensionality))
                else:
                    self.traces[attribute] = np.zeros(capacity)
            elif isinstance(dimensionality, tuple):
                self.traces[attribute] = np.zeros((capacity,) + dimensionality)
            else:
                self.traces[attribute] = np.zeros([capacity] + dimensionality)

        # Indices of attributes (such as scene, reward, value, etc.)
        self.attributes = attributes

        # Max number of memories
        self.capacity = capacity

        # Current number of memory traces
        self.length = 0

        # Reward discount
        self.reward_discount = reward_discount

        # Memories to store into
        self.memories = memories

        # Experience replay to store into
        self.experience_replay = experience_replay

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
            if isinstance(self.memories, list):
                self.memories[int(memory["action"])].store(memory)
            else:
                self.memories.store(memory)

            # Add memory to experience replay
            if self.experience_replay is not None:
                self.experience_replay.store(memory)

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
                if isinstance(self.memories, list):
                    self.memories[int(trace["action"])].store(trace)
                else:
                    self.memories.store(trace)

                # Add memory to experience replay
                if self.experience_replay is not None:
                    self.experience_replay.store(trace)

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
            if isinstance(dimensionality, int):
                if dimensionality > 1:
                    self.traces[attribute] = np.zeros((self.capacity, dimensionality))
                else:
                    self.traces[attribute] = np.zeros(self.capacity)
            elif isinstance(dimensionality, tuple):
                self.traces[attribute] = np.zeros((self.capacity,) + dimensionality)
            else:
                self.traces[attribute] = np.zeros([self.capacity] + dimensionality)


class Memories:
    def __init__(self, capacity, attributes, target_attribute="target", num_similar_memories=32, tensorflow=True):
        # Initialize memories
        self.memories = {}

        # Arrays of memory
        for attribute, dimensionality in attributes.items():
            if isinstance(dimensionality, int):
                if dimensionality > 1:
                    self.memories[attribute] = np.zeros((capacity, dimensionality))
                else:
                    self.memories[attribute] = np.zeros(capacity)
            elif isinstance(dimensionality, tuple):
                self.memories[attribute] = np.zeros((capacity,) + dimensionality)
            else:
                self.memories[attribute] = np.zeros([capacity] + dimensionality)

        # Max number of memories
        self.capacity = capacity

        # Memory attributes (such as scene, state, reward, value, etc.) and their dimensionality
        self.attributes = attributes

        # Attribute being predicted
        self.target_attribute = target_attribute

        # Number of similar memories to retrieve
        self.num_similar_memories = num_similar_memories

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

                # Parameters
                parameters.update({"input_dim": projection.parameters["output_dim"],
                                   "representation_dim": self.attributes["representation"],
                                   "output_dim": self.attributes[self.target_attribute]})

                # Placeholders
                placeholders.update(projection.placeholders)

                # Components
                components.update({"inputs": projection.brain})
                if projection.components is not None:
                    if "mask" in projection.components:
                        components.update({"mask": projection.components["mask"]})

                # Embedding weights and bias
                embedding_weights = tf.get_variable("memory_embedding_weights",
                                                    [parameters["input_dim"], parameters["representation_dim"]])
                embedding_bias = tf.get_variable("memory_embedding_bias", [parameters["representation_dim"]])

                # Embedding for memory TODO account for 1-dim
                representation = tf.einsum("aj,jk->ak" if len(projection.brain.shape) == 2 else "aij,jk->aik",
                                           projection.brain, embedding_weights) + embedding_bias

                # Standardize to batch x time x dim even if no time provided
                if len(projection.brain.shape) == 2:
                    representation = tf.expand_dims(representation, axis=1)

                # If mask needed
                if "mask" in components:
                    # Mask for canceling out padding in dynamic sequences
                    representation *= tf.tile(components["mask"], [1, 1, parameters["representation_dim"]])

                # Components
                components.update({"representation": representation})

                # Retrieved memories
                remembered_representations = tf.placeholder("float", [None, None, self.num_similar_memories,
                                                                      self.attributes["representation"]])
                remembered_attributes = tf.placeholder("float", [None, None, self.num_similar_memories,
                                                                 parameters["output_dim"]])

                # Placeholders
                placeholders.update({"remembered_representations": remembered_representations,
                                     "remembered_attributes": remembered_attributes})

                # To prevent division by 0 and to prevent NaN gradients from square root of 0
                distance_delta = 0.001

                # Distances (batch x time x k)
                distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(
                    tf.tile(tf.expand_dims(representation, 2), [1, 1, self.num_similar_memories, 1]),
                    remembered_representations), 3) + distance_delta ** 2)
                # distances = tf.reduce_sum(tf.squared_difference(
                #     tf.tile(tf.expand_dims(representation, 2), [1, 1, self.num_similar_memories, 1]),
                #     remembered_representations), 3)

                # Weights (batch x time x k x attributes)
                weights = tf.tile(tf.expand_dims(1.0 / distances, axis=3), [1, 1, 1, parameters["output_dim"]])
                # weights = tf.tile(tf.expand_dims(1.0 / (distances + distance_delta), axis=3),
                #                   [1, 1, 1, self.attributes["attributes"]])

                # Division numerator and denominator (for weighted means)
                numerator = tf.reduce_sum(weights * remembered_attributes, axis=2)  # Weigh attributes
                denominator = tf.reduce_sum(weights, axis=2)  # Normalize weightings

                # In case denominator is zero
                safe_denominator = tf.where(tf.less(denominator, 1e-7), tf.ones_like(denominator), denominator)

                # Distance weighted memory attributes (batch x time x attributes)
                outputs = tf.divide(numerator, safe_denominator)

                # If mask needed
                if "mask" in components:
                    # Apply mask to outputs
                    outputs *= tf.tile(components["mask"], [1, 1, parameters["output_dim"]])

                # Components
                components.update({"outputs": outputs, "expectation": outputs})

                # Brain
                self.brain = Brains.Brains(brain=outputs, parameters=parameters, placeholders=placeholders,
                                           components=components)

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

    def retrieve(self, representation, perception_of_time=None, time_dims=None):
        # Batch x time inputs
        if len(representation.shape) < 3:
            if len(representation.shape) == 1:
                representation = np.expand_dims(representation, axis=0)
            representation = np.expand_dims(representation, axis=1)

        # Initialize time dims
        if time_dims is None:
            time_dims = np.full(shape=representation.shape[0], fill_value=representation.shape[1])

        # Initialize remembered attributes
        remembered = {attribute: np.zeros([representation.shape[0], representation.shape[1], self.num_similar_memories,
                                           self.attributes[attribute]] if isinstance(self.attributes[attribute], int)
                                          else [representation.shape[0], representation.shape[1],
                                                self.num_similar_memories] + list(self.attributes[attribute]))
                      for attribute in self.attributes}

        # Initialize duplicate (negative means no duplicate)
        duplicate = []

        # For each item in batch
        for batch_dim in range(representation.shape[0]):
            # For each time
            for time_dim in range(time_dims[batch_dim]):
                # Number of similar memories
                num_similar_memories = min(self.length, self.num_similar_memories)

                # If there are (enough) memories to draw from (changed from comparing to 0 and added self.tree check)
                if num_similar_memories == self.num_similar_memories and self.tree is not None:
                    # If not all zero
                    if representation[batch_dim, time_dim].any():
                        # Retrieve most similar memories
                        distances, indices = self.tree.query([representation[batch_dim, time_dim]],
                                                             k=min(self.num_similar_memories, self.length))
                        # dist = [np.ones(k), 0]
                        # ind = [np.ones(k, dtype=np.int32), 0]
                        # ind, dist = self.tree.nn_index(experience.reshape(1, experience.shape[0]), min(k, self.length))

                        # Return memories

                        # Set remembered attributes
                        for attribute in self.attributes:
                            if isinstance(self.attributes[attribute], int):  # TODO allow multi-dim
                                remembered[attribute][batch_dim, time_dim] = np.expand_dims(
                                    self.memories[attribute][indices[0]], 1) if self.attributes[attribute] == 1 else \
                                    self.memories[attribute][indices[0]]

                        # Set time accessed
                        if "time_accessed" in self.attributes:
                            self.memories["time_accessed"][indices[0]] = perception_of_time

        # # Attributes
        # assert "representations" in remembered
        # remembered_attributes = []
        # if "attributes" not in remembered:
        #     # remembered["attributes"] =
        #     for attribute in remembered:
        #         if attribute != "representation":
        #             remembered_attributes.append(remembered[attribute])
        # remembered_attributes = np.stack(remembered_attributes)

        # Return remembered
        return remembered

    def represent(self, placeholders=None, do_partial_run=False):
        # Get memory representation
        return self.brain.run(placeholders, "representation", do_partial_run=do_partial_run)

    def expect(self, placeholders=None, do_partial_run=False):
        # Get memory
        return self.brain.run(placeholders, "expectation", do_partial_run=do_partial_run)

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

    def random_batch(self, batch_dim=32):
        # Batch indices
        indices = random.sample(range(self.length), min(batch_dim, self.length))

        # Return batch
        return {attribute: self.memories[attribute][indices] for attribute in self.attributes}

    def reset(self, population=None):
        # Reset internal states
        self.tree = None
        self.length = 0

        # For each memory attribute  TODO multi-dim memories
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

    def adapt(self, capacity=None, attributes=None, num_similar_memories=None, tensorflow=None):
        # Bodies
        bodies = []

        # Genes
        genes = [self.capacity, self.attributes, self.num_similar_memories, self.tensorflow]

        # Mutations
        body_mutations = []
        gene_mutations = [capacity, attributes, num_similar_memories, tensorflow]

        # Default bodies
        for i, mutation in enumerate(body_mutations):
            if mutation is None and bodies[i] is not None:
                body_mutations[i] = bodies[i].adapt()

        # Default genes
        for i, mutation in enumerate(gene_mutations):
            if mutation is None:
                gene_mutations[i] = genes[i]

        # Return adapted agent
        return self.__class__(capacity=gene_mutations[0], attributes=gene_mutations[1],
                              num_similar_memories=gene_mutations[2], tensorflow=gene_mutations[3])


class MFEC(Memories):
    def store(self, memory, check_duplicate=False):
        # Duplicate index (positive if exists)
        check_duplicate = check_duplicate and not np.isnan(memory["duplicate"])
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
