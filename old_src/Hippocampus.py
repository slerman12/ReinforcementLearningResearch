from __future__ import division
import numpy as np


# Divide into two and put in Memory
class Attributes:
    def __init__(self, keys, indices, values=None):
        # Names of attributes
        self.keys = keys

        # Indices of attributes in array
        self.indices = {}
        for i, key in enumerate(keys):
            self.indices[key] = indices[i]

        # Values assigned to attributes
        if values is not None:
            self.values = {}
            for i, key in enumerate(keys):
                self.values[key] = values[i]

    def __repr__(self):
        return self.get_attribute_names()

    def get_attribute_names(self):
        return self.keys

    def index(self, key):
        return self.indices[key]

    def value(self, key):
        return self.values[key]

    def set(self, key, value):
        self.values[key] = value


class Memory:
    def __init__(self, memories, attributes):
        self.memories = memories
        self.attributes = attributes

        self.memory_pointer = None
        self.array = None
        self.duplicate = None

    def set_memory_pointer(self, index):
        self.memory_pointer = index

    def get_array(self):
        return self.memories.memories_array[self.memory_pointer]

    def is_duplicate(self):
        return self.duplicate is not None

    def reconcile_duplicates(self):
        # Check if duplicate
        is_dup = self.is_duplicate()

        if is_dup:
            # Reconcile duplicates by taking the maximum reward
            self.attributes.set("reward",
                                max(self.attributes.value("reward"), self.duplicate.attributes.value("reward")))

        # Return true if duplicate
        return is_dup

    def replace_with(self, memory):
        # Copy all attributes (while keeping all pointers)
        for attribute in self.attributes:
            self.attributes.set(attribute, memory.attributes.value(attribute))
            self.get_array()[self.attributes.index(attribute)] = self.attributes.value(attribute) # Doesn't work for multi-dim indexing


class Memories:
    def __init__(self, capacity, width, attributes):
        # Number of memories
        self.length = 0

        # Perception of time
        self.time = 0

        # Max capacity to be stored
        self.capacity = capacity

        # Dimensions per individual memory
        self.width = width

        # Array indices for attributes such as reward, time, etc.
        self.attributes = attributes

        # Memory objects
        self.memories = []

        # Memory array
        self.memories_array = np.zeros(self.capacity, self.width)

    def store(self, memory):
        # Reconcile duplicates if there is one
        if not memory.reconcile_duplicates():
            # If memory capacity has been reached
            if self.is_capacity_reached():
                # Replace least recently accessed memory with new memory
                self.get_least_recently_accessed_memory().replace_with(memory)
            else:
                # Add memory
                self.add_to_memories(memory)

        # Update time
        self.increment_time()

    def retrieve(self, index):
        # Retrieve memory and update access time
        memory = self.memories[index]
        memory.set("access_time", self.time)
        return memory

    def get_array(self):
        # Return memories array
        return self.memories_array[:self.length]

    def add_to_memories(self, memory):
        # Set pointers
        memory.set_memories_pointer(self.length)

        # Append memory
        self.memories.append(memory)
        self.memories_array[self.length] = memory.get_array()

        # Update length
        self.length += 1

    def is_capacity_reached(self):
        # Check if length exceeds capacity
        return self.length >= self.capacity

    def get_least_recently_accessed_memory(self):
        # Return memory with the smallest value for access time
        index = np.argmin(self.get_array()[self.attributes.index("time")])
        return self.retrieve(index)

    def increment_time(self):
        # Increment internal clock
        self.time += 0.01


class Hippocampus:
    def __init__(self, width, short_term_horizon, long_term_horizon):
        self.width = width
        self.long_term_horizon = long_term_horizon
        self.short_term_horizon = short_term_horizon

    class ShortTerm:
        def __init__(self, short_term_horizon, width):
            self.pointer = 0
            self.short_term_horizon = short_term_horizon
            self.width = width

            self.memory_array = np.zeros(self.short_term_horizon, self.width)

        def store(self, memory):
            assert(self.pointer < self.short_term_horizon)
            self.memory_array[self.pointer] = memory.get_array()
            self.pointer += 1

        def update(self):



    def add(self, experience):