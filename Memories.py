import numpy as np
from pyflann.index import FLANN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors.kd_tree import KDTree


class Traces:
    def __init__(self, capacity, width, attributes, memories, gamma):
        # Array of memory traces for calculating values from rewards
        self.traces = np.zeros((capacity, width))

        # Max number of memories
        self.capacity = capacity

        # Dimensions per memory
        self.width = width

        # Current number of memory traces
        self.length = 0

        # Indices of attributes (such as reward, value, etc.)
        self.attributes = attributes

        # Memories to store into
        self.memories = memories

        # Reward discount
        self.gamma = gamma

    def add(self, trace):
        # Reward index
        reward_index = self.attributes["reward"]

        # Value index
        value_index = self.attributes["value"]

        # Action index
        action_index = self.attributes["action"]

        # Terminal index
        terminal_index = self.attributes["terminal"]

        # Reward of trace
        reward = trace[reward_index]

        # Initialize trace's value to reward
        trace[value_index] = reward

        # Update values of existing traces except oldest
        # for i in range(1, self.length):
        for i in range(0, self.length):
            self.traces[i, value_index] += self.gamma ** (self.length - i) * reward

        # If memory capacity has not been reached
        if self.length < self.capacity:
            # Update oldest trace
            # self.traces[0, value_index] += self.gamma ** self.length * reward

            # Add trace
            self.traces[self.length] = trace

            # Update length
            self.length += 1
        else:
            # Oldest trace
            memory = self.traces[0]

            # Expected value index
            expected_index = self.attributes["expected"]

            # Update memory with off-policy prediction
            # memory[value_index] = self.gamma ** self.length * trace[expected_index]

            # Add memory to long term memory
            # TODO: Since I'm not updating the KD tree at every step, it is possible for a duplicate to enter memory
            self.memories[int(memory[action_index])].store(memory)

            # Pop memory from traces
            self.traces[:-1] = self.traces[1:]

            # Add trace
            self.traces[self.length - 1] = trace

        # If terminal state
        if trace[terminal_index]:
            # Dump all traces into memory
            for i in range(self.length):
                trace = self.traces[i]
                self.memories[int(trace[action_index])].store(trace)

            # Reset traces
            self.traces = np.zeros((self.capacity, self.width))
            self.length = 0


class Memories:
    # Internal perception of time
    time = 0

    # Consolidated memories
    tree = None

    # Number of duplicates
    num_duplicates = 0

    def __init__(self, capacity, width, attributes):
        # Array of long term memory
        self.memories = np.zeros((capacity, width))

        # Max number of memories
        self.capacity = capacity

        # Dimensions per memory
        self.width = width

        # Current number of memories
        self.length = 0

        # Indices of attributes (such as reward, value, etc.)
        self.attributes = attributes

    def store(self, memory, check_duplicate=False):
        # Set time accessed to current time
        self.set_time_accessed(memory)

        # Duplicate index (positive if exists)
        duplicate_index = int(memory[self.attributes["duplicate"]]) if check_duplicate else -1

        # Value index
        value_index = self.attributes["value"]

        # If not a duplicate
        if duplicate_index < 0:
            # If memory capacity has not been reached
            if self.length < self.capacity:
                # Add memory
                self.memories[self.length] = memory

                # Update length
                self.length += 1
            else:
                # Time last accessed index
                time_accessed_index = self.attributes["time_accessed"]

                # Replace least recently accessed memory with new memory
                self.memories[np.argmin(self.memories[:self.length, time_accessed_index])] = memory
        else:
            # Increment duplicates counter
            self.num_duplicates += 1

            # Pair of duplicates
            duplicates = [self.memories[duplicate_index], memory]

            # Reconcile duplicate by using one with max value
            chosen = duplicates[int(np.argmax([dup[value_index] for dup in duplicates]))]

            # Reconcile duplicate TODO: Update other attributes as well
            self.memories[duplicate_index, value_index] = chosen[value_index]

            # Set time accessed
            self.set_time_accessed(self.memories[duplicate_index])

    def retrieve(self, experience, k):
        # Retrieve k most similar memories
        dist, ind = self.tree.query([experience], k=min(k, self.length))
        # ind, dist = self.tree.nn_index(experience.reshape(1, experience.shape[0]), min(k, self.length))

        # Update access times
        for i in ind:
            self.set_time_accessed(self.memories[i])

        # Return memories
        return dist[0], ind[0]

    def consolidate(self, short_term_memories):
        # If there are short term memories available
        if short_term_memories.length > 0:
            # Store short term memories
            for memory in range(short_term_memories.length):
                self.store(short_term_memories.memories[memory], True)

            # Empty out short term memories
            short_term_memories.reset()

            # Number of attributes
            num_attributes = self.attributes["num_attributes"]

            # Build tree of long term memories
            self.tree = KDTree(self.memories[:self.length, :-num_attributes], leaf_size=400)
            # self.tree = FLANN()
            # self.tree.build_index(self.memories[:self.length, :-num_attributes])
            # self.tree = KNeighborsRegressor(n_neighbors=min(50, self.length))
            # self.tree.fit(self.memories[:self.length, :-num_attributes],
            #               self.memories[:self.length, self.attributes["value"]])

    def set_time_accessed(self, memory):
        # Time of last access index
        time_accessed_index = self.attributes["time_accessed"]

        # Set time of access
        memory[time_accessed_index] = self.time

        # Increment time
        self.time += 0.1
        
    def reset(self):
        # Reset internal states
        self.tree = None
        self.length = 0
        self.memories = np.zeros((self.capacity, self.width))
