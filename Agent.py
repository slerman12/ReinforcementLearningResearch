from __future__ import division
import time
import numpy as np


class Agent:
    # For measuring performance
    timer = 0

    def __init__(self, vision, long_term_memory, short_term_memory, traces, attributes, actions, epsilon, k):
        # Visual model
        self.vision = vision

        # Memories
        self.long_term_memory = long_term_memory
        self.short_term_memory = short_term_memory

        # Traces
        self.traces = traces

        # Indices of attributes (such as reward, value, etc.)
        self.attribute_indices = attributes

        # Parameters
        self.actions = actions
        self.epsilon = epsilon
        self.k = k

    def see(self, state):
        # Start timing
        start_time = time.time()

        # See state
        scene = self.vision.see(state) if self.vision is not None else state

        # Measure time
        self.timer = time.time() - start_time

        # Return scene
        return scene

    def act(self, scene):
        # Start timing
        start_time = time.time()

        # Number of actions
        num_actions = self.actions.size

        # Initialize expected values
        expected_per_action = np.zeros(num_actions)

        # Initialize duplicate (negative means no duplicate) and respective action
        duplicate = np.full(num_actions, -1)

        # Get expected value for each action
        for action in self.actions:
            # Initialize expected value at 0
            expected = 0

            # Number of similar memories
            num_similar_memories = min(self.long_term_memory[action].length, self.k)

            # If there are memories to draw from
            if num_similar_memories > 0:
                # Similar memories
                distances, indices = self.long_term_memory[action].retrieve(scene, self.k)

                # For each similar memory
                for i in range(num_similar_memories):
                    # Value index
                    value_index = self.attribute_indices["value"]

                    # Note if duplicate and if so use its value
                    if distances[i] == 0:
                        # Note duplicate and action
                        duplicate[action] = indices[i]

                        # Use this value
                        expected = self.long_term_memory[action].memories[indices[i], value_index]
                        break

                    # Add to running sum of values
                    expected += self.long_term_memory[action].memories[indices[i], value_index]

                # Finish computing expected value
                if duplicate[action] < 0:
                    expected /= num_similar_memories
                # expected = self.long_term_memory[action].tree.predict([scene])[0]

            # Record expected value for this action
            expected_per_action[action] = expected

        # Decide whether to explore according to epsilon probability
        explore = np.random.choice(np.arange(2), p=[1 - self.epsilon, self.epsilon])

        # Choose best action or explore
        action = expected_per_action.argmax() if not explore else np.random.choice(np.arange(self.actions.size))

        # Measure time
        self.timer = time.time() - start_time

        # Return the chosen action, the expected return value, and whether or not this experience happened before
        return self.actions[action], expected_per_action[action], duplicate[action]

    def experience(self, scene, action, reward, expected, duplicate, terminal):
        # Start timing
        start_time = time.time()

        # Attribute data TODO: make each attribute, including state, a unique array in a dict
        num_attributes = self.attribute_indices["num_attributes"]
        action_index = self.attribute_indices["action"]
        reward_index = self.attribute_indices["reward"]
        expected_index = self.attribute_indices["expected"]
        duplicate_index = self.attribute_indices["duplicate"]
        terminal_index = self.attribute_indices["terminal"]

        # Set attributes
        attributes = np.zeros(num_attributes)
        attributes[action_index] = action
        attributes[reward_index] = reward
        attributes[expected_index] = expected
        attributes[duplicate_index] = duplicate
        attributes[terminal_index] = terminal

        # Add trace
        self.traces.add(trace=np.append(scene, attributes))

        # Measure time
        self.timer = time.time() - start_time

    def learn(self):
        # Start timing
        start_time = time.time()

        # Consolidate memories
        for action in self.actions:
            self.long_term_memory[action].consolidate(short_term_memories=self.short_term_memory[action])

        # Measure time
        self.timer = time.time() - start_time
