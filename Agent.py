from __future__ import division
import time
import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, vision=None, long_term_memory=None, short_term_memory=None, traces=None, attributes=None,
                 actions=None, exploration_rate=None, k=None):
        # For measuring performance
        self.timer = 0

        # Internal perception of time
        self.perception_of_time = 0

        # Visual model
        self.vision = vision

        # Memories
        self.long_term_memory = long_term_memory
        self.short_term_memory = short_term_memory

        # Traces
        self.traces = traces

        # Attributes (such as state, scene, reward, value, etc.)
        self.attributes = attributes

        # Parameters
        self.actions = actions
        self.exploration_rate = exploration_rate
        self.k = k

        # Learning
        self.brain = self.session = self.loss = self.train = self.accuracy = None

    def start_brain(self):
        # Default brain
        if self.vision is not None:
            self.brain = self.vision.brain

    def stop_brain(self):
        # Close tensorflow session
        tf.Session.close(self.session)

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

        # Initialize duplicate (negative means no duplicate)
        duplicate = []

        # Memory Module
        if self.long_term_memory is not None:
            # Number of similar memories
            num_similar_memories = min(self.long_term_memory.length, self.k)

            # If there are memories to draw from
            if num_similar_memories > 0:
                # Similar memories
                distances, indices = self.long_term_memory.retrieve(scene, self.k)

                # Weights
                weights = 0

                # For each similar memory
                for i in range(num_similar_memories):
                    # Set time accessed
                    self.long_term_memory.memories["time_accessed"][indices[i]] = self.perception_of_time

                    # Distance
                    distance = distances[i]

                    # Note if duplicate
                    if distance == 0:
                        # Note duplicate
                        duplicate.append(indices[i])

                    # Weight of each memory is inverse of distance
                    weight = 1 / (distance + 0.001)
                    weights += weight

        # Decide whether to explore according to epsilon probability
        # explore = 1 if random.random() < self.exploration_rate else 0
        explore = np.random.choice(np.arange(2), p=[1 - self.exploration_rate, self.exploration_rate])

        # Measure time
        self.timer = time.time() - start_time

    def experience(self, experience):
        # Start timing
        start_time = time.time()

        # Set time of memory and time accessed to current time
        experience["time"] = self.perception_of_time
        experience["time_accessed"] = self.perception_of_time

        # Update visual model
        if self.vision is not None:
            self.vision.experience(experience)

        # Add trace
        self.traces.add(experience)

        # Move perception of time forward
        self.move_time_forward()

        # Measure time
        self.timer = time.time() - start_time

    def learn(self, placeholders=None):
        # Start timing
        start_time = time.time()

        # Default variables
        if placeholders is None:
            placeholders = {}
        loss = None

        # Consolidate memories
        if self.long_term_memory is not None:
            self.long_term_memory.consolidate(self.short_term_memory)

        # Train brain
        if self.train is not None:
            _, loss = self.brain.run(placeholders, [self.train, self.loss])

        # Measure time
        self.timer = time.time() - start_time

        # Return loss
        return loss

    def move_time_forward(self):
        # Size of discrete time unit
        time_quantum = 0.1

        # Move perception of time forward
        self.perception_of_time += time_quantum


class MFEC(Agent):
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
                    # Set time accessed  # TODO: account for duplicate
                    self.long_term_memory[action].memories["time_accessed"][indices[i]] = self.perception_of_time

                    # Distance
                    distance = distances[i]

                    # Note if duplicate and if so use its value
                    if distance == 0:
                        # Note duplicate and action
                        duplicate[action] = indices[i]

                        # Use this value
                        expected = self.long_term_memory[action].memories["value"][indices[i]]
                        break

                    # Add to running sum of values
                    expected += self.long_term_memory[action].memories["value"][indices[i]]

                # Finish computing expected value
                if duplicate[action] < 0:
                    expected /= num_similar_memories

            # Record expected value for this action
            expected_per_action[action] = expected

        # Decide whether to explore according to epsilon probability
        explore = np.random.choice(np.arange(2), p=[1 - self.exploration_rate, self.exploration_rate])

        # Choose best action or explore
        action = expected_per_action.argmax() if not explore else np.random.choice(np.arange(self.actions.size))

        # Measure time
        self.timer = time.time() - start_time

        # Return the chosen action, the expected return value, and whether or not this experience happened before
        return self.actions[action], expected_per_action[action], duplicate[action]

    def learn(self, placeholders=None):
        # Start timing
        start_time = time.time()

        # Default variables
        if placeholders is None:
            placeholders = {}
        loss = None

        # Consolidate memories
        if self.long_term_memory is not None:
            for action in self.actions:
                self.long_term_memory[action].consolidate(short_term_memories=self.short_term_memory[action])

        # Train brain
        if self.train is not None:
            _, loss = self.brain.run(placeholders, [self.train, self.loss])

        # Measure time
        self.timer = time.time() - start_time

        # Return loss
        return loss


class NEC(MFEC):
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

                # Weights
                weights = 0

                # For each similar memory
                for i in range(num_similar_memories):
                    # Set time accessed
                    self.long_term_memory[action].memories["time_accessed"][indices[i]] = self.perception_of_time

                    # Distance
                    distance = distances[i]

                    # Note if duplicate
                    if distance == 0:
                        # Note duplicate and action
                        duplicate[action] = indices[i]

                    # Weight of each memory is inverse of distance
                    weight = 1 / (distance + 0.001)
                    weights += weight

                    # Add to running sum of values
                    expected += self.long_term_memory[action].memories["value"][indices[i]] * weight

                # Finish computing expected value
                expected /= weights

            # Record expected value for this action
            expected_per_action[action] = expected

        # Decide whether to explore according to epsilon probability
        explore = np.random.choice(np.arange(2), p=[1 - self.exploration_rate, self.exploration_rate])

        # Choose best action or explore
        action = expected_per_action.argmax() if not explore else np.random.choice(np.arange(self.actions.size))

        # Measure time
        self.timer = time.time() - start_time

        # Return the chosen action, the expected return value, and whether or not this experience happened before
        return self.actions[action], expected_per_action[action], duplicate[action]


class Classifier(Agent):
    def start_brain(self):
        # Use vision
        self.brain = self.vision.brain

        # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
        if "max_time_dim" in self.brain.parameters.keys() and len(self.brain.brain.shape.as_list()) > 2:
            # Cross entropy
            cross_entropy = self.brain.placeholders["desired_outputs"] * tf.log(tf.nn.softmax(self.brain.brain))
            cross_entropy = -tf.reduce_sum(cross_entropy, 2)

            # Mask for canceling out padding in dynamic sequences
            mask = tf.sign(tf.reduce_max(tf.abs(self.brain.placeholders["desired_outputs"]), 2))

            # Explicit masking of loss (necessary in case a bias was added to the padding!)
            cross_entropy *= mask

            # Average over the correct sequence lengths (this is where the mask is used)
            cross_entropy = tf.reduce_sum(cross_entropy, 1)
            cross_entropy /= tf.reduce_sum(mask, 1)

            # Average loss over each batch
            self.loss = tf.reduce_mean(cross_entropy)
        else:
            # Softmax cross entropy
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.brain.brain, labels=self.brain.placeholders["desired_outputs"]))

        # Training optimization method
        optimizer = tf.train.GradientDescentOptimizer(self.brain.parameters["learning_rate"])

        # If gradient clipping
        if "max_gradient_clip_norm" in self.brain.parameters.keys():
            # Trainable variables
            trainable_variables = tf.trainable_variables()

            # Get gradients of loss and clip them according to max gradient clip norm
            gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables),
                                                  self.brain.parameters["max_gradient_clip_norm"])

            # Training
            self.train = optimizer.apply_gradients(zip(gradients, trainable_variables))
        else:
            # Training
            self.train = optimizer.minimize(self.loss)

        # Test accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(self.brain.brain, 1), tf.argmax(self.brain.placeholders["desired_outputs"], 1)), tf.float32))

        # For initializing variables
        initialize_variables = tf.global_variables_initializer()

        # Start session
        self.session = tf.Session()

        # Initialize variables
        self.session.run(initialize_variables)

        # Assign session to brain
        self.brain.session = self.session


class TruncatedBPTTClassifier(Classifier):
    def learn(self, placeholders=None):
        # Allow total time parameter to be called either max_time_dim or time_dim
        total_time = self.brain.parameters["time_dim"] if "time_dim" in self.brain.parameters.keys() \
            else self.brain.parameters["max_time_dim"]

        # Assert truncated time divides total time (since I haven't made the sequence padding automatic yet) TODO
        assert total_time % self.brain.parameters["truncated_time_dim"] == 0

        # Start timing
        start_time = time.time()

        # Consolidate memories
        if self.long_term_memory is not None:
            for action in self.actions:
                self.long_term_memory[action].consolidate(short_term_memories=self.short_term_memory[action])

        # Initialize initial hidden state
        initial_state = self.brain.run(None, self.brain.placeholders["initial_state"])

        # Initialize sequence subset time boundaries for the truncated iteration
        sequence_subset_begin = 0
        sequence_subset_end = self.brain.parameters["truncated_time_dim"]

        # Counter of truncated iterations
        truncated_iterations = 0

        # Initialize loss
        loss = 0

        # Truncated iteration through each sequence
        while sequence_subset_begin < total_time:
            # Get sequence subsets
            inputs_subset = placeholders["inputs"][:, sequence_subset_begin:sequence_subset_end, :]
            desired_outputs_subset = placeholders["desired_outputs"][:, sequence_subset_begin:sequence_subset_end, :]

            # Placeholders for truncated iteration
            subset_placeholders = {"inputs": inputs_subset, "desired_outputs": desired_outputs_subset,
                                   "initial_state": initial_state}

            # If input sequences have dynamic numbers of time dimensions
            if "max_time_dim" in self.brain.parameters.keys():
                # Update the dynamic lengths across each iteration's subsets
                sequence_lengths_subset = np.maximum(np.minimum(placeholders["sequence_length"] -
                                                                self.brain.parameters["truncated_time_dim"] *
                                                                truncated_iterations,
                                                                self.brain.parameters["truncated_time_dim"]), 0)

                # If one of the sequences is all empty, break
                # TODO: Bucket batches to prevent breaking unfinished sequences and take batch subsets here otherwise
                if not inputs_subset.any(axis=2).any(axis=1).all():
                    break

                # Placeholders for truncated iteration
                subset_placeholders["sequence_length"] = sequence_lengths_subset

            # Train
            _, subset_loss, initial_state = self.brain.run(subset_placeholders, [self.train, self.loss,
                                                                                 self.brain.components["final_state"]])

            # Loss
            loss += subset_loss

            # Increment sequence subset boundaries
            sequence_subset_begin += self.brain.parameters["truncated_time_dim"]
            sequence_subset_end += self.brain.parameters["truncated_time_dim"]

            # Increment truncated iterations counter
            truncated_iterations += 1

        # Normalize loss
        loss /= truncated_iterations

        # Measure time
        self.timer = time.time() - start_time

        # Return loss
        return loss


class Regressor(Agent):
    def start_brain(self):
        # Use vision
        self.brain = self.vision.brain

        # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
        if "max_time_dim" in self.brain.parameters.keys() and len(self.brain.brain.shape.as_list()) > 2:
            # Mean squared difference
            mean_squared_difference = tf.reduce_mean(
                tf.squared_difference(self.brain.brain, self.brain.placeholders["desired_outputs"]), axis=2)

            # Mask for canceling out padding in dynamic sequences
            mask = tf.sign(tf.reduce_max(tf.abs(self.brain.placeholders["desired_outputs"]), 2))

            # Explicit masking of loss (necessary in case a bias was added to the padding!)
            mean_squared_difference *= mask

            # Average loss over each (padded) sequence and batch
            self.loss = tf.reduce_mean(tf.reduce_sum(mean_squared_difference, 1) / tf.reduce_sum(mask, 1))
        else:
            # Mean squared error
            self.loss = tf.losses.mean_squared_error(self.brain.placeholders["desired_outputs"], self.brain.brain)

        # Training optimization method
        optimizer = tf.train.AdamOptimizer(self.brain.parameters["learning_rate"])

        # If gradient clipping
        if "max_gradient_clip_norm" in self.brain.parameters.keys():
            # Trainable variables
            trainable_variables = tf.trainable_variables()

            # Get gradients of loss and clip them according to max gradient clip norm
            gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables),
                                                  self.brain.parameters["max_gradient_clip_norm"])

            # Training
            self.train = optimizer.apply_gradients(zip(gradients, trainable_variables))
        else:
            # Training
            self.train = optimizer.minimize(self.loss)

        # Test accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(self.brain.brain, 1), tf.argmax(self.brain.placeholders["desired_outputs"], 1)), tf.float32))

        # For initializing variables
        initialize_variables = tf.global_variables_initializer()

        # Start session
        self.session = tf.Session()

        # Initialize variables
        self.session.run(initialize_variables)

        # Assign session to brain
        self.brain.session = self.session
