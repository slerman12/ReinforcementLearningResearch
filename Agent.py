from __future__ import division
import time
import numpy as np
import tensorflow as tf
import Brains


class Agent:
    def __init__(self, vision=None, long_term_memory=None, short_term_memory=None, traces=None, attributes=None,
                 actions=None, exploration_rate=None, k=None, tensorflow=True, scope_name="training", session=None):
        # Start timing (wall time)
        start_time = time.time()

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

        # Brain defaults
        self.brain = self.loss = self.train = self.accuracy = \
            self.learning_steps = self.increment_learning_steps = self.gradients = self.variables = \
            self.tensorflow_partial_run = self.tensorflow_saver = self.tensorboard_writer = self.tensorboard_logs = None

        # If using TensorFlow
        self.tensorflow = tensorflow

        # If using TensorFlow, create contexts
        if self.tensorflow:
            # Agent's scope name
            self.scope_name = scope_name

            # TensorFlow session
            self.session = session

            # Open scope, start brain graphs, and begin session
            self.start_tensorflow()

        # Measure time
        self.timer = time.time() - start_time

    def start_brain(self):
        # Default empty brain
        self.brain = Brains.Brains(tensorflow=self.tensorflow, session=self.session)

    def stop_brain(self):
        # Start timing
        start_time = time.time()

        # If using TensorFlow
        if self.tensorflow:
            # Close TensorFlow session
            tf.Session.close(self.session)

        # Measure time
        self.timer = time.time() - start_time

    def see(self, state, batch_dims=1, time_dims=1):
        # Start timing
        start_time = time.time()

        # Vision
        if self.vision is None:
            # See
            scene = state
        elif self.tensorflow:
            # Setup a partial run such that the graph is not recomputed later
            partial_run_fetches = [fetch for fetch in [self.train, self.loss] if fetch is not None]
            partial_run_setup = [partial_run_fetches, self.brain.placeholders] if len(partial_run_fetches) > 0 else None

            # See
            if partial_run_setup is None:
                scene = self.vision.see(state)
            else:
                scene, self.tensorflow_partial_run = self.vision.see(state, partial_run_setup=partial_run_setup)
        else:
            # See
            scene = self.vision.see(state)

        # Measure time
        self.timer = time.time() - start_time

        # Return scene
        return scene

    def remember(self, concept, batch_dims=1, max_time_dim=1, time_dims=None):
        # Start timing
        start_time = time.time()

        # Initialize memories
        remember_concepts = np.zeros([batch_dims, max_time_dim, self.k, self.attributes["concepts"]])
        remember_attributes = np.zeros([batch_dims, max_time_dim, self.k, self.attributes["attributes"]])

        # Initialize duplicate (negative means no duplicate)
        duplicate = []

        # Initialize default time dims
        if time_dims is None:
            time_dims = np.full(batch_dims, max_time_dim)

        # Memory Module
        if self.long_term_memory is not None:
            # For each batch
            for batch_dim in range(batch_dims):
                # For each time
                for time_dim in range(time_dims):
                    # If padding
                    if time_dim > time_dims[batch_dim]:
                        break
                    else:
                        # Number of similar memories
                        num_similar_memories = min(self.long_term_memory.length, self.k)

                        # If there are memories to draw from
                        if num_similar_memories > 0:
                            # Query to memory
                            query = concept[batch_dim, time_dim]

                            # If not all zero
                            if query.any():
                                # Similar memories TODO check if duplicate with distance == 0
                                distances, indices = self.long_term_memory.retrieve(query, self.k)
                                remember_concepts[batch_dim, time_dim] = \
                                    self.long_term_memory.memories["concepts"][indices]
                                remember_attributes[batch_dim, time_dim] = \
                                    self.long_term_memory.memories["attributes"][indices]

        # Measure time
        self.timer = time.time() - start_time

        # Return memories
        return remember_concepts, remember_attributes

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
        if self.traces is not None:
            self.traces.add(experience)

        # Move perception of time forward
        self.move_time_forward()

        # Measure time
        self.timer = time.time() - start_time

    def learn(self, placeholders=None):
        # Start timing
        start_time = time.time()

        # Defaults
        if placeholders is None:
            placeholders = {}
        loss = None

        # Consolidate memories
        if self.long_term_memory is not None:
            self.long_term_memory.consolidate(self.short_term_memory)

        # Train brain TODO make this and measure loss prettier
        if self.train is not None and self.tensorflow:
            if self.tensorboard_logs is None:
                if self.tensorflow_partial_run is None:
                    _, loss = self.brain.run(placeholders, [self.train, self.loss])
                else:
                    _, loss, _ = self.brain.run(placeholders, [self.train, self.loss], self.tensorflow_partial_run)
            else:
                learning_steps = self.brain.run(components=[self.increment_learning_steps])[0]

                if self.tensorflow_partial_run is None:
                    _, loss, logs = self.brain.run(placeholders, [self.train, self.loss, self.tensorboard_logs])
                else:
                    _, loss, logs, _ = self.brain.run(placeholders, [self.train, self.loss, self.tensorboard_logs],
                                                      self.tensorflow_partial_run)

                # Output TensorBoard data
                self.tensorboard_writer.add_summary(logs, learning_steps)

        # Measure time
        self.timer = time.time() - start_time

        # Return loss
        return loss

    def measure_loss(self, data):
        # Start timing
        start_time = time.time()

        # Get loss
        if self.tensorboard_logs is None:
            if self.tensorflow_partial_run is None:
                loss = self.brain.run(data, self.loss)
            else:
                loss, _ = self.brain.run(data, self.loss, partial_run=self.tensorflow_partial_run)
        else:
            learning_steps = self.brain.run(components=[self.learning_steps])[0]

            if self.tensorflow_partial_run is None:
                loss, logs = self.brain.run(data, [self.loss, self.tensorboard_logs])
            else:
                loss, logs, _ = self.brain.run(data, [self.loss, self.tensorboard_logs], self.tensorflow_partial_run)

            # Output TensorBoard data
            self.tensorboard_writer.add_summary(logs, learning_steps)

        # Measure time
        self.timer = time.time() - start_time

        # Return loss and logs
        return loss

    def move_time_forward(self):
        # Size of discrete time unit
        time_quantum = 0.1

        # Move perception of time forward
        self.perception_of_time += time_quantum

    def start_tensorflow(self):
        # Open scope, start brains (build graphs), and begin session
        with tf.name_scope(self.scope_name):
            with tf.variable_scope("brain", reuse=None or self.scope_name != "training", initializer=None):
                # Start vision
                if self.vision is not None:
                    self.vision.start_brain()

                # Start brain
                self.start_brain()

                # Learning steps counter
                self.learning_steps = tf.get_variable("learning_steps", [], tf.int32, tf.zeros_initializer(), False)
                self.increment_learning_steps = tf.assign_add(self.learning_steps, 1)

        # If no session provided
        if self.session is None:
            # For initializing variables
            initialize_variables = tf.global_variables_initializer()

            # Start session
            self.session = tf.Session()

            # Initialize variables (only for training)
            if self.scope_name == "training":
                self.session.run(initialize_variables)

        # Assign session to brains
        for brain in [self.brain, self.vision.brain]:
            if brain is not None:
                brain.session = self.session

        # Saver
        if self.scope_name == "training":
            self.tensorflow_saver = tf.train.Saver()

    def save(self, filename="Saved/brain"):
        # Save to file
        self.tensorflow_saver.save(self.session, filename)

    def load(self, filename="Saved/brain"):
        # max(glob.glob('Saved/*'), key=os.path.getctime)
        if tf.train.checkpoint_exists(filename):
            self.tensorflow_saver.restore(self.session, filename)

    def start_tensorboard(self, scalars=None, gradients=None, variables=None, tensorboard_writer=None,
                          directory_name="Logs"):
        # Defaults
        if scalars is None:
            scalars = {}

        # Scope
        with tf.name_scope(self.scope_name):
            # Logs for scalars
            for name in scalars:
                tf.summary.scalar(name, scalars[name])

            # Histogram logs for all weights
            if variables is not None:
                for variable in variables:
                    tf.summary.histogram(variable.name, variable)

            # Histogram logs for all gradients
            if gradients is not None:
                if variables is None:
                    variables = tf.trainable_variables()
                for gradient, variable in list(zip(gradients, variables)):
                    tf.summary.histogram(variable.name + '/gradient', gradient)

        # TensorBoard logs
        self.tensorboard_logs = None if scalars is None and gradients is None and variables is None else \
            tf.summary.merge_all(scope=self.scope_name)

        # TensorBoard writer
        if tensorboard_writer is None:
            self.tensorboard_writer = tf.summary.FileWriter(directory_name, graph=tf.get_default_graph())
            self.tensorboard_writer.flush()
        else:
            self.tensorboard_writer = tensorboard_writer


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

    def learn(self, placeholders=None, tensorboard_logs_scope_name=None):
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
        if self.train is not None and self.tensorflow:
            if tensorboard_logs_scope_name is None:
                _, loss = self.brain.run(placeholders, [self.train, self.loss])
            else:
                _, loss, logs = self.brain.run(placeholders, [self.train, self.loss, tensorboard_logs_scope_name])

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
    def start_brain(self, session=None):
        # Start timing
        start_time = time.time()

        # Default brain
        if self.brain is None:
            if self.vision is not None:
                self.brain = self.vision.brain

        # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
        if "max_time_dim" in self.brain.parameters and len(self.brain.brain.shape.as_list()) > 2:
            # # Cross entropy
            # cross_entropy = self.brain.placeholders["desired_outputs"] * tf.log(tf.nn.softmax(self.brain.brain))
            # cross_entropy = -tf.reduce_sum(cross_entropy, 2)
            #
            # # Mask for canceling out padding in dynamic sequences
            # mask = tf.sign(tf.reduce_max(tf.abs(self.brain.placeholders["desired_outputs"]), 2))
            #
            # # Explicit masking of loss (necessary in case a bias was added to the padding!)
            # cross_entropy *= mask
            #
            # # Average over the correct sequence lengths (this is where the mask is used)
            # cross_entropy = tf.reduce_sum(cross_entropy, 1)
            # cross_entropy /= tf.reduce_sum(mask, 1)
            #
            # # Average loss over each batch
            # self.loss = tf.reduce_mean(cross_entropy)

            # Alternative short form:
            mask = tf.sequence_mask(self.brain.placeholders["time_dims"], maxlen=self.brain.brain.shape[1])
            self.loss = tf.contrib.seq2seq.sequence_loss(self.brain.brain, self.brain.placeholders["desired_outputs"],
                                                         weights=mask)
        else:
            # Softmax cross entropy
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.brain.brain, labels=self.brain.placeholders["desired_outputs"]))

        # Training optimization method
        optimizer = tf.train.GradientDescentOptimizer(self.brain.placeholders["learning_rate"])

        # If gradient clipping
        if "max_gradient_clip_norm" in self.brain.parameters:
            # Trainable variables
            self.variables = tf.trainable_variables()

            # Get gradients of loss and clip them according to max gradient clip norm
            self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.variables),
                                                       self.brain.parameters["max_gradient_clip_norm"])

            # Training
            self.train = optimizer.apply_gradients(zip(self.gradients, self.variables))
        else:
            # Training
            self.train = optimizer.minimize(self.loss)

        # Test accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(self.brain.brain, 1), tf.argmax(self.brain.placeholders["desired_outputs"], 1)), tf.float32))

        # Measure time
        self.timer = time.time() - start_time

        # Return session
        return self.session


class TruncatedBPTTClassifier(Classifier):
    def learn(self, placeholders=None, tensorboard_timestep=None):
        # Allow total time parameter to be called either max_time_dim or time_dim
        total_time = self.brain.parameters["time_dim"] if "time_dim" in self.brain.parameters \
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
            if "max_time_dim" in self.brain.parameters:
                # Update the dynamic time dimensions across each iteration's subsets
                time_dims_subset = np.maximum(np.minimum(placeholders["time_dims"] -
                                                         self.brain.parameters["truncated_time_dim"] *
                                                         truncated_iterations,
                                                         self.brain.parameters["truncated_time_dim"]), 0)

                # If one of the sequences is all empty, break
                # TODO: Bucket batches to prevent breaking unfinished sequences and take batch subsets here otherwise
                if not inputs_subset.any(axis=2).any(axis=1).all():
                    break

                # Placeholders for truncated iteration
                subset_placeholders["time_dims"] = time_dims_subset

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
    def start_brain(self, session=None, tensorboard_writer=None):
        # Start timing
        start_time = time.time()

        # Default brain
        if self.brain is None:
            if self.vision is not None:
                self.brain = self.vision.brain

        # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
        if "max_time_dim" in self.brain.parameters and len(self.brain.brain.shape.as_list()) > 2:
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

        # Training
        if self.scope_name == "training":
            # Training optimization method
            optimizer = tf.train.AdamOptimizer(self.brain.placeholders["learning_rate"])

            # If gradient clipping
            if "max_gradient_clip_norm" in self.brain.parameters:
                # Trainable variables
                self.variables = tf.trainable_variables()

                # Get gradients of loss and clip them according to max gradient clip norm
                self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.variables),
                                                           self.brain.parameters["max_gradient_clip_norm"])

                # Training
                self.train = optimizer.apply_gradients(zip(self.gradients, self.variables))
            else:
                # Training
                self.train = optimizer.minimize(self.loss)

        # Measure time
        self.timer = time.time() - start_time

        # Return session
        return self.session


class LifelongMemory(Agent):
    def start_brain(self):
        # Desired outputs
        desired_outputs = tf.placeholder("float", [None, None, self.attributes["attributes"]])

        # Time ahead
        time_ahead = tf.placeholder("float", [None, None])

        # Retrieved memories
        remember_concepts = tf.placeholder("float", [None, None, self.k, self.attributes["concepts"]])
        remember_attributes = tf.placeholder("float", [None, None, self.k, self.attributes["attributes"]])

        # Placeholders (Note: this would probably modify the placeholders of vision as well if I didn't copy)
        placeholders = self.vision.brain.placeholders.copy()
        placeholders["desired_outputs"] = desired_outputs
        placeholders["time_ahead"] = time_ahead
        placeholders["remember_concepts"] = remember_concepts
        placeholders["remember_attributes"] = remember_attributes

        # Weights
        weights = tf.norm(remember_concepts - self.vision.brain.brain, axis=3)

        # Distance weighted memory attributes
        distance_weighted_memory_attributes = tf.divide(tf.reduce_sum(weights * remember_attributes, axis=2),
                                                        tf.reduce_sum(weights, axis=2))

        # Add time ahead before final dense layer
        outputs = tf.concat([distance_weighted_memory_attributes, tf.expand_dims(time_ahead, 2)], 2) \
            if self.vision.brain.parameters["time_ahead"] else distance_weighted_memory_attributes

        # Final dense layer weights
        output_weights = tf.get_variable("output_weights", [self.attributes["attributes"] + 1 if
                                                            self.vision.brain.parameters["time_ahead"] else
                                                            self.attributes["attributes"],
                                                            self.attributes["attributes"]])

        # Final dense layer bias
        output_bias = tf.get_variable("output_bias", [self.attributes["attributes"]])

        # Dense layer (careful: bias or cudnn would corrupt padding. Hence mask needed)
        outputs = tf.einsum('aij,jk->aik', outputs, output_weights) + output_bias

        # Set brain for agent
        self.brain = Brains.Brains(brain=outputs, parameters=self.vision.brain.parameters, placeholders=placeholders)

        # If not just representing memories, compute the loss
        if self.scope_name != "memory_representation":
            # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
            if "max_time_dim" in self.brain.parameters and len(self.brain.brain.shape.as_list()) > 2:
                # Mean squared difference
                mean_squared_difference = tf.reduce_mean(
                    tf.squared_difference(self.brain.brain, desired_outputs), axis=2)

                # Mask for canceling out padding in dynamic sequences
                mask = tf.sign(tf.reduce_max(tf.abs(desired_outputs), 2))

                # Explicit masking of loss (necessary in case a bias was added to the padding!)
                mean_squared_difference *= mask

                # Average loss over each (padded) sequence and batch
                self.loss = tf.reduce_mean(tf.reduce_sum(mean_squared_difference, 1) / tf.reduce_sum(mask, 1))
            else:
                # Mean squared error
                self.loss = tf.losses.mean_squared_error(desired_outputs, self.brain.brain)

            # If training
            if self.scope_name == "training":
                # Learning rate
                learning_rate = tf.placeholder(tf.float32, shape=[])
                self.brain.placeholders["learning_rate"] = learning_rate

                # Training optimization method
                optimizer = tf.train.AdamOptimizer(learning_rate)

                # If gradient clipping
                if "max_gradient_clip_norm" in self.brain.parameters:
                    # Trainable variables
                    self.variables = tf.trainable_variables()

                    # Get gradients of loss and clip them according to max gradient clip norm
                    self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.variables),
                                                               self.brain.parameters["max_gradient_clip_norm"])

                    # Training
                    self.train = optimizer.apply_gradients(zip(self.gradients, self.variables))
                else:
                    # Training
                    self.train = optimizer.minimize(self.loss)
