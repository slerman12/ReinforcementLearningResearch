from __future__ import division
import time
import numpy as np
import tensorflow as tf
from copy import deepcopy

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
        self.brain = self.loss = self.errors = self.train = self.accuracy = \
            self.learning_steps = self.increment_learning_steps = self.gradients = self.variables = \
            self.tensorflow_partial_run = self.tensorflow_saver = self.tensorboard_logging_interval = \
            self.tensorboard_writer = self.tensorboard_logs = None

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
            # Setup a partial run such that the graph is not recomputed later TODO arbitrary projections
            error_fetches = self.errors if isinstance(self.errors, list) else [self.errors]
            partial_run_fetches = [fetch for fetch in [self.train, self.loss, self.tensorboard_logs] + error_fetches
                                   if fetch is not None]
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

    def remember(self, query, memory_embedding=None, memory=None, batch_dims=None, max_time_dim=None, time_dims=None):
        # Start timing
        start_time = time.time()

        # Representation for memory
        if memory_embedding is None:
            if self.long_term_memory is not None:
                if self.long_term_memory.brain.brain is not None:
                    memory_embedding = self.long_term_memory.brain.brain

        # Memory
        if memory is None:
            memory = self.long_term_memory

        # If TensorFlow and a representation for memory can be made
        if self.tensorflow and memory_embedding is not None:
            # If no partial run already active
            if self.tensorflow_partial_run is None:
                # Setup a partial run such that the graph is not recomputed later TODO arbitrary projections
                error_fetches = self.errors if isinstance(self.errors, list) else [self.errors]
                partial_run_fetches = [fetch for fetch in [self.train, self.loss, self.tensorboard_logs] + error_fetches
                                       if fetch is not None]
                partial_run_setup = [partial_run_fetches, self.brain.placeholders] if len(partial_run_fetches) else None

                # Get embedding for memory
                if partial_run_setup is None:
                    query = self.brain.run(query, memory_embedding)
                else:
                    query, self.tensorflow_partial_run = self.brain.run(query, memory_embedding,
                                                                        partial_run_setup=partial_run_setup)
            else:
                # Get representation using active partial run
                query, _ = self.brain.run(query, memory_embedding, self.tensorflow_partial_run)

        # Initialize batch dims
        batch_dims = query.shape[0] if batch_dims is None else batch_dims

        # Initialize max time dim
        max_time_dim = query.shape[1] if max_time_dim is None else max_time_dim

        # Initialize time dims
        time_dims = np.full(batch_dims, max_time_dim) if time_dims is None else time_dims

        # Initialize memories
        remember_concepts = np.zeros([batch_dims, max_time_dim, self.k, self.attributes["concepts"]])
        remember_attributes = np.zeros([batch_dims, max_time_dim, self.k, self.attributes["attributes"]])

        # Initialize duplicate (negative means no duplicate)
        duplicate = []

        # Memory Module
        if memory is not None:
            # For each batch
            for batch_dim in range(batch_dims):
                # For each time
                for time_dim in range(time_dims[batch_dim]):
                    # Number of similar memories
                    num_similar_memories = min(memory.length, self.k)

                    # If there are memories to draw from
                    if num_similar_memories > 0:
                        # If not all zero
                        if query[batch_dim, time_dim].any():
                            # Similar memories TODO check if duplicate with distance == 0 TODO account for <k memory
                            distances, indices = memory.retrieve(query[batch_dim, time_dim], self.k)
                            remember_concepts[batch_dim, time_dim] = memory.memories["concepts"][indices]
                            temp = memory.memories["attributes"][indices]
                            if len(temp.shape) == 1:
                                remember_attributes[batch_dim, time_dim] = np.expand_dims(temp, 1)
                            else:
                                remember_attributes[batch_dim, time_dim] = temp

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
        errors = None

        # Consolidate memories
        if self.long_term_memory is not None:
            self.long_term_memory.consolidate(self.short_term_memory)

        # Train brain
        if self.train is not None and self.tensorflow:
            # Increment learning steps
            learning_steps = self.brain.run(components=[self.increment_learning_steps])[0]

            # Fetches
            fetches = {"train": self.train}

            # Errors and logs
            if self.errors is None:
                fetches.update({"errors": self.loss})
            else:
                fetches.update({"errors": self.errors})
            if self.tensorboard_logs is not None and learning_steps % self.tensorboard_logging_interval == 0:
                fetches.update({"logs": self.tensorboard_logs})

            # Fetched
            if self.tensorflow_partial_run is None:
                [fetched] = self.brain.run(placeholders, fetches)
            else:
                fetched, _ = self.brain.run(placeholders, fetches, self.tensorflow_partial_run)

            # Errors
            errors = fetched["errors"]

            # Output TensorBoard logs
            if self.tensorboard_logs is not None and learning_steps % self.tensorboard_logging_interval == 0:
                self.tensorboard_writer.add_summary(fetched["logs"], learning_steps)

        # Reset partial run
        self.tensorflow_partial_run = None

        # Measure time
        self.timer = time.time() - start_time

        # Return loss
        return errors

    def measure_errors(self, data):
        # Start timing
        start_time = time.time()

        # Learning steps
        learning_steps = self.brain.run(components=[self.learning_steps])[0]

        # Fetches
        fetches = {}

        # Errors and logs
        if self.errors is None:
            fetches.update({"errors": self.loss})
        else:
            fetches.update({"errors": self.errors})
        if self.tensorboard_logs is not None and learning_steps % self.tensorboard_logging_interval == 0:
            fetches.update({"logs": self.tensorboard_logs})

        # Fetched
        if self.tensorflow_partial_run is None:
            [fetched] = self.brain.run(data, fetches)
        else:
            fetched, _ = self.brain.run(data, fetches, self.tensorflow_partial_run)

        # Output TensorBoard logs
        if self.tensorboard_logs is not None and learning_steps % self.tensorboard_logging_interval == 0:
            self.tensorboard_writer.add_summary(fetched["logs"], learning_steps)

        # Reset partial run
        self.tensorflow_partial_run = None

        # Measure time
        self.timer = time.time() - start_time

        # Return loss and logs
        return fetched["errors"]

    def move_time_forward(self):
        # Size of discrete time unit
        time_quantum = 0.1

        # Move perception of time forward
        self.perception_of_time += time_quantum

    def start_tensorflow(self):
        # Open scope, start brains (build graphs), and begin session
        with tf.name_scope(self.scope_name):
            with tf.variable_scope("brain", reuse=None or self.scope_name != "training", initializer=None):
                # Start brain
                self.start_brain()

                # Learning steps counter
                self.learning_steps = tf.get_variable("learning_steps", [], tf.int32, tf.zeros_initializer(), False)

                # Training only operations
                if self.scope_name == "training":
                    # Learning steps incrementer
                    self.increment_learning_steps = tf.assign_add(self.learning_steps, 1)

                    # Saver
                    self.tensorflow_saver = tf.train.Saver()

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
        for body in [self, self.vision, self.long_term_memory]:
            if body is not None:
                if body.brain is not None:
                    body.brain.session = self.session

    def adapt(self, vision=None, long_term_memory=None, short_term_memory=None, traces=None, attributes=None,
              actions=None, exploration_rate=None, k=None, tensorflow=None, scope_name="adapted", session=None):
        # Bodies
        bodies = [self.vision, self.long_term_memory, self.short_term_memory, self.traces]

        # Genes
        genes = [self.attributes, self.actions, self.exploration_rate, self.k, self.tensorflow, self.session]

        # Mutations
        body_mutations = [vision, long_term_memory, short_term_memory, traces]
        gene_mutations = [attributes, actions, exploration_rate, k, tensorflow, session]

        # Default bodies
        for i, mutation in enumerate(body_mutations):
            if mutation is None and bodies[i] is not None:
                body_mutations[i] = bodies[i].adapt()

        # Default genes
        for i, mutation in enumerate(gene_mutations):
            if mutation is None:
                gene_mutations[i] = genes[i]

        # Return adapted agent
        return self.__class__(body_mutations[0], body_mutations[1], body_mutations[2], body_mutations[3],
                              gene_mutations[0], gene_mutations[1], gene_mutations[2], gene_mutations[3],
                              gene_mutations[4], scope_name, gene_mutations[5])

    def save(self, filename="Saved/brain"):
        # Save to file
        self.tensorflow_saver.save(self.session, filename)

    def load(self, filename="Saved/brain"):
        # max(glob.glob('Saved/*'), key=os.path.getctime)
        if tf.train.checkpoint_exists(filename):
            self.tensorflow_saver.restore(self.session, filename)

    def start_tensorboard(self, scalars=None, gradients=None, variables=None, tensorboard_writer=None,
                          logging_interval=1, directory_name="Logs"):
        # Defaults
        if scalars is None:
            scalars = {}

        # Logging interval
        self.tensorboard_logging_interval = logging_interval

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
        # Start vision
        self.vision.start_brain()

        # Start long term memory
        self.long_term_memory.start_brain(projection=self.vision.brain)

        # Parameters, placeholders, and components
        parameters = {}
        placeholders = {}
        components = {}

        # Parameters
        parameters.update(self.vision.brain.parameters)
        parameters.update({"input_dim": parameters["midstream_dim"], "output_dim": self.attributes["attributes"]})

        # Desired outputs
        desired_outputs = tf.placeholder("float", [None, None, self.attributes["attributes"]])

        # Retrieved memories
        remember_concepts = tf.placeholder("float", [None, None, self.k, self.attributes["concepts"]])
        remember_attributes = tf.placeholder("float", [None, None, self.k, self.attributes["attributes"]])

        # Placeholders
        placeholders.update(self.vision.brain.placeholders)
        placeholders.update({"remember_concepts": remember_concepts, "remember_attributes": remember_attributes,
                             "desired_outputs": desired_outputs})

        # Components
        components.update(self.vision.brain.components)
        components.update({"vision": self.vision.brain.brain,
                           "memory": self.long_term_memory.brain.brain})

        # To prevent division by 0 and to prevent NaN gradients from square root of 0
        distance_delta = 0.001

        # Distances (batch x time x k)
        distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(tf.tile(tf.expand_dims(components["memory"], axis=2),
                                                                        [1, 1, self.k, 1]), remember_concepts), axis=3)
                            + distance_delta ** 2)

        # Weights (batch x time x k x attributes)
        weights = tf.tile(tf.expand_dims(1.0 / distances, axis=3), [1, 1, 1, self.attributes["attributes"]])

        # Division numerator and denominator (for weighted means)
        numerator = tf.reduce_sum(weights * remember_attributes, axis=2)  # Weigh attributes
        denominator = tf.reduce_sum(weights, axis=2)  # Normalize weightings

        # In case denominator is zero
        safe_denominator = tf.where(tf.less(denominator, 1e-7), tf.ones_like(denominator), denominator)

        # Distance weighted memory attributes (batch x time x attributes)
        distance_weighted_memory_attributes = tf.divide(numerator, safe_denominator)

        # Default outputs (memory module's distance-weighted predictions)
        outputs = distance_weighted_memory_attributes

        if "downstream_weights" in parameters:
            if parameters["downstream_weights"]:
                # Add time ahead before final dense layer
                outputs = tf.concat(
                    [distance_weighted_memory_attributes, tf.expand_dims(placeholders["time_ahead"], 2)], 2) \
                    if parameters["time_ahead_downstream"] else distance_weighted_memory_attributes

                # Context vector residual
                if "raw_input_context_vector" in parameters:
                    if parameters["raw_input_context_vector"]:
                        outputs = tf.concat([outputs, placeholders["inputs"]], 2)
                if "visual_representation_context_vector" in parameters:
                    if parameters["visual_representation_context_vector"]:
                        outputs = tf.concat([outputs, components["vision"]], 2)
                if "dorsal_representation_context_vector" in parameters:
                    if parameters["dorsal_representation_context_vector"]:
                        # Default cell mode ("basic", "block", "cudnn") and number of layers
                        num_layers = parameters["num_layers"] if "num_layers" in parameters else 1

                        context_inputs = placeholders["inputs"]

                        # Dropout
                        if "dropout" in parameters:
                            context_inputs = tf.nn.dropout(context_inputs, keep_prob=1 - parameters["dropout"][3])

                        # Add time ahead before lstm layer
                        if "time_ahead_upstream" in parameters:
                            if parameters["time_ahead_upstream"]:
                                context_inputs = tf.concat(
                                    [context_inputs, tf.expand_dims(placeholders["time_ahead"], 2)], 2)

                        with tf.variable_scope('lstm2'):
                            # Dropout
                            lstm_layers = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                                tf.contrib.rnn.LSTMBlockCell(self.vision.brain.parameters["output_dim"], forget_bias=5),
                                output_keep_prob=
                                1 - parameters["dropout"][4 if layer + 1 < num_layers else 5]) for layer in
                                range(num_layers)])

                            # Outputs and states of lstm layers
                            context_inputs, final_states = tf.nn.dynamic_rnn(lstm_layers, context_inputs,
                                                                             placeholders["time_dims"],
                                                                             dtype=tf.float32)

                        # Mask for canceling out padding in dynamic sequences
                        context_inputs *= tf.tile(components["mask"],
                                                  [1, 1, self.vision.brain.parameters["output_dim"]])

                        # Concatenate
                        outputs = tf.concat([outputs, context_inputs], 2)

                # Update midstream dim
                parameters["downstream_dim"] = outputs.shape[2]

                # Final dense layer weights TODO get rid of second dense layer
                output_weights = tf.get_variable("output_weights_ya",
                                                 [parameters["downstream_dim"], 32])

                # Final dense layer bias
                output_bias = tf.get_variable("output_bias_ya", [32])

                # Dense layer
                outputs = tf.einsum('aij,jk->aik', outputs, output_weights) + output_bias

                # Dropout
                tf.nn.dropout(outputs, keep_prob=0.65)

                # Final dense layer weights
                output_weights = tf.get_variable("output_weights", [32, parameters["output_dim"]])

                # Final dense layer bias
                output_bias = tf.get_variable("output_bias", [parameters["output_dim"]])

                # Dense layer
                outputs = tf.einsum('aij,jk->aik', outputs, output_weights) + output_bias

        # Apply mask to outputs
        outputs *= tf.tile(components["mask"], [1, 1, parameters["output_dim"]])

        # Set brain for agent
        self.brain = Brains.Brains(brain=outputs, parameters=parameters, placeholders=placeholders)

        # If not just representing memories, compute the loss
        if self.scope_name != "memory_representation":
            # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
            if "max_time_dim" in self.brain.parameters and len(self.brain.brain.shape.as_list()) > 2:
                # Mask for canceling out padding in dynamic sequences
                mask = tf.squeeze(components["mask"])

                # Mean squared difference for brain output
                final_prediction_loss = tf.reduce_mean(tf.squared_difference(self.brain.brain, desired_outputs), 2)
                memory_prediction_loss = tf.reduce_mean(tf.squared_difference(distance_weighted_memory_attributes,
                                                                              desired_outputs), axis=2)

                # Mean squared difference of output (option to add to memory prediction loss)
                self.loss = final_prediction_loss

                # Explicit masking of loss (necessary in case a bias was added to the padding)
                self.loss *= mask

                # Loss function to optimize
                self.loss = tf.reduce_mean(tf.reduce_sum(self.loss, 1) / tf.reduce_sum(mask, 1))

                # Error(s) to record
                self.errors = [tf.reduce_mean(tf.reduce_sum(final_prediction_loss * mask, 1) / tf.reduce_sum(mask, 1)),
                               tf.reduce_mean(tf.reduce_sum(memory_prediction_loss * mask, 1) / tf.reduce_sum(mask, 1))]

            # else:
            #     # Mean squared error TODO add memory output
            #     self.loss = tf.losses.mean_squared_error(desired_outputs, self.brain.brain)

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
                    train = optimizer.apply_gradients(zip(self.gradients, self.variables))
                else:
                    # Training TODO add memory output
                    train = optimizer.minimize(self.loss)

                # Dummy train operation for partial run
                with tf.control_dependencies([train]):
                    self.train = tf.constant(0)


class MultiActionLifelongMemory(Agent):
    def start_brain(self):
        # Parameters (Careful, not deepcopy)
        parameters = {}
        parameters.update(self.vision.brain.parameters)
        parameters.update({"input_dim": parameters["midstream_dim"], "output_dim": self.attributes["attributes"]})

        # Desired outputs
        desired_outputs = tf.placeholder("float", [None, None, self.attributes["attributes"]])

        # Retrieved memories
        remember_concepts = tf.placeholder("float", [None, None, self.k, self.attributes["concepts"]])
        remember_attributes = tf.placeholder("float", [None, None, self.k, self.attributes["attributes"]])

        # Placeholders (Careful, not deepcopy)
        placeholders = {}
        placeholders.update(self.vision.brain.placeholders)
        placeholders.update({"remember_concepts": remember_concepts, "remember_attributes": remember_attributes,
                             "desired_outputs": desired_outputs})

        # Components (Careful, not deepcopy)
        components = {}
        components.update(self.vision.brain.components)
        components.update({"vision": self.vision.brain.brain,
                           "memory_embedding_weights": self.long_term_memory.brain.components["embedding_weights"],
                           "memory_embedding_bias": self.long_term_memory.brain.components["embedding_bias"]})

        # Embedding for memory
        self.memory_embedding = tf.einsum('aij,jk->aik', components["vision"], components["memory_embedding_weights"])
        self.memory_embedding += components["memory_embedding_bias"]

        # Give mask the right dimensionality
        mask = tf.tile(components["mask"], [1, 1, parameters["memory_embedding_dim"]])

        # Mask for canceling out padding in dynamic sequences
        self.memory_embedding *= mask

        # To prevent division by 0 and to prevent NaN gradients from square root of 0
        distance_delta = 0.001

        # Distances (batch x time x k)
        distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(tf.tile(tf.expand_dims(self.memory_embedding, axis=2),
                                                                        [1, 1, self.k, 1]), remember_concepts), axis=3)
                            + distance_delta ** 2)

        # Weights (batch x time x k x attributes)
        weights = tf.tile(tf.expand_dims(1.0 / distances, axis=3), [1, 1, 1, self.attributes["attributes"]])

        # Division numerator and denominator (for weighted means)
        numerator = tf.reduce_sum(weights * remember_attributes, axis=2)  # Weigh attributes
        denominator = tf.reduce_sum(weights, axis=2)  # Normalize weightings

        # In case denominator is zero
        safe_denominator = tf.where(tf.less(denominator, 1e-7), tf.ones_like(denominator), denominator)

        # Distance weighted memory attributes (batch x time x attributes)
        distance_weighted_memory_attributes = tf.divide(numerator, safe_denominator)

        action_suggestion_list = [(tf.einsum('aij,jk->aik', components["vision"],
                                             tf.get_variable("output_weights_{}".format(i),
                                                             [parameters["midstream_dim"],
                                                              parameters["output_dim"]]))
                                   + tf.get_variable("output_bias_{}".format(i), [parameters["output_dim"]]))
                                  * tf.tile(components["mask"], [1, 1, parameters["output_dim"]]) for i in
                                  range(parameters["num_action_suggestions"])]

        action_suggestions = tf.stack(action_suggestion_list)

        # Set brain for agent
        self.brain = Brains.Brains(brain=distance_weighted_memory_attributes, parameters=parameters,
                                   placeholders=placeholders)

        # If not just representing memories, compute the loss
        if self.scope_name != "memory_representation":
            # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
            if "max_time_dim" in self.brain.parameters and len(self.brain.brain.shape.as_list()) > 2:
                # Mask for canceling out padding in dynamic sequences
                mask = tf.squeeze(components["mask"])

                # Mean squared difference for brain output
                self.loss = tf.reduce_mean(tf.squared_difference(self.brain.brain, desired_outputs), 2) * mask

                # Loss function to optimize
                self.loss = tf.reduce_mean(tf.reduce_sum(self.loss, 1) / tf.reduce_sum(mask, 1))

                # Errors
                self.errors = [self.loss]

                # Record error for each suggested action
                for action in action_suggestion_list:
                    mse = tf.reduce_mean(tf.squared_difference(action, desired_outputs), 2) * mask
                    action_loss = tf.reduce_mean(tf.reduce_sum(mse, 1) / tf.reduce_sum(mask, 1))
                    self.errors += [action_loss]

                # Add loss for each suggested action
                action_mask = tf.tile(tf.expand_dims(mask, 0), [parameters["num_action_suggestions"], 1, 1])
                mse = tf.reduce_mean(tf.squared_difference(
                    action_suggestions, tf.tile(tf.expand_dims(desired_outputs, 0),
                                                [parameters["num_action_suggestions"], 1, 1, 1])), axis=3) * action_mask
                self.loss += tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(mse, 2) / tf.reduce_sum(action_mask, 2), 1))

                # Action difference from memory
                action_memory_difference = tf.reduce_sum(
                    tf.squared_difference(action_suggestions,
                                          tf.tile(tf.expand_dims(self.brain.brain, 0),
                                                  [parameters["num_action_suggestions"], 1, 1, 1])), 3)

                # Action indices
                batch_indices = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(parameters["batch_dim"], dtype=tf.int64),
                                                                      axis=1), [1, parameters["max_time_dim"]]), axis=2)
                time_indices = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(parameters["max_time_dim"]), axis=0),
                                                      [parameters["batch_dim"], 1]), axis=2)
                memory_informed_action_indices = tf.argmin(action_memory_difference, axis=0)
                memory_informed_action_indices = tf.expand_dims(memory_informed_action_indices, 2)
                memory_informed_action_indices = tf.concat([memory_informed_action_indices, batch_indices], 2)
                memory_informed_action_indices = tf.concat([memory_informed_action_indices, time_indices], 2)

                # Action taken
                action_taken = tf.gather_nd(action_suggestions, memory_informed_action_indices)

                # Error
                error = tf.reduce_mean(tf.squared_difference(action_taken, desired_outputs), 2) * mask
                error = tf.reduce_mean(tf.reduce_sum(error, 1) / tf.reduce_sum(mask, 1))

                # Push final error
                self.errors.insert(0, error)
                self.loss = 3 * error + self.errors[1]

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
                    train = optimizer.apply_gradients(zip(self.gradients, self.variables))
                else:
                    # Training TODO add memory output
                    train = optimizer.minimize(self.loss)

                # Dummy train operation for partial run
                with tf.control_dependencies([train]):
                    self.train = tf.constant(0)


class WeightedMultiActionMemory(Agent):
    def start_brain(self):
        # Parameters (Careful, not deepcopy)
        parameters = {}
        parameters.update(self.vision.brain.parameters)
        parameters.update({"input_dim": parameters["midstream_dim"], "output_dim": self.attributes["attributes"]})

        # Desired outputs
        desired_outputs = tf.placeholder("float", [None, None, self.attributes["attributes"]])

        # Retrieved memories
        remember_concepts = tf.placeholder("float", [None, None, self.k, self.attributes["concepts"]])
        remember_attributes = tf.placeholder("float", [None, None, self.k, self.attributes["attributes"]])

        # Placeholders (Careful, not deepcopy)
        placeholders = {}
        placeholders.update(self.vision.brain.placeholders)
        placeholders.update({"remember_concepts": remember_concepts, "remember_attributes": remember_attributes,
                             "desired_outputs": desired_outputs})

        # Components (Careful, not deepcopy)
        components = {}
        components.update(self.vision.brain.components)
        components.update({"vision": self.vision.brain.brain,
                           "memory_embedding_weights": self.long_term_memory.brain.components["embedding_weights"],
                           "memory_embedding_bias": self.long_term_memory.brain.components["embedding_bias"]})

        # Embedding for memory
        self.memory_embedding = tf.einsum('aij,jk->aik', components["vision"], components["memory_embedding_weights"])
        self.memory_embedding += components["memory_embedding_bias"]

        # Give mask the right dimensionality
        mask = tf.tile(components["mask"], [1, 1, parameters["memory_embedding_dim"]])

        # Mask for canceling out padding in dynamic sequences
        self.memory_embedding *= mask

        # To prevent division by 0 and to prevent NaN gradients from square root of 0
        distance_delta = 0.001

        # Distances (batch x time x k)
        distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(tf.tile(tf.expand_dims(self.memory_embedding, axis=2),
                                                                        [1, 1, self.k, 1]), remember_concepts), axis=3)
                            + distance_delta ** 2)

        # Weights (batch x time x k x attributes)
        weights = tf.tile(tf.expand_dims(1.0 / (distances + 1) ** 2, axis=3), [1, 1, 1, self.attributes["attributes"]])

        # Division numerator and denominator (for weighted means)
        numerator = tf.reduce_sum(weights * remember_attributes, axis=2)  # Weigh attributes
        denominator = tf.reduce_sum(weights, axis=2)  # Normalize weightings

        # In case denominator is zero
        safe_denominator = tf.where(tf.less(denominator, 1e-7), tf.ones_like(denominator), denominator)

        # Distance weighted memory attributes (batch x time x attributes)
        distance_weighted_memory_attributes = tf.divide(numerator, safe_denominator)

        action_suggestions = [(tf.einsum('aij,jk->aik', components["vision"],
                                         tf.get_variable("output_weights_{}".format(i), [parameters["midstream_dim"],
                                                                                         parameters["output_dim"]]))
                               + tf.get_variable("output_bias_{}".format(i), [parameters["output_dim"]]))
                              * tf.tile(components["mask"], [1, 1, parameters["output_dim"]]) for i in
                              range(parameters["num_action_suggestions"])]

        # Distances (a x batch x time)
        distances = tf.sqrt(tf.reduce_sum(
            tf.squared_difference(tf.tile(tf.expand_dims(distance_weighted_memory_attributes, axis=0),
                                          [parameters["num_action_suggestions"], 1, 1, 1]), action_suggestions), axis=3)
                            + distance_delta ** 2)

        # Weights (a x batch x time x attributes)
        weights = tf.tile(tf.expand_dims(1.0 / (distances + 1) ** 2, axis=3), [1, 1, 1, self.attributes["attributes"]])

        # Division numerator and denominator (for weighted means)
        numerator = tf.reduce_sum(weights * action_suggestions, axis=0)  # Weigh attributes
        denominator = tf.reduce_sum(weights, axis=0)  # Normalize weightings

        # In case denominator is zero
        safe_denominator = tf.where(tf.less(denominator, 1e-7), tf.ones_like(denominator), denominator)

        # Distance weighted memory attributes (batch x time x attributes)
        memory_distance_weighted_predictions = tf.divide(numerator, safe_denominator)

        # Set brain for agent
        self.brain = Brains.Brains(brain=memory_distance_weighted_predictions, parameters=parameters,
                                   placeholders=placeholders)

        # If not just representing memories, compute the loss
        if self.scope_name != "memory_representation":
            # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
            if "max_time_dim" in self.brain.parameters and len(self.brain.brain.shape.as_list()) > 2:
                # Mean squared difference for brain output
                error = tf.reduce_mean(tf.squared_difference(self.brain.brain, desired_outputs), 2)

                # Mask for canceling out padding in dynamic sequences
                mask = tf.squeeze(components["mask"])

                # Explicit masking of loss (necessary in case a bias was added to the padding)
                error *= mask

                # Loss function to optimize
                error = tf.reduce_mean(tf.reduce_sum(error, 1) / tf.reduce_sum(mask, 1))

                memory_error = tf.reduce_mean(tf.squared_difference(distance_weighted_memory_attributes,
                                                                    desired_outputs), 2) * mask
                memory_error = tf.reduce_mean(tf.reduce_sum(memory_error, 1) / tf.reduce_sum(mask, 1))

                self.errors = [error, memory_error]
                self.loss = error + memory_error

                # Add loss for each suggested action
                for action in action_suggestions:
                    mse = tf.reduce_mean(tf.squared_difference(action, desired_outputs), 2) * mask
                    action_loss = tf.reduce_mean(tf.reduce_sum(mse, 1) / tf.reduce_sum(mask, 1))
                    self.errors += [action_loss]
                    self.loss += action_loss

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
                    train = optimizer.apply_gradients(zip(self.gradients, self.variables))
                else:
                    # Training TODO add memory output
                    train = optimizer.minimize(self.loss)

                # Dummy train operation for partial run
                with tf.control_dependencies([train]):
                    self.train = tf.constant(0)


class VariedMultiActionMemory(Agent):
    def start_brain(self):
        # Parameters (Careful, not deepcopy)
        parameters = {}
        parameters.update(self.vision.brain.parameters)
        parameters.update({"input_dim": parameters["midstream_dim"], "output_dim": self.attributes["attributes"]})

        # Desired outputs
        desired_outputs = tf.placeholder("float", [None, None, self.attributes["attributes"]])

        # Retrieved memories
        remember_concepts = tf.placeholder("float", [None, None, self.k, self.attributes["concepts"]])
        remember_attributes = tf.placeholder("float", [None, None, self.k, self.attributes["attributes"]])

        # Placeholders (Careful, not deepcopy)
        placeholders = {}
        placeholders.update(self.vision.brain.placeholders)
        placeholders.update({"remember_concepts": remember_concepts, "remember_attributes": remember_attributes,
                             "desired_outputs": desired_outputs})

        # Components (Careful, not deepcopy)
        components = {}
        components.update(self.vision.brain.components)
        components.update({"vision": self.vision.brain.brain,
                           "memory_embedding_weights": self.long_term_memory.brain.components["embedding_weights"],
                           "memory_embedding_bias": self.long_term_memory.brain.components["embedding_bias"]})

        # Embedding for memory
        self.memory_embedding = tf.einsum('aij,jk->aik', components["vision"], components["memory_embedding_weights"])
        self.memory_embedding += components["memory_embedding_bias"]

        # Give mask the right dimensionality
        mask = tf.tile(components["mask"], [1, 1, parameters["memory_embedding_dim"]])

        # Mask for canceling out padding in dynamic sequences
        self.memory_embedding *= mask

        # To prevent division by 0 and to prevent NaN gradients from square root of 0
        distance_delta = 0.001

        # Distances (batch x time x k)
        distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(tf.tile(tf.expand_dims(self.memory_embedding, axis=2),
                                                                        [1, 1, self.k, 1]), remember_concepts), axis=3)
                            + distance_delta ** 2)

        # Weights (batch x time x k x attributes)
        weights = tf.tile(tf.expand_dims(1.0 / distances, axis=3), [1, 1, 1, self.attributes["attributes"]])

        # Division numerator and denominator (for weighted means)
        numerator = tf.reduce_sum(weights * remember_attributes, axis=2)  # Weigh attributes
        denominator = tf.reduce_sum(weights, axis=2)  # Normalize weightings

        # In case denominator is zero
        safe_denominator = tf.where(tf.less(denominator, 1e-7), tf.ones_like(denominator), denominator)

        # Distance weighted memory attributes (batch x time x attributes)
        distance_weighted_memory_attributes = tf.divide(numerator, safe_denominator)

        action_suggestions = [(tf.einsum('aij,jk->aik', components["vision"],
                                         tf.get_variable("output_weights_{}".format(i), [parameters["midstream_dim"],
                                                                                         parameters["output_dim"]]))
                               + tf.get_variable("output_bias_{}".format(i), [parameters["output_dim"]]))
                              * tf.tile(components["mask"], [1, 1, parameters["output_dim"]]) for i in
                              range(parameters["num_action_suggestions"])]

        # Set brain for agent
        self.brain = Brains.Brains(brain=distance_weighted_memory_attributes, parameters=parameters,
                                   placeholders=placeholders)

        # If not just representing memories, compute the loss
        if self.scope_name != "memory_representation":
            # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
            if "max_time_dim" in self.brain.parameters and len(self.brain.brain.shape.as_list()) > 2:
                # Mean squared difference for brain output
                self.loss = tf.reduce_mean(tf.squared_difference(self.brain.brain, desired_outputs), 2)

                # Mask for canceling out padding in dynamic sequences
                mask = tf.squeeze(components["mask"])

                # Explicit masking of loss (necessary in case a bias was added to the padding)
                self.loss *= mask

                # Loss function to optimize
                self.loss = tf.reduce_mean(tf.reduce_sum(self.loss, 1) / tf.reduce_sum(mask, 1))

                self.errors = [self.loss]

                # Add loss for each suggested action
                for action in action_suggestions:
                    mse = tf.reduce_mean(tf.squared_difference(action, desired_outputs), 2) * mask
                    action_loss = tf.reduce_mean(tf.reduce_sum(mse, 1) / tf.reduce_sum(mask, 1))
                    self.loss += action_loss
                    self.errors += [action_loss]

                mean_action = tf.reduce_mean(action_suggestions, axis=0)
                # TODO mask, axis
                # avg_dist_from_mean = tf.reduce_mean(
                #     tf.reduce_sum(tf.squared_difference(tf.convert_to_tensor(action_suggestions),
                #                                         tf.tile(mean_action, parameters["num_action_suggestions"])), 3))

                sum_squared_diff_from_mean = tf.reduce_sum(
                    tf.squared_difference(tf.convert_to_tensor(action_suggestions),
                                          tf.tile(tf.expand_dims(mean_action, 0), [parameters["num_action_suggestions"],
                                                                                   1, 1, 1])))

                self.loss *= 100000
                self.loss -= (sum_squared_diff_from_mean + distance_delta)

                self.errors += [sum_squared_diff_from_mean]

                min_distance = 999999999999.
                action_taken = distance_weighted_memory_attributes
                for action in action_suggestions:
                    distance = tf.norm(action - self.brain.brain)
                    min_distance, action_taken = tf.cond(distance < min_distance,
                                                         lambda: (distance, action),
                                                         lambda: (min_distance, action_taken))

                error = tf.reduce_mean(tf.squared_difference(action_taken, desired_outputs), 2) * mask
                error = tf.reduce_mean(tf.reduce_sum(error, 1) / tf.reduce_sum(mask, 1))

                # Push final error
                self.errors.insert(0, error)

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
                    train = optimizer.apply_gradients(zip(self.gradients, self.variables))
                else:
                    # Training TODO add memory output
                    train = optimizer.minimize(self.loss)

                # Dummy train operation for partial run
                with tf.control_dependencies([train]):
                    self.train = tf.constant(0)


class VariedByTakenMultiActionMemory(Agent):
    def start_brain(self):
        # Parameters (Careful, not deepcopy)
        parameters = {}
        parameters.update(self.vision.brain.parameters)
        parameters.update({"input_dim": parameters["midstream_dim"], "output_dim": self.attributes["attributes"]})

        # Desired outputs
        desired_outputs = tf.placeholder("float", [None, None, self.attributes["attributes"]])

        # Retrieved memories
        remember_concepts = tf.placeholder("float", [None, None, self.k, self.attributes["concepts"]])
        remember_attributes = tf.placeholder("float", [None, None, self.k, self.attributes["attributes"]])

        # Placeholders (Careful, not deepcopy)
        placeholders = {}
        placeholders.update(self.vision.brain.placeholders)
        placeholders.update({"remember_concepts": remember_concepts, "remember_attributes": remember_attributes,
                             "desired_outputs": desired_outputs})

        # Components (Careful, not deepcopy)
        components = {}
        components.update(self.vision.brain.components)
        components.update({"vision": self.vision.brain.brain,
                           "memory_embedding_weights": self.long_term_memory.brain.components["embedding_weights"],
                           "memory_embedding_bias": self.long_term_memory.brain.components["embedding_bias"]})

        # Embedding for memory
        self.memory_embedding = tf.einsum('aij,jk->aik', components["vision"], components["memory_embedding_weights"])
        self.memory_embedding += components["memory_embedding_bias"]

        # Give mask the right dimensionality
        mask = tf.tile(components["mask"], [1, 1, parameters["memory_embedding_dim"]])

        # Mask for canceling out padding in dynamic sequences
        self.memory_embedding *= mask

        # To prevent division by 0 and to prevent NaN gradients from square root of 0
        distance_delta = 0.001

        # Distances (batch x time x k)
        distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(tf.tile(tf.expand_dims(self.memory_embedding, axis=2),
                                                                        [1, 1, self.k, 1]), remember_concepts), axis=3)
                            + distance_delta ** 2)

        # Weights (batch x time x k x attributes)
        weights = tf.tile(tf.expand_dims(1.0 / distances, axis=3), [1, 1, 1, self.attributes["attributes"]])

        # Division numerator and denominator (for weighted means)
        numerator = tf.reduce_sum(weights * remember_attributes, axis=2)  # Weigh attributes
        denominator = tf.reduce_sum(weights, axis=2)  # Normalize weightings

        # In case denominator is zero
        safe_denominator = tf.where(tf.less(denominator, 1e-7), tf.ones_like(denominator), denominator)

        # Distance weighted memory attributes (batch x time x attributes)
        distance_weighted_memory_attributes = tf.divide(numerator, safe_denominator)

        action_suggestion_list = [(tf.einsum('aij,jk->aik', components["vision"],
                                             tf.get_variable("output_weights_{}".format(i),
                                                             [parameters["midstream_dim"],
                                                              parameters["output_dim"]]))
                                   + tf.get_variable("output_bias_{}".format(i), [parameters["output_dim"]]))
                                  * tf.tile(components["mask"], [1, 1, parameters["output_dim"]]) for i in
                                  range(parameters["num_action_suggestions"])]
        action_suggestions = tf.stack(action_suggestion_list)

        # Set brain for agent
        self.brain = Brains.Brains(brain=distance_weighted_memory_attributes, parameters=parameters,
                                   placeholders=placeholders)

        # If not just representing memories, compute the loss
        if self.scope_name != "memory_representation":
            # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
            if "max_time_dim" in self.brain.parameters and len(self.brain.brain.shape.as_list()) > 2:
                # Mask for canceling out padding in dynamic sequences
                mask = tf.squeeze(components["mask"])

                # Mean squared difference for brain output
                memory_loss = tf.reduce_mean(tf.squared_difference(self.brain.brain, desired_outputs), 2)

                # Loss function to optimize
                self.loss = tf.reduce_mean(tf.reduce_sum(memory_loss * mask, 1) / tf.reduce_sum(mask, 1))
                self.errors = [self.loss]

                action_memory_difference = tf.reduce_sum(
                    tf.squared_difference(action_suggestions,
                                          tf.tile(tf.expand_dims(self.brain.brain, 0),
                                                  [parameters["num_action_suggestions"], 1, 1, 1])), 3)

                memory_informed_action_indices = tf.argmin(action_memory_difference, axis=0)

                batch_indices = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(parameters["batch_dim"], dtype=tf.int64),
                                                                      axis=1), [1, parameters["max_time_dim"]]), axis=2)
                time_indices = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(parameters["max_time_dim"]), axis=0),
                                                      [parameters["batch_dim"], 1]), axis=2)

                memory_informed_action_indices = tf.expand_dims(memory_informed_action_indices, 2)
                memory_informed_action_indices = tf.concat([memory_informed_action_indices, batch_indices], 2)
                memory_informed_action_indices = tf.concat([memory_informed_action_indices, time_indices], 2)

                action_taken = tf.gather_nd(action_suggestions, memory_informed_action_indices)

                # Add loss for each suggested action
                for action in action_suggestion_list:
                    mse = tf.reduce_mean(tf.squared_difference(action, desired_outputs), 2) * mask
                    action_loss = tf.reduce_mean(tf.reduce_sum(mse, 1) / tf.reduce_sum(mask, 1))
                    action_difference_from_taken = tf.reduce_sum(tf.squared_difference(action, action_taken), 2)
                    action_difference_from_taken_loss = tf.reduce_mean(
                        tf.reduce_sum(action_difference_from_taken, 1) / tf.reduce_sum(mask, 1))
                    self.loss += action_loss / (action_difference_from_taken_loss + 0.001)
                    self.errors += [action_loss]

                error = tf.reduce_mean(tf.squared_difference(action_taken, desired_outputs), 2) * mask
                error = tf.reduce_mean(tf.reduce_sum(error, 1) / tf.reduce_sum(mask, 1))

                # Push final error
                self.errors.insert(0, error)

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
                    train = optimizer.apply_gradients(zip(self.gradients, self.variables))
                else:
                    # Training TODO add memory output
                    train = optimizer.minimize(self.loss)

                # Dummy train operation for partial run
                with tf.control_dependencies([train]):
                    self.train = tf.constant(0)


class SigmoidVariedMultiActionMemory(Agent):
    def start_brain(self):
        # Parameters (Careful, not deepcopy)
        parameters = {}
        parameters.update(self.vision.brain.parameters)
        parameters.update({"input_dim": parameters["midstream_dim"], "output_dim": self.attributes["attributes"]})

        # Desired outputs
        desired_outputs = tf.placeholder("float", [None, None, self.attributes["attributes"]])

        # Retrieved memories
        remember_concepts = tf.placeholder("float", [None, None, self.k, self.attributes["concepts"]])
        remember_attributes = tf.placeholder("float", [None, None, self.k, self.attributes["attributes"]])

        # Placeholders (Careful, not deepcopy)
        placeholders = {}
        placeholders.update(self.vision.brain.placeholders)
        placeholders.update({"remember_concepts": remember_concepts, "remember_attributes": remember_attributes,
                             "desired_outputs": desired_outputs})

        # Components (Careful, not deepcopy)
        components = {}
        components.update(self.vision.brain.components)
        components.update({"vision": self.vision.brain.brain,
                           "memory_embedding_weights": self.long_term_memory.brain.components["embedding_weights"],
                           "memory_embedding_bias": self.long_term_memory.brain.components["embedding_bias"]})

        # Embedding for memory
        self.memory_embedding = tf.einsum('aij,jk->aik', components["vision"], components["memory_embedding_weights"])
        self.memory_embedding += components["memory_embedding_bias"]

        # Give mask the right dimensionality
        mask = tf.tile(components["mask"], [1, 1, parameters["memory_embedding_dim"]])

        # Mask for canceling out padding in dynamic sequences
        self.memory_embedding *= mask

        # To prevent division by 0 and to prevent NaN gradients from square root of 0
        distance_delta = 0.001

        # Distances (batch x time x k)
        distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(tf.tile(tf.expand_dims(self.memory_embedding, axis=2),
                                                                        [1, 1, self.k, 1]), remember_concepts), axis=3)
                            + distance_delta ** 2)

        # Weights (batch x time x k x attributes)
        weights = tf.tile(tf.expand_dims(1.0 / distances, axis=3), [1, 1, 1, self.attributes["attributes"]])

        # Division numerator and denominator (for weighted means)
        numerator = tf.reduce_sum(weights * remember_attributes, axis=2)  # Weigh attributes
        denominator = tf.reduce_sum(weights, axis=2)  # Normalize weightings

        # In case denominator is zero
        safe_denominator = tf.where(tf.less(denominator, 1e-7), tf.ones_like(denominator), denominator)

        # Distance weighted memory attributes (batch x time x attributes)
        distance_weighted_memory_attributes = tf.divide(numerator, safe_denominator)

        action_suggestions = [(tf.einsum('aij,jk->aik', components["vision"],
                                         tf.get_variable("output_weights_{}".format(i), [parameters["midstream_dim"],
                                                                                         parameters["output_dim"]]))
                               + tf.get_variable("output_bias_{}".format(i), [parameters["output_dim"]]))
                              * tf.tile(components["mask"], [1, 1, parameters["output_dim"]]) for i in
                              range(parameters["num_action_suggestions"])]

        # Set brain for agent
        self.brain = Brains.Brains(brain=distance_weighted_memory_attributes, parameters=parameters,
                                   placeholders=placeholders)

        # If not just representing memories, compute the loss
        if self.scope_name != "memory_representation":
            # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
            if "max_time_dim" in self.brain.parameters and len(self.brain.brain.shape.as_list()) > 2:
                # Mean squared difference for brain output
                self.loss = tf.reduce_mean(tf.squared_difference(self.brain.brain, desired_outputs), 2)

                # Mask for canceling out padding in dynamic sequences
                mask = tf.squeeze(components["mask"])

                # Explicit masking of loss (necessary in case a bias was added to the padding)
                self.loss *= mask

                # Loss function to optimize
                self.loss = tf.reduce_mean(tf.reduce_sum(self.loss, 1) / tf.reduce_sum(mask, 1))

                self.errors = [self.loss]

                # Add loss for each suggested action
                for action in action_suggestions:
                    mse = tf.reduce_mean(tf.squared_difference(action, desired_outputs), 2) * mask
                    action_loss = tf.reduce_mean(tf.reduce_sum(mse, 1) / tf.reduce_sum(mask, 1))
                    self.loss += action_loss
                    self.errors += [action_loss]

                mean_action = tf.reduce_mean(action_suggestions, axis=0)
                # TODO mask, axis
                # avg_dist_from_mean = tf.reduce_mean(
                #     tf.reduce_sum(tf.squared_difference(tf.convert_to_tensor(action_suggestions),
                #                                         tf.tile(mean_action, parameters["num_action_suggestions"])), 3))

                sum_squared_diff_from_mean = tf.reduce_sum(
                    tf.squared_difference(tf.convert_to_tensor(action_suggestions),
                                          tf.tile(tf.expand_dims(mean_action, 0), [parameters["num_action_suggestions"],
                                                                                   1, 1, 1])))

                self.loss *= tf.sigmoid(1 / sum_squared_diff_from_mean)

                self.errors += [tf.sigmoid(1 / sum_squared_diff_from_mean)]

                min_distance = 999999999999.
                action_taken = distance_weighted_memory_attributes
                for action in action_suggestions:
                    distance = tf.norm(action - self.brain.brain)
                    min_distance, action_taken = tf.cond(distance < min_distance,
                                                         lambda: (distance, action),
                                                         lambda: (min_distance, action_taken))

                error = tf.reduce_mean(tf.squared_difference(action_taken, desired_outputs), 2) * mask
                error = tf.reduce_mean(tf.reduce_sum(error, 1) / tf.reduce_sum(mask, 1))

                # Push final error
                self.errors.insert(0, error)

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
                    train = optimizer.apply_gradients(zip(self.gradients, self.variables))
                else:
                    # Training TODO add memory output
                    train = optimizer.minimize(self.loss)

                # Dummy train operation for partial run
                with tf.control_dependencies([train]):
                    self.train = tf.constant(0)


class ProbabalisticVariedByTakenMultiActionMemory(Agent):
    def start_brain(self):
        # Parameters (Careful, not deepcopy)
        parameters = {}
        parameters.update(self.vision.brain.parameters)
        parameters.update({"input_dim": parameters["midstream_dim"], "output_dim": self.attributes["attributes"]})

        # Desired outputs
        desired_outputs = tf.placeholder("float", [None, None, self.attributes["attributes"]])

        # Retrieved memories
        remember_concepts = tf.placeholder("float", [None, None, self.k, self.attributes["concepts"]])
        remember_attributes = tf.placeholder("float", [None, None, self.k, self.attributes["attributes"]])

        # Placeholders (Careful, not deepcopy)
        placeholders = {}
        placeholders.update(self.vision.brain.placeholders)
        placeholders.update({"remember_concepts": remember_concepts, "remember_attributes": remember_attributes,
                             "desired_outputs": desired_outputs})

        # Components (Careful, not deepcopy)
        components = {}
        components.update(self.vision.brain.components)
        components.update({"vision": self.vision.brain.brain,
                           "memory_embedding_weights": self.long_term_memory.brain.components["embedding_weights"],
                           "memory_embedding_bias": self.long_term_memory.brain.components["embedding_bias"]})

        # Embedding for memory
        self.memory_embedding = tf.einsum('aij,jk->aik', components["vision"], components["memory_embedding_weights"])
        self.memory_embedding += components["memory_embedding_bias"]

        # Give mask the right dimensionality
        mask = tf.tile(components["mask"], [1, 1, parameters["memory_embedding_dim"]])

        # Mask for canceling out padding in dynamic sequences
        self.memory_embedding *= mask

        # To prevent division by 0 and to prevent NaN gradients from square root of 0
        distance_delta = 0.001

        # Distances (batch x time x k)
        distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(tf.tile(tf.expand_dims(self.memory_embedding, axis=2),
                                                                        [1, 1, self.k, 1]), remember_concepts), axis=3)
                            + distance_delta ** 2)

        # Weights (batch x time x k x attributes)
        weights = tf.tile(tf.expand_dims(1.0 / distances, axis=3), [1, 1, 1, self.attributes["attributes"]])

        # Division numerator and denominator (for weighted means)
        numerator = tf.reduce_sum(weights * remember_attributes, axis=2)  # Weigh attributes
        denominator = tf.reduce_sum(weights, axis=2)  # Normalize weightings

        # In case denominator is zero
        safe_denominator = tf.where(tf.less(denominator, 1e-7), tf.ones_like(denominator), denominator)

        # Distance weighted memory attributes (batch x time x attributes)
        distance_weighted_memory_attributes = tf.divide(numerator, safe_denominator)

        action_suggestion_list = [(tf.einsum('aij,jk->aik', components["vision"],
                                             tf.get_variable("output_weights_{}".format(i),
                                                             [parameters["midstream_dim"],
                                                              parameters["output_dim"]]))
                                   + tf.get_variable("output_bias_{}".format(i), [parameters["output_dim"]]))
                                  * tf.tile(components["mask"], [1, 1, parameters["output_dim"]]) for i in
                                  range(parameters["num_action_suggestions"])]
        action_suggestions = tf.stack(action_suggestion_list)

        # Set brain for agent
        self.brain = Brains.Brains(brain=distance_weighted_memory_attributes, parameters=parameters,
                                   placeholders=placeholders)

        # If not just representing memories, compute the loss
        if self.scope_name != "memory_representation":
            # If outputs are sequences with dynamic numbers of time dimensions (this is an ugly way to check that)
            if "max_time_dim" in self.brain.parameters and len(self.brain.brain.shape.as_list()) > 2:
                # Mask for canceling out padding in dynamic sequences
                mask = tf.squeeze(components["mask"])

                # Mean squared difference for brain output
                memory_loss = tf.reduce_mean(tf.squared_difference(self.brain.brain, desired_outputs), 2)

                # Loss function to optimize
                self.loss = tf.reduce_mean(tf.reduce_sum(memory_loss * mask, 1) / tf.reduce_sum(mask, 1))
                self.errors = [self.loss]

                action_memory_difference = tf.reduce_sum(
                    tf.squared_difference(action_suggestions,
                                          tf.tile(tf.expand_dims(self.brain.brain, 0),
                                                  [parameters["num_action_suggestions"], 1, 1, 1])), 3)

                memory_informed_action_indices = tf.argmin(action_memory_difference, axis=0)

                batch_indices = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(parameters["batch_dim"], dtype=tf.int64),
                                                                      axis=1), [1, parameters["max_time_dim"]]), axis=2)
                time_indices = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(parameters["max_time_dim"]), axis=0),
                                                      [parameters["batch_dim"], 1]), axis=2)

                memory_informed_action_indices = tf.expand_dims(memory_informed_action_indices, 2)
                memory_informed_action_indices = tf.concat([memory_informed_action_indices, batch_indices], 2)
                memory_informed_action_indices = tf.concat([memory_informed_action_indices, time_indices], 2)

                action_taken = tf.gather_nd(action_suggestions, memory_informed_action_indices)

                # Add loss for each suggested action
                for action in action_suggestions:
                    mse = tf.reduce_mean(tf.squared_difference(action, desired_outputs), 2) * mask
                    action_loss = tf.reduce_mean(tf.reduce_sum(mse, 1) / tf.reduce_sum(mask, 1))
                    action_difference_from_taken = tf.reduce_sum(tf.squared_difference(action, action_taken), 2)
                    action_difference_from_taken_loss = tf.reduce_mean(
                        tf.reduce_sum(action_difference_from_taken, 1) / tf.reduce_sum(mask, 1))
                    self.loss += action_loss / (action_difference_from_taken_loss + 0.001)
                    self.errors += [action_loss]

                error = tf.reduce_mean(tf.squared_difference(action_taken, desired_outputs), 2) * mask
                error = tf.reduce_mean(tf.reduce_sum(error, 1) / tf.reduce_sum(mask, 1))

                # Push final error
                self.errors.insert(0, error)

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
                    train = optimizer.apply_gradients(zip(self.gradients, self.variables))
                else:
                    # Training TODO add memory output
                    train = optimizer.minimize(self.loss)

                # Dummy train operation for partial run
                with tf.control_dependencies([train]):
                    self.train = tf.constant(0)
