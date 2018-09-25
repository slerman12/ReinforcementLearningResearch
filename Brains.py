import tensorflow as tf
from copy import deepcopy, copy


class Brains:
    def __init__(self, stream_methods=None, projections=None, parameters=None, name_scope=None, tensorflow=True,
                 session=None):
        # Stream
        self.stream = {}

        # Final stream projection
        self.stream_projection = None

        # Projection name
        self.stream_projection_name = None

        # All projections in stream
        self.projections = {}

        # Start brain with add_to_stream
        self.add_to_stream(stream_methods, projections, parameters, name_scope)

        # If using TensorFlow
        self.tensorflow = tensorflow

        # TensorFlow partial run
        self.tensorflow_partial_run = None

        # Initialize session
        self.session = session

    def run(self, projections=None, out_projections=None, do_partial_run=False):
        # If TensorFlow
        if self.tensorflow:
            # Default projections
            if not isinstance(projections, dict):
                projections = {} if projections is None else {"inputs": projections}

            # Projections to use
            if do_partial_run and self.tensorflow_partial_run is not None:
                projections = {key: projections[key] for key in projections
                               if key in self.tensorflow_partial_run["feeds"]}

            # Projections to use
            projections = {self.build(key): projections[key] for key in projections if key in
                           self.projections and self.build(key, update_stream=True) is not None}

            # Last partial run
            last_partial_run = False
            if self.tensorflow_partial_run is not None:
                if self.tensorflow_partial_run["last_fetch"] is not None:
                    if isinstance(out_projections, dict) or isinstance(out_projections, list):
                        if not isinstance(self.tensorflow_partial_run["last_fetch"], list):
                            last_partial_run = self.tensorflow_partial_run["last_fetch"] in out_projections
                    else:
                        last_partial_run = self.tensorflow_partial_run["last_fetch"] == out_projections

            # Stream projections to run
            if isinstance(out_projections, dict):
                out_projections = {key: out_projections[key] for key in out_projections
                                   if out_projections[key] is not None}
            if out_projections is None:
                out_projections = [self.build(self.stream_projection_name)]
            elif isinstance(out_projections, str):
                out_projections = [self.projections[out_projections]]
            elif isinstance(out_projections, list):
                for i, component in enumerate(out_projections):
                    if isinstance(component, str):
                        out_projections[i] = self.projections[component]
            else:
                out_projections = [out_projections]
            out_projections = [projection for projection in out_projections if projection is not None]

            # Output single element if components is a single item list
            if isinstance(out_projections, list):
                if len(out_projections) == 1:
                    out_projections = out_projections[0]

            # If doing a partial run
            if do_partial_run and self.tensorflow_partial_run is not None:
                # If partial run still needs to be set up
                if self.tensorflow_partial_run["partial_run"] is None:

                    # Check if fetches None
                    self.tensorflow_partial_run["fetches"] = [fetch for fetch in self.tensorflow_partial_run["fetches"]
                                                              if fetch is not None]

                    # Map strings in fetches to stream projections
                    for i, key in enumerate(self.tensorflow_partial_run["fetches"]):
                        if isinstance(key, str):
                            self.tensorflow_partial_run["fetches"][i] = self.projections[key]

                    # Fetches
                    fetches = self.tensorflow_partial_run["fetches"]

                    # Add special fetches
                    if self.tensorflow_partial_run["special_fetches"] is not None:
                        if self.tensorflow_partial_run["special_fetches"][1]():
                            fetches = fetches + self.tensorflow_partial_run["special_fetches"][0]

                    # Set up partial run
                    self.tensorflow_partial_run["partial_run"] = self.session.partial_run_setup(
                        fetches, list(self.tensorflow_partial_run["feeds"].values()))

                # Return the result for the partially computed graph
                result = self.session.partial_run(self.tensorflow_partial_run["partial_run"], out_projections,
                                                  projections)

                # If final stream projection of partial run, reset
                if last_partial_run:
                    self.tensorflow_partial_run["partial_run"] = None

                # Return
                return result
            else:
                # Return the regular run
                return self.session.run(out_projections, feed_dict=projections)

    def build(self, out_projection=None, projections=None, name_scope=None, update_stream=False):
        # Stream projection to build
        if out_projection is None:
            out_projection = self.stream_projection_name
        elif callable(out_projection):
            out_projection = out_projection.__name__

        # Add projection
        if out_projection not in self.projections:
            self.projections[out_projection] = None

        # Name scope
        with optional_name_scope(name_scope):
            # If building a stream method whose projections might need to be replaced TODO: ideally, track dependencies
            if out_projection in self.stream and projections:
                # Projections to pass into the stream method
                stream_method_projections = {}

                # Convert unique projection name to list
                if not isinstance(projections, list) and not isinstance(projections, dict):
                    projections = [projections]

                # Convert list of projection names to dict
                if isinstance(projections, list):
                    projections_dict = {}
                    for projection in projections:
                        if callable(projection):
                            projection = projection.__name__
                        projections_dict[projection] = projection
                    projections = projections_dict

                # Run upstream using passed in projections
                for projection in self.stream[out_projection]["projections"]:
                    # Use passed in projections
                    if projection in projections:
                        # Update stream
                        if update_stream:
                            self.projections[projection] = projections[projection]
                        elif projection not in self.projections:
                            # Add to projections
                            self.projections[projection] = None

                        # Convert string to projection
                        if isinstance(projections[projection], str):
                            projections[projection] = self.build(projections[projection], projections,
                                                                 update_stream=update_stream)

                        # Replace
                        stream_method_projections[projection] = projections[projection]

                        # Assumes unique and prunes; faster, but comment out if methods reuse the same projection
                        # del projections[projection]
                    else:
                        stream_method_projections[projection] = self.build(projection, projections,
                                                                           update_stream=update_stream)

                # Stream method projection (Note: does scoping work with recursion?)
                with tf.variable_scope(self.stream[out_projection]["variable_scope"], reuse=tf.AUTO_REUSE):
                    stream_method_projection = self.stream[out_projection]["method"](stream_method_projections,
                                                                                     copy(self.stream[out_projection]
                                                                                          ["parameters"]))

                # Update stream
                if update_stream:
                    if not isinstance(self.projections[out_projection], str):
                        self.projections[out_projection] = stream_method_projection

                # Return stream method projection
                return stream_method_projection

            # In case no projections to replace, but stream method being built still has to be started
            if self.projections[out_projection] is None and out_projection in self.stream:
                stream_method_projections = {}
                for projection in self.stream[out_projection]["projections"]:
                    # Stream method projections
                    stream_method_projections[projection] = self.build(projection, projections,
                                                                       update_stream=update_stream)

                    # Check if None
                    if stream_method_projections[projection] is None:
                        pass

                    # Update stream
                    if update_stream:
                        if not isinstance(self.projections[projection], str):
                            self.projections[projection] = stream_method_projections[projection]

                # Stream method projection
                with tf.variable_scope(self.stream[out_projection]["variable_scope"], reuse=tf.AUTO_REUSE):
                    stream_method_projection = self.stream[out_projection]["method"](stream_method_projections,
                                                                                     copy(self.stream[out_projection]
                                                                                          ["parameters"]))
                # Update stream
                if update_stream:
                    self.projections[out_projection] = stream_method_projection

                # Return stream method projection
                return stream_method_projection

            # Account for a string reference to another projection
            if isinstance(self.projections[out_projection], str):
                # Return reference projection
                return self.build(self.projections[out_projection], projections, update_stream=update_stream)

            # Account for a placeholder
            if isinstance(self.projections[out_projection], (list, tuple, tf.TensorShape)):
                if all(isinstance(dim, (int, tf.Dimension, type(None))) for dim in self.projections[out_projection]):
                    # Use placeholder
                    placeholder = tf.placeholder("float", list(self.projections[out_projection]))

                    # Update stream
                    if update_stream:
                        self.projections[out_projection] = placeholder

                    # Return placeholder
                    return placeholder

            # Return projection
            return self.projections[out_projection]

    def add_to_stream(self, stream_methods=None, projections=None, parameters=None, name_scope=None,
                      variable_scope=None, use_stream_projection=True, start=False):
        # Nothing to do
        if not stream_methods and not projections:
            return

        # Default projections
        if not projections:
            projections = []

        # Convert unique item to list
        if not isinstance(projections, list):
            projections = [projections]

        # Initialize dict of projections to pass into the stream method
        stream_method_projections = {}

        # Build the stream method projections using passed in projections
        for projection in projections:
            # If a projection name
            if isinstance(projection, str):
                # Use projection
                stream_method_projections[projection] = self.build(projection) if start else None

                # Store projection if not reference
                if projection not in self.projections:
                    self.projections[projection] = stream_method_projections[projection]

                # Don't use stream projection redundantly
                if projection == self.stream_projection_name:
                    use_stream_projection = False

            # If a dict of projections
            if isinstance(projection, dict):
                # Store definition and add to stream method
                for key in projection:
                    # Verify distinct
                    assert key not in self.projections

                    # Don't use stream projection redundantly
                    if key == self.stream_projection_name:
                        use_stream_projection = False

                    # Shapes to placeholders
                    if isinstance(projection[key], (list, tuple, tf.TensorShape)):
                        # If starting and all values of list are integers
                        if start and all(isinstance(dim, (int, tf.Dimension, type(None))) for dim in projection[key]):
                            # Use placeholder
                            projection[key] = tf.placeholder("float", list(projection[key]))

                        # Store definition (Note: can be a list of anything except integers)
                        self.projections[key] = projection[key]

                    # Names to projections
                    elif isinstance(projection[key], str):
                        # Don't use stream projection redundantly
                        if projection[key] == self.stream_projection_name:
                            use_stream_projection = False

                        # Store reference
                        self.projections[key] = projection[key]

                        # Use projection
                        projection[key] = self.build(projection[key])

                    # Brains to projections
                    elif isinstance(projection[key], Brains):
                        # Store reference
                        self.projections[key] = projection[key].stream_projection_name

                        # Copy over brain definitions
                        self.stream += projection[key].stream
                        assert not set(projection[key].projections.keys()) & set(self.projections.keys())
                        self.projections.update(projection[key].projections)

                        # Use brain's stream projection
                        # (Note: may have inconsistent/redundant behavior if stream proj name is this stream proj name)
                        projection[key] = projection[key].stream_projection

                    # If projection passed in
                    else:
                        # Store definition (Note: can be anything except list, string, or brain)
                        self.projections[key] = projection[key]

                    # Add to stream method
                    stream_method_projections[key] = projection[key]

            # If a brain
            if isinstance(projection, Brains):
                # Copy over brain definitions
                self.stream += projection.stream
                assert not set(projection.projections.keys()) & set(self.projections.keys())
                self.projections.update(projection.projections)

                # Use brain's stream projection
                stream_method_projections[projection.stream_projection_name] = projection.stream_projection

                # Don't use stream projection redundantly
                if projection.stream_projection_name == self.stream_projection_name:
                    use_stream_projection = False

        # If using stream projection, add to projections
        if use_stream_projection:
            if self.stream_projection_name:
                stream_method_projections[self.stream_projection_name] = self.build() if start else None

        # If no stream methods passed in
        if not stream_methods:
            return

        # Convert to list
        stream_methods = stream_methods if isinstance(stream_methods, list) else [stream_methods]

        # Name scope
        with optional_name_scope(name_scope):
            # Add to stream
            for stream_method in stream_methods:
                # Convert everything to standard list of dicts with name: method
                if not isinstance(stream_method, dict):
                    if callable(stream_method):
                        assert stream_method.__name__ not in self.projections
                        stream_method = {stream_method.__name__: stream_method}
                    if isinstance(stream_method, Brains):
                        stream_method = {stream_method.stream_projection_name: stream_method}

                # For each dict item
                for name in stream_method:
                    # If brain
                    if isinstance(stream_method[name], Brains):
                        # Copy dependencies of needed projections as well to the projected brain
                        def copy_over_dependencies(dependencies, copied_from, copied_to):
                            for dependant in dependencies:
                                # Add projection
                                if dependant not in copied_to.projections and dependant in copied_from.projections:
                                    copied_to.projections[dependant] = copy(copied_from.projections[dependant])
                                # Add stream definition and its dependant projections
                                if dependant not in copied_to.stream and dependant in copied_from.stream:
                                    copied_to.stream[dependant] = copy(copied_from.stream[dependant])
                                    copy_over_dependencies(copied_from.stream[dependant]["projections"],
                                                           copied_from, copied_to)

                        # Potentially start projected brain
                        start_and_copy_projected_brain = False

                        # Needed by the projected brain
                        needed_projections = {}

                        # Potentially start projected brain (if no variable scope passed in)
                        if start and not variable_scope:
                            if stream_method_projections:
                                # For collecting stream projections needed or unused
                                start_and_copy_projected_brain = True

                                # If projected brain needs these projections to start or doesn't use any of them, start
                                for projection in stream_method_projections:
                                    if projection in stream_method[name].projections:
                                        if stream_method[name].projections[projection] is not None:
                                            start_and_copy_projected_brain = False
                                        else:
                                            needed_projections[projection] = stream_method_projections[projection]

                                # Build
                                if start_and_copy_projected_brain:
                                    stream_method[name].projections.update(needed_projections)
                                    stream_method[name].build(update_stream=True)
                            else:
                                # Simply start and copy projected brain if no projections being passed to it
                                start_and_copy_projected_brain = True
                                stream_method[name].build(update_stream=True)

                        # Copy over projected brain's projections
                        for projection in stream_method[name].projections:
                            if projection not in stream_method_projections:
                                assert projection not in self.projections
                                # Copy projections if projected brain was started or if reference
                                self.projections[projection] = stream_method[name].projections[projection] \
                                    if isinstance(stream_method[name].projections[projection], str) \
                                       or start_and_copy_projected_brain else None

                        # Copy over projected brain's stream
                        for projection in stream_method[name].stream:
                            if projection not in stream_method_projections:
                                self.stream[projection] = stream_method[name].stream[projection]

                        # In case name given in dict is different from the projected brain's stream projection name
                        if name != stream_method[name].stream_projection_name:
                            assert name not in self.projections
                            self.projections[name] = stream_method[name].stream_projection_name

                        # If projected brain was copied over in full
                        if start_and_copy_projected_brain:
                            copy_over_dependencies(needed_projections, self, stream_method[name])
                        else:
                            with optional_variable_scope(variable_scope):
                                # Build this brain
                                self.build(name, update_stream=True)

                    # If method
                    if callable(stream_method[name]):
                        # Store stream definition
                        self.stream[name] = {"method": stream_method[name],
                                             "projections": list(stream_method_projections.keys()),
                                             "parameters": parameters,
                                             "variable_scope": variable_scope if variable_scope else name}

                        # Store stream projections definition
                        self.projections[name] = None

                        # Start brain
                        if start:
                            self.build(name, update_stream=True)

                    # Assign stream projection
                    self.stream_projection_name = name
                    self.stream_projection = self.projections[name]

                    # Pass this projection into the next stream method
                    stream_method_projections = {self.stream_projection_name: self.stream_projection}

    def adapt(self, stream_projections=None, projections=None, parameters=None, session=None, in_place=False):
        # Adapted brain TODO: allow changing which projection names are in stream method and the saved class projections
        if in_place:
            adapted_brain = self
        else:
            adapted_brain = Brains()
            adapted_brain.stream = copy(self.stream)
            for projection in self.projections:
                if isinstance(self.projections[projection], (str, list, tuple, tf.TensorShape)):
                    adapted_brain.projections[projection] = self.projections[projection]
                elif isinstance(self.projections[projection], tf.Tensor):
                    adapted_brain.projections[projection] = None
                    if self.projections[projection].op.type == 'Placeholder':
                        adapted_brain.projections[projection] = self.projections[projection].shape
                else:
                    adapted_brain.projections[projection] = None
            adapted_brain.stream_projection_name = self.stream_projection_name

            # print(adapted_brain.stream)
            # adapted_brain.build(update_stream=True)

        # Parameters
        if parameters is not None:
            for projection in adapted_brain.stream:
                if adapted_brain.stream[projection]["parameters"] is not None:
                    for parameter in adapted_brain.stream[projection]["parameters"]:
                        if parameter in parameters:
                            adapted_brain.stream[projection]["parameters"][parameter] = parameters[parameter]

        # Session
        if session is not None:
            adapted_brain.session = session

        # Update stream
        # if stream_projections is None:
        #     adapted_brain.build(projections=projections, update_stream=True)
        # else:
        #     if isinstance(stream_projections, list):
        #         for stream_projection in stream_projections:
        #             adapted_brain.build(stream_projection, projections, update_stream=True)
        #
        #         adapted_brain.stream_projection_name = stream_projections[0]
        #     if isinstance(stream_projections, str):
        #         adapted_brain.build(stream_projections, projections, update_stream=True)
        #     adapted_brain.stream_projection = adapted_brain.projections[adapted_brain.stream_projection_name]

        # Return adapted brain
        return adapted_brain


class Scope:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, arg1, arg2, arg3):
        pass


def optional_variable_scope(scope, reuse=tf.AUTO_REUSE):
    return Scope() if scope is None else tf.variable_scope(scope, reuse=reuse)


def optional_name_scope(scope):
    return Scope() if scope is None else tf.name_scope(scope)


def pd_lstm(input_dim, output_dim, num_layers=1, dropout=0, time_ahead_upstream=False, time_ahead_downstream=False):
    def mask(projections, parameters):
        # Mask for canceling out padding in dynamic sequences
        return tf.expand_dims(tf.sign(tf.reduce_max(tf.abs(projections["inputs"]), axis=2)), axis=2)

    def lstm(projections, parameters):
        # Default cell mode ("basic", "block", "cudnn") and number of layers
        num_layers = parameters["num_layers"] if parameters["num_layers"] else 1

        # Dropout
        if "dropout" in parameters:
            projections["inputs"] = tf.nn.dropout(projections["inputs"], keep_prob=1 - parameters["dropout"][0])

        # Add time ahead before lstm layer
        if "time_ahead_upstream" in parameters:
            if parameters["time_ahead_upstream"]:
                projections["inputs"] = tf.concat([projections["inputs"],
                                                   tf.expand_dims(projections["time_ahead"], 2)], 2)
                parameters["output_dim"] += 1

        # Dropout
        lstm_layers = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMBlockCell(parameters["output_dim"], forget_bias=5), output_keep_prob=
            1 - parameters["dropout"][1 if layer + 1 < num_layers else 2]) for layer in range(num_layers)])

        # Outputs and states of lstm layers
        outputs, _ = tf.nn.dynamic_rnn(lstm_layers, projections["inputs"], projections["time_dims"], dtype=tf.float32)

        # Add time ahead before lstm layer
        if "time_ahead_downstream" in parameters:
            if parameters["time_ahead_downstream"]:
                outputs = tf.concat([outputs, tf.expand_dims(projections["time_ahead"], 2)], 2)

        # Mask for canceling out padding in dynamic sequences
        outputs *= tf.tile(projections["mask"], [1, 1, outputs.shape[2]])

        # Return outputs
        return outputs

    # Return brains
    brain = Brains(mask, {"inputs": [None, None, input_dim]})
    brain.add_to_stream(lstm, ["inputs", {"time_dims": [None], "time_ahead": [None, None]}],
                         {"time_ahead_upstream": time_ahead_upstream, "time_ahead_downstream": time_ahead_downstream,
                          "dropout": dropout, "output_dim": output_dim, "num_layers": num_layers})
    return brain


def pd_lifelong_memory(representation_dim, num_similar_memories, output_dim, projection_name="projection_to_represent"):
    def representation(projections, parameters):
        # Inputs
        inputs = projections[projection_name]

        # Representation weights and bias
        weights = tf.get_variable("representation_weights", [inputs.shape[2], parameters["representation_dim"]])
        bias = tf.get_variable("representation_bias", [parameters["representation_dim"]])

        # Representation for memory
        out_projection = tf.einsum("aij,jk->aik", inputs, weights) + bias

        # If mask provided
        if projections["mask"] is not None:
            # Mask for canceling out padding in dynamic sequences
            out_projection *= tf.tile(projections["mask"], [1, 1, parameters["representation_dim"]])

        # Return representation
        return out_projection

    def expectation(projections, parameters):
        # To prevent division by 0 and to prevent NaN gradients from square root of 0
        distance_delta = 0.001

        # Distances (batch x time x k)
        distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(
            tf.tile(tf.expand_dims(projections["representation"], 2), [1, 1, parameters["num_similar_memories"], 1]),
            projections["remembered_representations"]), 3) + distance_delta ** 2)

        # distances = tf.reduce_sum(tf.squared_difference(
        #     tf.tile(tf.expand_dims(representation, 2), [1, 1, self.num_similar_memories, 1]),
        #     remembered_representations), 3)

        # Weights (batch x time x k x attributes)
        weights = tf.tile(tf.expand_dims(1.0 / distances, axis=3), [1, 1, 1, parameters["output_dim"]])
        # weights = tf.tile(tf.expand_dims(1.0 / (distances + distance_delta), axis=3),
        #                   [1, 1, 1, self.attributes["attributes"]])

        # Division numerator and denominator (for weighted means)
        numerator = tf.reduce_sum(weights * projections["remembered_attributes"], axis=2)  # Weigh attributes
        denominator = tf.reduce_sum(weights, axis=2)  # Normalize weightings

        # In case denominator is zero
        safe_denominator = tf.where(tf.less(denominator, 1e-7), tf.ones_like(denominator), denominator)

        # Distance weighted memory attributes (batch x time x attributes)
        out_projection = tf.divide(numerator, safe_denominator)

        # If mask needed
        if projections["mask"] is not None:
            # Apply mask to outputs
            out_projection *= tf.tile(projections["mask"], [1, 1, parameters["output_dim"]])

        # Return expectation
        return out_projection

    # Return brain
    brain = Brains(representation, [projection_name, "mask"], {"representation_dim": representation_dim})
    brain.add_to_stream(expectation,
                        [{"remembered_representations": (None, None, num_similar_memories, representation_dim),
                          "remembered_attributes": (None, None, num_similar_memories, output_dim)}, "mask"],
                        {"num_similar_memories": num_similar_memories, "output_dim": output_dim})
    return brain
