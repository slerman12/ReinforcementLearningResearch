import tensorflow as tf
from copy import deepcopy


class Brains:
    def __init__(self, stream_methods=None, projections=None, parameters=None, tensorflow=True, session=None):
        # Stream
        self.stream = {}

        # Final stream projection
        self.stream_projection = None

        # Projection name
        self.stream_projection_name = None

        # All projections in stream
        self.projections = {}

        # Start brain with add_to_stream
        self.add_to_stream(stream_methods, projections, parameters)

        # If using TensorFlow
        self.tensorflow = tensorflow

        # TensorFlow partial run
        self.tensorflow_partial_run = None

        # Initialize session
        self.session = session

    def run(self, projections=None, out_projections=None, do_partial_run=False):
        # If TensorFlow
        if self.tensorflow:
            # TODO: if projection not started, build it
            # Default projections
            if not isinstance(projections, dict):
                projections = {} if projections is None else {"inputs": projections}

            # Projections to use
            if do_partial_run and self.tensorflow_partial_run is not None:
                projections = {key: projections[key] for key in projections
                               if key in self.tensorflow_partial_run["feeds"]}

            # Projections to use
            projections = {self.projections[key]: projections[key] for key in projections if key in
                           self.projections and self.build(key) is not None}

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
                out_projections = [self.stream_projection]
            elif isinstance(out_projections, str):
                out_projections = [self.projections[out_projections]]
            elif isinstance(out_projections, list):
                for i, component in enumerate(out_projections):
                    if isinstance(component, str):
                        out_projections[i] = self.projections[component]
            else:
                out_projections = [out_projections]
            out_projections = [component for component in out_projections if component is not None]

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
                result = self.session.partial_run(self.tensorflow_partial_run["partial_run"], out_projections, projections)

                # If final stream projection of partial run, reset
                if last_partial_run:
                    self.tensorflow_partial_run["partial_run"] = None

                # Return
                return result
            else:
                # Return the regular run
                return self.session.run(out_projections, feed_dict=projections)

    def build(self, out_projection=None, projections=None, name_scope=None, variable_scope=None, update_stream=False):
        # Stream projection to build
        if out_projection is None:
            out_projection = self.stream_projection_name
        elif isinstance(out_projection, function):
            out_projection = out_projection.__name__

        # Name scope
        with optional_name_scope(name_scope if self.tensorflow else None):
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
                        if isinstance(projection, function):
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

                        # Convert string to projection
                        if isinstance(projections[projection], str):
                            projections[projection] = self.build(projections[projection], projections)

                        # Replace
                        stream_method_projections[projection] = projections[projection]

                        # Assumes unique and prunes; faster, but comment out if methods reuse the same projection
                        # del projections[projection]
                    else:
                        stream_method_projections[projection] = self.build(projection, projections)

                # Stream method projection (Note: does scoping work with recursion?)
                with tf.variable_scope(self.stream[out_projection]["variable_scope"] if variable_scope is None
                                       else variable_scope, reuse=tf.AUTO_REUSE):
                    stream_method_projection = self.stream[out_projection]["method"](stream_method_projections,
                                                                                     self.stream[out_projection]
                                                                                     ["parameters"])

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
                    stream_method_projections[projection] = self.build(projection, projections)

                    # Update stream
                    if update_stream:
                        if not isinstance(self.projections[projection], str):
                            self.projections[projection] = stream_method_projections[projection]

                # Stream method projection
                with tf.variable_scope(self.stream[out_projection]["variable_scope"] if variable_scope is None
                                       else variable_scope, reuse=tf.AUTO_REUSE):
                    stream_method_projection = self.stream[out_projection]["method"](stream_method_projections,
                                                                                     self.stream[out_projection]
                                                                                     ["parameters"])
                # Update stream
                if update_stream:
                    self.projections[out_projection] = stream_method_projection

                # Return stream method projection
                return stream_method_projection

            # Account for a string reference to another projection
            if isinstance(self.projections[out_projection], str):
                # Return reference projection
                return self.build(self.projections[out_projection], projections)

            # Return projection
            return self.projections[out_projection]

    def add_to_stream(self, stream_methods=None, projections=None, parameters=None, name_scope=None,
                      variable_scope=None, use_stream_projection=True, start=True):
        # Nothing to do
        if not stream_methods and not projections:
            return

        # Default projections
        if not projections:
            projections = []

        # Convert unique item to list
        if not isinstance(projections, list):
            projections = [projections]

        # If using stream projection, add to projections
        if use_stream_projection:
            if self.stream_projection_name:
                projections.append(self.stream_projection_name)

        # Initialize dict of projections to pass into the stream method
        stream_method_projections = {}

        # Build the stream method projections using passed in projections
        for projection in projections:
            # If a projection name
            if isinstance(projection, str):
                # Use projection
                stream_method_projections[projection] = self.build(projection)

            # If a dict of projections
            if isinstance(projection, dict):
                # Store definition and add to stream method
                for key in projection:
                    # Verify distinct
                    assert key not in self.projections

                    # Shapes to placeholders
                    if isinstance(projection[key], (list, tuple)):
                        # If starting and all values of list are integers
                        if start and all(isinstance(value, int) for value in projection[key]):
                            # Use placeholder
                            projection[key] = tf.placeholder("float", list(projection[key]))

                        # Store definition (Note: can be a list of anything except integers)
                        self.projections[key] = projection[key]

                    # Names to projections
                    elif isinstance(projection[key], str):
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

        # If no stream methods passed in
        if not stream_methods:
            return

        # Convert to list
        if not isinstance(stream_methods, list):
            stream_methods = [stream_methods]

        # Name scope
        with optional_name_scope(name_scope if self.tensorflow else None):
            # Add to stream
            for stream_method in stream_methods:
                # Convert everything to standard list of dicts with name: method
                if not isinstance(stream_method, dict):
                    if isinstance(stream_method, function):
                        stream_method = {stream_method.__name__: stream_method}
                        assert stream_method.__name__ not in self.projections
                    if isinstance(stream_method, Brains):
                        stream_method = {stream_method.stream_projection_name: stream_method}

                # For each dict item
                for name in stream_method:
                    # If brain
                    if isinstance(stream_method[name], Brains):
                        # Copy over projection definitions
                        for projection in stream_method[name].projections:
                            if projection not in stream_method_projections:
                                assert projection not in self.projections
                                self.projections[projection] = stream_method[name].projections[projection] \
                                    if isinstance(stream_method[name].projections[projection], str) else None

                        # Copy over stream definition
                        self.stream += stream_method[name].stream

                        # Store stream definition
                        if name != stream_method[name].stream_projection_name:
                            assert name not in self.projections
                            self.projections[name] = stream_method[name].stream_projection_name

                        # Store stream projections definition
                        self.projections[name] = self.build(name, update_stream=True) if start else None

                    # If method
                    if isinstance(stream_method[name], function):
                        # Store stream definition
                        self.stream[name] = {"method": stream_method[name],
                                             "projections": list(stream_method_projections.keys()),
                                             "parameters": parameters,
                                             "name_scope": name_scope,
                                             "variable_scope": variable_scope}

                        # Store stream projections definition
                        if self.tensorflow and start:
                            with tf.variable_scope(name if name_scope is None else name_scope, reuse=tf.AUTO_REUSE):
                                self.projections[name] = self.stream[name]["method"](projections, parameters)
                        elif start:
                            self.projections[name] = self.stream[name]["method"](projections, parameters)
                        else:
                            self.projections[name] = None

                    # Assign stream projection
                    self.stream_projection_name = name
                    self.stream_projection = self.projections[name]

                    # Pass this projection into the next stream method
                    stream_method_projections = {self.stream_projection_name: self.stream_projection}

    def adapt(self, stream_projections=None, projections=None, tensorflow=None, session=None, in_place=False):
        # Adapted brain TODO: allow changing which projection names are in stream method and the saved class projections
        adapted_brain = self if in_place else deepcopy(self)

        # TensorFlow
        if tensorflow is not None:
            adapted_brain.tensorflow = tensorflow

        # Session
        if session is not None:
            adapted_brain.session = session

        # Update stream
        if stream_projections is None:
            adapted_brain.build(projections=projections, update_stream=True)
        else:
            if isinstance(stream_projections, list):
                for stream_projection in stream_projections:
                    adapted_brain.build(stream_projection, projections, update_stream=True)

                adapted_brain.stream_projection_name = stream_projections[0]
            if isinstance(stream_projections, str):
                adapted_brain.build(stream_projections, projections, update_stream=True)
            adapted_brain.stream_projection = adapted_brain.projections[adapted_brain.stream_projection_name]

        # Return adapted brain
        return adapted_brain


class Scope:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, arg1, arg2, arg3):
        pass


def optional_variable_scope(scope, reuse=None):
    return Scope() if scope is None else tf.variable_scope(scope, reuse=reuse)


def optional_name_scope(scope):
    return Scope() if scope is None else tf.name_scope(scope)
