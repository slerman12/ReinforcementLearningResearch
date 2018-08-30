import tensorflow as tf


class Brains:
    def __init__(self, parameters=None, brain=None, placeholders=None, components=None, tensorflow=True, session=None):
        # Initialize brain 
        self.brain = brain

        # Set parameters
        self.parameters = parameters

        # Initialize graph placeholders
        self.placeholders = placeholders

        # Initialize graph components
        self.components = components

        # If using TensorFlow
        self.tensorflow = tensorflow

        # Initialize session
        self.session = session

    def run(self, placeholders=None, components=None, partial_run=None, partial_run_setup=None):
        # Default no inputs
        if placeholders is None:
            placeholders = {}

        # Partial run setup if list not provided but requested
        if partial_run_setup is True:
            partial_run_setup = [[fetch for fetch in list(self.components.values()) if fetch is not None],
                                 self.placeholders]
        elif isinstance(partial_run_setup, list):
            assert len(partial_run_setup) == 2
            partial_run_setup[0] = [fetch for fetch in partial_run_setup[0] if fetch is not None]
            for i, fetch in enumerate(partial_run_setup[0]):
                if isinstance(fetch, str):
                    partial_run_setup[0][i] = self.components[fetch]

        # Graph placeholders to use
        if partial_run_setup is None:
            placeholders = {self.placeholders[key]: placeholders[key] for key in placeholders if key in
                            self.placeholders and self.placeholders[key] is not None}
        else:
            placeholders = {partial_run_setup[1][key]: placeholders[key] for key in placeholders if key in
                            partial_run_setup[1]}

        # Graph component(s) to run
        if components is None:
            components = [self.brain]
        elif isinstance(components, str):
            components = [self.components[components]]
        elif isinstance(components, list):
            for i, component in enumerate(components):
                if isinstance(component, str):
                    components[i] = self.components[component]
        else:
            components = [components]
        components = [component for component in components if component is not None]

        # If TensorFlow
        if self.tensorflow:
            # Set up partial run
            if partial_run is None and partial_run_setup is not None:
                partial_run = self.session.partial_run_setup(partial_run_setup[0] + components,
                                                             list(partial_run_setup[1].values()))

            # Output single element if components is a single item list
            if isinstance(components, list):
                if len(components) == 1:
                    components = components[0]

            # If not partial run, return regular run
            if partial_run is None:
                # Return the regular run
                return self.session.run(components, feed_dict=placeholders), None
            else:
                # Return the result and the partially computed graph
                return self.session.partial_run(partial_run, components, placeholders), partial_run

    def build(self):
        pass

    def adapt(self, parameters=None, placeholders=None, components=None, tensorflow=None, session=None):
        # Default mutations
        if parameters is None:
            parameters = {}
        if placeholders is None:
            placeholders = {}
        if components is None:
            components = {}
        if tensorflow is None:
            tensorflow = self.tensorflow
        if session is None:
            session = self.session

        # Initialize adaptations
        adapted_parameters = parameters
        adapted_placeholders = placeholders
        adapted_components = components

        # Brain genes mutated
        if self.parameters is not None:
            adapted_parameters = dict(self.parameters)
            adapted_parameters.update(parameters)
        if self.placeholders is not None:
            adapted_placeholders = dict(self.placeholders)
            adapted_placeholders.update(placeholders)
        if self.components is not None:
            adapted_components = dict(self.components)
            adapted_components.update(components)

        # Return adapted brain
        return self.__class__(adapted_parameters, None, adapted_placeholders, adapted_components, tensorflow, session)


# An LSTM whose output is its last time step's output only
class LSTMOutputLast(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, None, self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [None, self.parameters["output_dim"]])
        time_dims = tf.placeholder(tf.int32, [None]) if "max_time_dim" in self.parameters else None
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs, "time_dims": time_dims,
                             "learning_rate": tf.placeholder(tf.float32, shape=[])}

        # Dropout
        if "dropout" in self.parameters:
            inputs = tf.nn.dropout(inputs, keep_prob=1 - self.parameters["dropout"])

        # Transpose to time_dim x batch_dim x input_dim
        inputs = tf.transpose(inputs, [1, 0, 2])

        # Layer of LSTM cells
        lstm_layer = tf.contrib.rnn.LSTMBlockFusedCell(self.parameters["hidden_dim"])

        # Outputs and states of lstm layer
        outputs, states = lstm_layer(inputs, dtype=tf.float32, time_dims=time_dims)

        # Dropout
        if "dropout" in self.parameters:
            outputs = tf.nn.dropout(outputs, keep_prob=1 - self.parameters["dropout"])

        # Index output of each lstm cell at the last time dimension (accounting for dynamic numbers of time dimensions)
        outputs = tf.gather_nd(tf.stack(outputs), tf.stack([time_dims - 1, tf.range(tf.shape(outputs)[1])], 1)) \
            if "max_time_dim" in self.parameters else outputs[-1]

        # Final dense layer weights
        output_weights = tf.get_variable("output_weights", [self.parameters["hidden_dim"],
                                                            self.parameters["output_dim"]])

        # Final dense layer bias
        output_bias = tf.get_variable("output_bias", [self.parameters["output_dim"]])

        # Logits
        outputs = tf.matmul(outputs, output_weights) + output_bias

        # Components
        self.components = {"outputs": outputs}

        # Brain
        self.brain = self.components["outputs"]


class LSTM(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, None, self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [None, None, self.parameters["output_dim"]])
        time_dims = tf.placeholder(tf.int32, [None]) if "max_time_dim" in self.parameters else None
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs, "time_dims": time_dims,
                             "learning_rate": tf.placeholder(tf.float32, shape=[])}

        # Default cell mode ("basic", "block", "cudnn") and number of layers
        mode = self.parameters["mode"] if "mode" in self.parameters else "block"
        num_layers = self.parameters["num_layers"] if "num_layers" in self.parameters else 1

        # Dropout
        if "dropout" in self.parameters:
            inputs = tf.nn.dropout(inputs, keep_prob=1 - self.parameters["dropout"][0])

        # Layers of LSTM cells
        if mode == "cudnn":
            # Transpose inputs to time_dim x batch_dim x input_dim
            inputs = tf.transpose(inputs, [1, 0, 2])

            # Layers of lstm cells
            lstm_layers = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=num_layers, num_units=self.parameters["hidden_dim"],
                                                         dropout=self.parameters["dropout"][1])

            # Initial state
            initial_state = (tf.zeros([num_layers, self.parameters["batch_dim"],
                                       self.parameters["hidden_dim"]], tf.float32),
                             tf.zeros([num_layers, self.parameters["batch_dim"],
                                       self.parameters["hidden_dim"]], tf.float32))
            self.placeholders["initial_state"] = initial_state

            # Outputs and states of lstm layers
            outputs, final_states = lstm_layers(inputs, initial_state)

            # Transpose outputs into batch_dim x time_dim x output_dim
            outputs = tf.transpose(outputs, [1, 0, 2])

            # Dropout
            if "dropout" in self.parameters:
                outputs = tf.nn.dropout(outputs, keep_prob=1 - self.parameters["dropout"][2])
        elif mode == "fused":
            # Transpose inputs to time_dim x batch_dim x input_dim
            inputs = tf.transpose(inputs, [1, 0, 2])

            # Initialize layers, states, and outputs
            layers = []
            initial_states = [(tf.zeros([self.parameters["batch_dim"], self.parameters["hidden_dim"]], tf.float32),
                               tf.zeros([self.parameters["batch_dim"], self.parameters["hidden_dim"]], tf.float32))
                              for _ in range(num_layers)]
            outputs = inputs
            final_states = []

            # Feed one layer into the next
            for layer in range(num_layers):
                layers.append(tf.contrib.rnn.LSTMBlockFusedCell(self.parameters["hidden_dim"]))
                outputs, final_state = layers[-1](outputs, initial_states[layer], tf.float32, time_dims)
                final_states.append(final_state)

                # Dropout
                if "dropout" in self.parameters:
                    outputs = tf.nn.dropout(outputs, keep_prob=1 - self.parameters["dropout"][1 if layer + 1 <
                                                                                                   num_layers else 2])

            # Transpose outputs into batch_dim x time_dim x output_dim
            outputs = tf.transpose(outputs, [1, 0, 2])
        else:
            # Dropout
            if "dropout" in self.parameters:
                lstm_layers = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.BasicLSTMCell(self.parameters["hidden_dim"]) if mode == "basic"
                        else tf.contrib.rnn.LSTMBlockCell(self.parameters["hidden_dim"]),
                        output_keep_prob=1 - self.parameters["dropout"][1 if layer + 1 < num_layers else 2])
                        for layer in range(num_layers)])
            else:
                # Layers of lstm cells
                lstm_layers = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.parameters["hidden_dim"])
                                                           if mode == "basic" else
                                                           tf.contrib.rnn.LSTMBlockCell(self.parameters["hidden_dim"])
                                                           for _ in range(num_layers)])

            # Initial state
            initial_state = lstm_layers.zero_state(self.parameters["batch_dim"], tf.float32)
            self.placeholders["initial_state"] = initial_state

            # Outputs and states of lstm layers
            outputs, final_states = tf.nn.dynamic_rnn(lstm_layers, inputs, time_dims, initial_state)

        # Final dense layer weights
        output_weights = tf.get_variable("output_weights", [self.parameters["hidden_dim"],
                                                            self.parameters["output_dim"]])

        # Final dense layer bias
        output_bias = tf.get_variable("output_bias", [self.parameters["output_dim"]])

        # Dense layer (careful: bias or cudnn would corrupt padding. Hence mask needed)
        outputs = tf.einsum('aij,jk->aik', outputs, output_weights) + output_bias

        # Components
        self.components = {"outputs": outputs, "final_state": final_states}

        # Brain
        self.brain = self.components["outputs"]


class BidirectionalLSTM(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, None, self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [None, None, self.parameters["output_dim"]])
        time_dims = tf.placeholder(tf.int32, [None]) if "max_time_dim" in self.parameters else None
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs, "time_dims": time_dims,
                             "learning_rate": tf.placeholder(tf.float32, shape=[])}

        # Dropout
        if "dropout" in self.parameters:
            inputs = tf.nn.dropout(inputs, keep_prob=1 - self.parameters["dropout"][0])

        # Layer of bidirectional LSTM
        cells = [tf.contrib.rnn.LSTMBlockCell(self.parameters["hidden_dim"]) for _ in range(2)]

        # Initial state
        initial_state = [cell.zero_state(self.parameters["batch_dim"], tf.float32) for cell in cells]
        self.placeholders["initial_state"] = initial_state

        # Outputs and states of lstm layers
        outputs, h1, h2 = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([cells[0]], [cells[1]], inputs,
                                                                         [initial_state[0]], [initial_state[1]],
                                                                         tf.float32, time_dims)

        # Final states
        final_states = [h1, h2]

        # Dropout
        if "dropout" in self.parameters:
            outputs = tf.nn.dropout(outputs, keep_prob=1 - self.parameters["dropout"][2])

        # Final dense layer weights
        output_weights = tf.get_variable("output_weights", [self.parameters["hidden_dim"] * 2,
                                                            self.parameters["output_dim"]])

        # Final dense layer bias
        output_bias = tf.get_variable("output_bias", [self.parameters["output_dim"]])

        # Dense layer (careful: bias or cudnn would corrupt padding. Hence mask needed)
        outputs = tf.einsum('aij,jk->aik', outputs, output_weights) + output_bias

        # Components
        self.components = {"outputs": outputs, "final_state": final_states}

        # Brain
        self.brain = self.components["outputs"]


class PD_LSTM_Model(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, None, self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [None, None, self.parameters["output_dim"]])
        time_dims = tf.placeholder(tf.int32, [None]) if "max_time_dim" in self.parameters else None
        time_ahead = tf.placeholder("float", [None, None])
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs, "time_dims": time_dims,
                             "time_ahead": time_ahead, "learning_rate": tf.placeholder(tf.float32, shape=[])}

        # Default cell mode ("basic", "block", "cudnn") and number of layers
        mode = self.parameters["mode"] if "mode" in self.parameters else "block"
        num_layers = self.parameters["num_layers"] if "num_layers" in self.parameters else 1

        # Dropout
        if "dropout" in self.parameters:
            inputs = tf.nn.dropout(inputs, keep_prob=1 - self.parameters["dropout"][0])

        # Add time ahead before lstm layer
        if "time_ahead_upstream" in self.parameters:
            if self.parameters["time_ahead_upstream"]:
                inputs = tf.concat([inputs, tf.expand_dims(time_ahead, 2)], 2)

        # Layers of LSTM cells
        if mode == "cudnn":
            # Transpose inputs to time_dim x batch_dim x input_dim
            inputs = tf.transpose(inputs, [1, 0, 2])

            # Layers of lstm cells
            lstm_layers = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=num_layers, num_units=self.parameters["hidden_dim"],
                                                         dropout=self.parameters["dropout"][1])

            # Initial state
            initial_state = (tf.zeros([num_layers, self.parameters["batch_dim"],
                                       self.parameters["hidden_dim"]], tf.float32),
                             tf.zeros([num_layers, self.parameters["batch_dim"],
                                       self.parameters["hidden_dim"]], tf.float32))
            self.placeholders["initial_state"] = initial_state

            # Outputs and states of lstm layers
            outputs, final_states = lstm_layers(inputs, initial_state)

            # Transpose outputs into batch_dim x time_dim x output_dim
            outputs = tf.transpose(outputs, [1, 0, 2])

            # Dropout
            if "dropout" in self.parameters:
                outputs = tf.nn.dropout(outputs, keep_prob=1 - self.parameters["dropout"][2])
        elif mode == "fused":
            # Transpose inputs to time_dim x batch_dim x input_dim
            inputs = tf.transpose(inputs, [1, 0, 2])

            # Initialize layers, states, and outputs
            layers = []
            initial_states = [(tf.zeros([self.parameters["batch_dim"], self.parameters["hidden_dim"]], tf.float32),
                               tf.zeros([self.parameters["batch_dim"], self.parameters["hidden_dim"]], tf.float32))
                              for _ in range(num_layers)]
            outputs = inputs
            final_states = []

            # Feed one layer into the next
            for layer in range(num_layers):
                layers.append(tf.contrib.rnn.LSTMBlockFusedCell(self.parameters["hidden_dim"]))
                outputs, final_state = layers[-1](outputs, initial_states[layer], tf.float32, time_dims)
                final_states.append(final_state)

                # Dropout
                if "dropout" in self.parameters:
                    outputs = tf.nn.dropout(outputs, keep_prob=1 - self.parameters["dropout"][1 if layer + 1 <
                                                                                                   num_layers else 2])

            # Transpose outputs into batch_dim x time_dim x output_dim
            outputs = tf.transpose(outputs, [1, 0, 2])
        else:
            # Dropout
            if "dropout" in self.parameters:
                lstm_layers = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.BasicLSTMCell(self.parameters["hidden_dim"]) if mode == "basic"
                        else tf.contrib.rnn.LSTMBlockCell(self.parameters["hidden_dim"], forget_bias=0.0),
                        output_keep_prob=1 - self.parameters["dropout"][1 if layer + 1 < num_layers else 2])
                        for layer in range(num_layers)])
            else:
                # Layers of lstm cells
                lstm_layers = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.parameters["hidden_dim"])
                                                           if mode == "basic" else
                                                           tf.contrib.rnn.LSTMBlockCell(self.parameters["hidden_dim"])
                                                           for _ in range(num_layers)])

            # Initial state
            initial_state = lstm_layers.zero_state(self.parameters["batch_dim"], tf.float32)
            self.placeholders["initial_state"] = initial_state

            # Outputs and states of lstm layers
            outputs, final_states = tf.nn.dynamic_rnn(lstm_layers, inputs, time_dims, initial_state)

        # Add time ahead before final dense layer
        hidden_dim = self.parameters["hidden_dim"]
        if "time_ahead_downstream" in self.parameters:
            if self.parameters["time_ahead_downstream"]:
                outputs = tf.concat([outputs, tf.expand_dims(time_ahead, 2)], 2)
                hidden_dim = self.parameters["hidden_dim"] + 1

        # Final dense layer weights
        output_weights = tf.get_variable("output_weights", [hidden_dim, self.parameters["output_dim"]])

        # Final dense layer bias
        output_bias = tf.get_variable("output_bias", [self.parameters["output_dim"]])

        # Dense layer (careful: bias or cudnn would corrupt padding. Hence mask needed)
        outputs = tf.einsum('aij,jk->aik', outputs, output_weights) + output_bias

        # Components
        self.components = {"outputs": outputs, "final_state": final_states}

        # Brain
        self.brain = self.components["outputs"]


class PD_LSTM_Memory_Model(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, None, self.parameters["input_dim"]])
        time_dims = tf.placeholder(tf.int32, [None]) if "max_time_dim" in self.parameters else None
        time_ahead = tf.placeholder("float", [None, None])
        self.placeholders = {"inputs": inputs, "time_dims": time_dims, "time_ahead": time_ahead}

        # Mask for canceling out padding in dynamic sequences
        self.components = {"mask": tf.expand_dims(tf.sign(tf.reduce_max(tf.abs(self.placeholders["inputs"]), axis=2)),
                                                  axis=2)}

        # Parameters
        self.parameters.update({"midstream_dim": self.parameters["output_dim"]})

        # Default cell mode ("basic", "block", "cudnn") and number of layers
        num_layers = self.parameters["num_layers"] if "num_layers" in self.parameters else 1

        # Dropout
        if "dropout" in self.parameters:
            inputs = tf.nn.dropout(inputs, keep_prob=1 - self.parameters["dropout"][0])

        # Add time ahead before lstm layer
        if "time_ahead_upstream" in self.parameters:
            if self.parameters["time_ahead_upstream"]:
                inputs = tf.concat([inputs, tf.expand_dims(time_ahead, 2)], 2)

        with tf.variable_scope('lstm1'):
            # Dropout
            lstm_layers = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMBlockCell(self.parameters["output_dim"], forget_bias=5), output_keep_prob=
                1 - self.parameters["dropout"][1 if layer + 1 < num_layers else 2]) for layer in range(num_layers)])

            # Outputs and states of lstm layers
            outputs, final_states = tf.nn.dynamic_rnn(lstm_layers, inputs, time_dims, dtype=tf.float32)

        # Add time ahead before lstm layer
        if "time_ahead_midstream" in self.parameters:
            if self.parameters["time_ahead_midstream"]:
                outputs = tf.concat([outputs, tf.expand_dims(time_ahead, 2)], 2)
                self.parameters.update({"midstream_dim": self.parameters["midstream_dim"] + 1})

        # Give mask the right dimensionality
        mask = tf.tile(self.components["mask"], [1, 1, self.parameters["midstream_dim"]])

        # Mask for canceling out padding in dynamic sequences
        outputs *= mask

        # Components
        self.components.update({"outputs": outputs, "final_state": final_states})

        # Brain
        self.brain = self.components["outputs"]
