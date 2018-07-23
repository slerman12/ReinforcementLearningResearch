import tensorflow as tf


class Brains:
    def __init__(self, parameters, scope_name="training", graph=None, pytorch=False):
        # Initialize brain 
        self.brain = None

        # If PyTorch
        self.pytorch = pytorch

        # Scope
        self.scope_name = scope_name
        self.graph = graph
        self.name_scope = None
        self.variable_scope = None
        self.initializer = None

        # Initialize session
        self.session = None

        # Initialize graph placeholders
        self.placeholders = None

        # Initialize graph components
        self.components = None

        # Set parameters
        self.parameters = parameters

        # Build the brain
        with self:
            self.build()

    def run(self, placeholders=None, component=None):
        # Default no inputs
        if placeholders is None:
            placeholders = {}

        # Run inputs through brain
        if not self.pytorch:
            return self.session.run(self.brain if component is None else component,
                                    feed_dict={self.placeholders[key]: placeholders[key] for key in placeholders})

    def build(self):
        pass

    def __enter__(self):
        # Initialize contexts
        if self.graph is None:
            self.graph = tf.Graph()
        if self.name_scope is None and self.variable_scope is None:
            self.name_scope = tf.name_scope(self.scope_name)
            self.variable_scope = tf.variable_scope("brain", reuse=None or self.scope_name != "training",
                                                    initializer=self.initializer)

        # Open contexts
        self.graph.as_default()
        self.name_scope.__enter__()
        self.variable_scope.__enter__()

    def __exit__(self, arg1, arg2, arg3):
        # Close context
        self.variable_scope.__exit__(arg1, arg2, arg3)
        self.name_scope.__exit__(arg1, arg2, arg3)


# An LSTM whose output is its last time step's output only
class LSTMOutputLast(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, None, self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [None, self.parameters["output_dim"]])
        time_dims = tf.placeholder(tf.int32, [None]) if "max_time_dim" in self.parameters else None
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs, "time_dims": time_dims,
                             "learning_rate": tf.placeholder(tf.float32, shape=[])}

        # Initializer
        self.initializer = tf.contrib.layers.xavier_initializer()

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

        # Initializer
        self.initializer = tf.contrib.layers.xavier_initializer()

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

        # Initializer
        self.initializer = tf.contrib.layers.xavier_initializer()

        # Default cell mode ("basic", "block", "cudnn") and number of layers
        mode = self.parameters["mode"] if "mode" in self.parameters else "block"
        num_layers = self.parameters["num_layers"] if "num_layers" in self.parameters else 1

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


class PDModel(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, None, self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [None, None, self.parameters["output_dim"]])
        time_dims = tf.placeholder(tf.int32, [None]) if "max_time_dim" in self.parameters else None
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs, "time_dims": time_dims,
                             "learning_rate": tf.placeholder(tf.float32, shape=[])}

        # Initializer
        self.initializer = tf.contrib.layers.xavier_initializer()

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
