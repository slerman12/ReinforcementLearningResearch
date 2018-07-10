import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import cudnn_rnn


class Brains:
    def __init__(self, parameters):
        # Initialize brain 
        self.brain = None

        # Initialize session
        self.session = None

        # Initialize graph placeholders
        self.placeholders = None

        # Initialize graph components
        self.components = None

        # Set parameters
        self.parameters = parameters

        # Build the brain
        self.build()

    def run(self, placeholders=None, component=None):
        # Default no inputs
        if placeholders is None:
            placeholders = {}

        # Run inputs through brain
        return self.session.run(self.brain if component is None else component,
                                feed_dict={self.placeholders[key]: placeholders[key] for key in placeholders.keys()})

    def build(self):
        pass


class StaticLSTMOutputLast(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, self.parameters["timesteps"], self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [None, self.parameters["output_dim"]])
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs}

        # Model parameters
        weights = {
            'output': tf.Variable(tf.random_normal([self.parameters["hidden_dim"], self.parameters["output_dim"]]))
        }
        biases = {
            'output': tf.Variable(tf.random_normal([self.parameters["output_dim"]]))
        }

        # Split up inputs into tensors of batch_size x num_input
        inputs = tf.unstack(inputs, self.parameters["timesteps"], 1)

        # Layer of LSTM cells
        lstm_layer = rnn.BasicLSTMCell(self.parameters["hidden_dim"], forget_bias=1.0)

        # Outputs and states of lstm layer
        outputs, states = rnn.static_rnn(lstm_layer, inputs, dtype=tf.float32)

        # Logits
        logits = tf.matmul(outputs[-1], weights['output']) + biases['output']

        # Activation
        self.components = {"output": tf.nn.softmax(logits), "logits": logits}

        # Brain
        self.brain = self.components["output"]


class DynamicLSTMOutputLast(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, None, self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [None, self.parameters["output_dim"]])
        sequence_length = tf.placeholder(tf.int32, [None])
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs, "sequence_length": sequence_length}

        # Model parameters
        weights = {
            'output': tf.Variable(tf.random_normal([self.parameters["hidden_dim"], self.parameters["output_dim"]]))
        }
        biases = {
            'output': tf.Variable(tf.random_normal([self.parameters["output_dim"]]))
        }

        # Transpose to timesteps x batch_size x num_input
        inputs = tf.transpose(inputs, [1, 0, 2])

        # Layer of LSTM cells
        lstm_layer = rnn.LSTMBlockFusedCell(self.parameters["hidden_dim"])

        # Outputs and states of lstm layer
        outputs, states = lstm_layer(inputs, dtype=tf.float32, sequence_length=sequence_length)

        # Dropout
        if "dropout" in self.parameters.keys():
            outputs = tf.nn.dropout(outputs, self.parameters["dropout"])

        # Stack outputs into batch_size x max_timesteps x input_dim
        outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])

        # Indexing the output of each lstm cell at the last time step
        outputs = tf.gather_nd(outputs, tf.stack([tf.range(tf.shape(outputs)[0]), sequence_length - 1], axis=1))

        # Logits
        logits = tf.matmul(outputs, weights['output']) + biases['output']

        # Activation
        self.components = {"output": tf.nn.softmax(logits), "logits": logits}

        # Brain
        self.brain = self.components["output"]


class DynamicLSTM(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, None, self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [None, None, self.parameters["output_dim"]])
        sequence_length = tf.placeholder(tf.int32, [None])
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs, "sequence_length": sequence_length}

        # Model parameters
        weights = {
            'output': tf.Variable(tf.random_normal([self.parameters["hidden_dim"], self.parameters["output_dim"]]))
        }
        biases = {
            'output': tf.Variable(tf.random_normal([self.parameters["output_dim"]]))
        }

        # Transpose to timesteps x batch_size x num_input
        inputs = tf.transpose(inputs, [1, 0, 2])

        # Layer of LSTM cells
        lstm_layer = rnn.LSTMBlockFusedCell(self.parameters["hidden_dim"])

        # Outputs and states of lstm layer
        outputs, states = lstm_layer(inputs, dtype=tf.float32, sequence_length=sequence_length)

        # Dropout
        if "dropout" in self.parameters.keys():
            outputs = tf.nn.dropout(outputs, self.parameters["dropout"])

        # Stack outputs into batch_size x max_timesteps x input_dim
        outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])

        # Logits
        logits = tf.reshape(outputs, [-1, self.parameters["hidden_dim"]])
        logits = tf.matmul(logits, weights["output"]) + biases['output']
        logits = tf.reshape(logits, [-1, self.parameters["max_timesteps"], self.parameters["output_dim"]])

        # Activation
        self.components = {"output": tf.nn.softmax(logits), "logits": logits}

        # Brain
        self.brain = self.components["output"]


class TruncatedDynamicLSTM(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, None, self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [None, None, self.parameters["output_dim"]])
        sequence_length = tf.placeholder(tf.int32, [None])
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs, "sequence_length": sequence_length}

        # Model parameters
        weights = {
            'output': tf.Variable(tf.random_normal([self.parameters["hidden_dim"], self.parameters["output_dim"]]))
        }
        biases = {
            'output': tf.Variable(tf.random_normal([self.parameters["output_dim"]]))
        }

        # Unstack to inputs of batch_size x num_input
        # inputs = tf.unstack(inputs, self.parameters["truncated_timesteps"], 1)

        # Layer of LSTM cells
        lstm_layer = rnn.BasicLSTMCell(self.parameters["hidden_dim"])

        initial_state = lstm_layer.zero_state(self.parameters["batch_size"], "float")
        self.placeholders["initial_state"] = initial_state

        # Iterate through truncated sequence
        outputs, final_states = tf.nn.dynamic_rnn(lstm_layer, inputs, initial_state=initial_state,
                                                  sequence_length=sequence_length)

        # Dropout
        if "dropout" in self.parameters.keys():
            outputs = tf.nn.dropout(outputs, self.parameters["dropout"])

        # Stack outputs into batch_size x truncated_timesteps x input_dim
        outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])

        # Logits
        logits = tf.reshape(outputs, [-1, self.parameters["hidden_dim"]])
        logits = tf.matmul(logits, weights["output"]) + biases['output']
        logits = tf.reshape(logits, [-1, self.parameters["truncated_timesteps"], self.parameters["output_dim"]])

        # Activation
        self.components = {"output": tf.nn.softmax(logits), "logits": logits, "final_state": final_states}

        # Brain
        self.brain = self.components["output"]
