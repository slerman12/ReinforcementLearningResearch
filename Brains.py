import tensorflow as tf


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


# An LSTM whose output is its last time step's output only
class LSTMOutputLast(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, None, self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [None, self.parameters["output_dim"]])
        sequence_length = tf.placeholder(tf.int32, [None]) if "max_time_dim" in self.parameters.keys() else None
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
        lstm_layer = tf.contrib.rnn.LSTMBlockFusedCell(self.parameters["hidden_dim"])

        # Outputs and states of lstm layer
        outputs, states = lstm_layer(inputs, dtype=tf.float32, sequence_length=sequence_length)

        # Dropout
        if "dropout" in self.parameters.keys():
            outputs = tf.nn.dropout(outputs, self.parameters["dropout"])

        # Index output of each lstm cell at the last time dimension (accounting for dynamic numbers of time dimensions)
        outputs = tf.gather_nd(tf.stack(outputs), tf.stack([sequence_length - 1, tf.range(tf.shape(outputs)[1])], 1)) \
            if "max_time_dim" in self.parameters.keys() else outputs[-1]

        # Logits
        outputs = tf.matmul(outputs, weights['output']) + biases['output']

        # Components
        self.components = {"outputs": outputs}

        # Brain
        self.brain = self.components["outputs"]


class LSTM(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [self.parameters["batch_dim"], None, self.parameters["input_dim"]])
        desired_outputs = tf.placeholder("float", [self.parameters["batch_dim"], None, self.parameters["output_dim"]])
        initial_state = (tf.zeros([self.parameters["batch_dim"], self.parameters["hidden_dim"]], tf.float32),
                         tf.zeros([self.parameters["batch_dim"], self.parameters["hidden_dim"]], tf.float32))
        sequence_length = tf.placeholder(tf.int32, [None]) if "max_time_dim" in self.parameters.keys() else None
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs, "initial_state": initial_state,
                             "sequence_length": sequence_length}

        # Model parameters
        weights = {
            'output': tf.Variable(tf.random_normal([self.parameters["hidden_dim"], self.parameters["output_dim"]]))
        }
        biases = {
            'output': tf.Variable(tf.random_normal([self.parameters["output_dim"]]))
        }

        # Transpose to time_dim x batch_dim x input_dim
        inputs = tf.transpose(inputs, [1, 0, 2])

        # Layer of LSTM cells
        lstm_layer = tf.contrib.rnn.LSTMBlockFusedCell(self.parameters["hidden_dim"])

        # Outputs and states of lstm layer
        outputs, final_states = lstm_layer(inputs, initial_state, tf.float32, sequence_length)

        # Dropout
        if "dropout" in self.parameters.keys():
            outputs = tf.nn.dropout(outputs, self.parameters["dropout"])

        # Stack outputs into batch_dim x time_dim x output_dim
        outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])

        # Dense layer (careful: bias corrupts padding! Hence mask is needed)
        # outputs = tf.reshape(outputs, [-1, self.parameters["hidden_dim"]])
        # outputs = tf.matmul(outputs, weights["output"]) + biases['output']
        # outputs = tf.reshape(outputs, [self.parameters["batch_dim"], -1, self.parameters["output_dim"]])
        outputs = tf.einsum('aij,jk->aik', outputs, weights["output"]) + biases['output']

        # Components
        self.components = {"outputs": outputs, "final_state": final_states}

        # Brain
        self.brain = self.components["outputs"]
