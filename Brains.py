import tensorflow as tf
from tensorflow.contrib import rnn


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


class LSTM(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, self.parameters["timesteps"], self.parameters["num_input"]])
        desired_outputs = tf.placeholder("float", [None, self.parameters["num_classes"]])
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs}

        # Model parameters
        weights = {
            'out': tf.Variable(tf.random_normal([self.parameters["num_hidden"], self.parameters["num_classes"]]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.parameters["num_classes"]]))
        }

        # Split up inputs into tensors of batch_size x num_input
        inputs = tf.unstack(inputs, self.parameters["timesteps"], 1)

        # Layer of LSTM cells
        lstm_layer = rnn.BasicLSTMCell(self.parameters["num_hidden"], forget_bias=1.0)

        # Outputs and states of lstm layer
        outputs, states = rnn.static_rnn(lstm_layer, inputs, dtype=tf.float32)

        # Logits
        logits = tf.matmul(outputs[-1], weights['out']) + biases['out']

        # Activation
        self.components = {"output": tf.nn.softmax(logits), "logits": logits}

        # Brain
        self.brain = self.components["output"]
