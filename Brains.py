import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data


class Brains:
    def __init__(self, params):
        # Initialize brain 
        self.brain = None

        # Initialize graph components
        self.placeholders = None

        # Set parameters
        self.params = params

        # Build the brain
        self.build()

    def run(self, session, inputs):
        # Run inputs through brain
        return session.run(self.brain, feed_dict={"inputs": inputs})

    def build(self):
        pass


class LSTM(Brains):
    def build(self):
        # Graph placeholders
        inputs = tf.placeholder("float", [None, self.params["timesteps"], self.params["num_input"]])
        desired_outputs = tf.placeholder("float", [None, self.params["num_classes"]])
        self.placeholders = {"inputs": inputs, "desired_outputs": desired_outputs}

        # Model parameters
        weights = {
            'out': tf.Variable(tf.random_normal([self.params["num_hidden"], self.params["num_classes"]]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.params["num_classes"]]))
        }

        # Unstack to get a list of "timesteps" tensors of shape (batch_size, n_input)
        x = tf.unstack(self.inputs, self.params["timesteps"], 1)

        # Layer of LSTM cells
        lstm_cell = rnn.BasicLSTMCell(self.params["num_hidden"], forget_bias=1.0)

        # Outputs and states of lstm layer
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Activation
        self.brain = tf.nn.softmax(tf.matmul(outputs[-1], weights['out']) + biases['out'])
