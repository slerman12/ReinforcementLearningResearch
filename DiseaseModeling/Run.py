from __future__ import division

import time

import Performance
import Vision
import Agent
import Brains
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Brain parameters
brain_parameters = dict(learning_rate=0.001, batch_size=128, num_input=28, timesteps=28, num_hidden=128, num_classes=10)

# Vision
vision = Vision.Vision(brain=Brains.LSTM(brain_parameters))

# Agent
agent = Agent.LSTMClassifier(vision=vision)

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Loss", "Accuracy"], epoch=200)

# Main method
if __name__ == "__main__":
    # Start agent
    agent.start_brain()

    # Training iterations
    for episode in range(10000):
        # Batch data
        batch_x, batch_y = mnist.train.next_batch(brain_parameters["batch_size"])

        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((brain_parameters["batch_size"], brain_parameters["timesteps"],
                                   brain_parameters["num_input"]))

        # Train
        loss = agent.learn({"inputs": batch_x, "desired_outputs": batch_y})

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Loss": loss})

        # Display performance
        performance.output_performance(episode)

    # Testing data and labels
    test_data = mnist.test.images[:128].reshape((-1, brain_parameters["timesteps"], brain_parameters["num_input"]))
    test_label = mnist.test.labels[:128]

    # Print testing accuracy
    print("Testing Accuracy:",
          agent.session.run(agent.accuracy, feed_dict={"inputs": test_data, "desired_outputs": test_label}))

    # Stop agent
    agent.stop_brain()
