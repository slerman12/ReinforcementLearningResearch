from __future__ import division
import Performance
import Vision
import Agent
import Brains
from tensorflow.examples.tutorials.mnist import input_data

# MNIST data
mnist = input_data.read_data_sets("Data", one_hot=True)

# Brain parameters
brain_parameters = dict(learning_rate=0.001, batch_dim=128, input_dim=28, time_dim=28, hidden_dim=128, output_dim=10)

# Vision
vision = Vision.Vision(brain=Brains.StaticLSTMOutputLast(brain_parameters))

# Agent
agent = Agent.Classifier(vision=vision)

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Loss"], run_throughs_per_epoch=200)

# Main method
if __name__ == "__main__":
    # Start agent
    agent.start_brain()

    # Training iterations
    for episode in range(1, 10000 + 1):
        # Batch data
        batch_inputs, batch_desired_outputs = mnist.train.next_batch(brain_parameters["batch_dim"])

        # Reshape image to 28 x 28
        batch_inputs = batch_inputs.reshape((brain_parameters["batch_dim"], brain_parameters["time_dim"],
                                             brain_parameters["input_dim"]))

        # Train
        loss = agent.learn({"inputs": batch_inputs, "desired_outputs": batch_desired_outputs})

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Loss": loss})

        # Display performance
        performance.output_performance(episode, aggregation=lambda x: x[-1])

    # Testing data and labels
    test_data = mnist.test.images[:128].reshape((-1, brain_parameters["time_dim"], brain_parameters["num_input"]))
    test_label = mnist.test.labels[:128]

    # Print testing accuracy
    print("Testing Accuracy:", agent.brain.run({"inputs": test_data, "desired_outputs": test_label}, agent.accuracy))

    # Stop agent
    agent.stop_brain()
