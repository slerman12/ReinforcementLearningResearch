from __future__ import division
import Performance
import Vision
import Agent
import Brains
from Examples.LSTM.Data import Data
import numpy as np

# Training data
training = Data.ReadFables("Data/Aesop.txt")

# Brain parameters
brain_parameters = dict(learning_rate=0.01, input_dim=training.word_dim, hidden_dim=64, output_dim=training.word_dim,
                        max_timesteps=training.max_fable_length, truncated_timesteps=5, batch_size=64)

# Vision
vision = Vision.Vision(brain=Brains.TruncatedDynamicLSTM(brain_parameters))

# Agent
agent = Agent.TruncatedDynamicClassifier(vision=vision)

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Loss"], episodes_or_run_throughs_per_epoch=200)

# Main method
if __name__ == "__main__":
    # Start agent
    agent.start_brain()

    # Training iterations
    for episode in range(1, 10000 + 1):
        # Batch data
        batch_inputs, batch_desired_outputs, batch_sequence_lengths = training.iterate_batch(brain_parameters
                                                                                             ["batch_size"])

        # Initialize loss
        loss = 0

        # Initialize sequence subset boundaries truncated iteration
        sequence_subset_begin = 0
        sequence_subset_end = brain_parameters["truncated_timesteps"]

        # Initialize initial hidden state
        initial_state = agent.brain.run(None, agent.brain.placeholders["initial_state"])

        # Counter of truncated iterations
        truncated_iterations = 0

        # Truncated iteration through sequences
        while sequence_subset_begin < brain_parameters["max_timesteps"]:
            # Get sequence subsets
            batch_inputs_subset = batch_inputs[:, sequence_subset_begin:sequence_subset_end, :]
            batch_desired_outputs_subset = batch_desired_outputs[:, sequence_subset_begin:sequence_subset_end, :]
            batch_sequence_lengths_subset = np.maximum(np.minimum(batch_sequence_lengths -
                                                                  brain_parameters["truncated_timesteps"] *
                                                                  truncated_iterations,
                                                                  brain_parameters["truncated_timesteps"]), 0)

            # If one of the sequences is all empty, break TODO: Bucket batches to prevent breaking unfinished sequences
            if not batch_inputs_subset.any(axis=2).any(axis=1).all():
                break

            # Train
            _, subset_loss, initial_state = agent.brain.run({"inputs": batch_inputs_subset,
                                                             "desired_outputs": batch_desired_outputs_subset,
                                                             "sequence_length": batch_sequence_lengths_subset,
                                                             "initial_state": initial_state},
                                                            [agent.train, agent.loss,
                                                             agent.brain.components["final_state"]])

            # Loss
            loss += subset_loss

            # Increment sequence subset boundaries
            sequence_subset_begin += brain_parameters["truncated_timesteps"]
            sequence_subset_end += brain_parameters["truncated_timesteps"]

            # Increment truncated iterations counter
            truncated_iterations += 1

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Loss": loss})

        # Display performance
        performance.output_performance(episode, aggregation=lambda x: x[-1])

    # Testing data
    # test_data = testing.data
    # test_label = testing.labels
    # test_sequence_length = testing.seqlen
    #
    # # Print testing accuracy
    # print("Testing Accuracy:", agent.brain.run({"inputs": test_data, "desired_outputs": test_label,
    #                                             "sequence_length": test_sequence_length}, agent.accuracy))

    # Stop agent
    agent.stop_brain()
