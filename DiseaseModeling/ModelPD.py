from __future__ import division
import Performance
import Vision
import Agent
import Brains
from DiseaseModeling.Data import Data
import numpy as np

# Training data
training = Data.ReadPD("Data/Processed/encoded.csv", targets=["UPDRS_I", "UPDRS_II", "UPDRS_III"])

# Brain parameters
brain_parameters = dict(learning_rate=0.01, input_dim=training.input_dim, output_dim=training.desired_output_dim,
                        hidden_dim=64, max_time_dim=training.max_num_records, batch_dim=64, max_gradient_clip_norm=5)

# Vision
vision = Vision.Vision(brain=Brains.LSTM(brain_parameters))

# Agent
agent = Agent.Regressor(vision=vision)

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Loss", "Root Mean Squared Error"],
                                      run_throughs_per_epoch=200)

# Main method
if __name__ == "__main__":
    # Start agent
    agent.start_brain()

    # Training iterations
    for episode in range(1, 100000000000 + 1):
        # Batch data
        batch_inputs, batch_desired_outputs, batch_sequence_lengths = training.iterate_batch(brain_parameters
                                                                                             ["batch_dim"])

        # Train
        loss = agent.learn({"inputs": batch_inputs, "desired_outputs": batch_desired_outputs,
                            "sequence_length": batch_sequence_lengths})

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Loss": loss,
                                         "Root Mean Squared Error": np.sqrt(loss)})

        # Display performance
        performance.output_performance(run_through=episode, special_aggregation={"Episode": lambda x: x[-1]})

    # Stop agent
    agent.stop_brain()
