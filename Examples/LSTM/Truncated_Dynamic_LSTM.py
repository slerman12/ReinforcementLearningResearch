from __future__ import division
import Performance
import Vision
import Agent
import Brains
from Examples.LSTM.Data import Data


# Training data
training = Data.ReadFables("Data/Aesop.txt")

# Brain parameters
brain_parameters = dict(learning_rate=0.01, input_dim=training.word_dim, hidden_dim=64, output_dim=training.word_dim,
                        max_time_dim=training.max_fable_length, batch_dim=64, max_gradient_clip_norm=5,
                        truncated_time_dim=5)

# Vision
vision = Vision.Vision(brain=Brains.LSTM(brain_parameters))

# Agent
agent = Agent.TruncatedBPTTClassifier(vision=vision)

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Loss"], run_throughs_per_epoch=200)

# Main method
if __name__ == "__main__":
    # Start agent
    agent.start_brain()

    # Training iterations
    for episode in range(1, 10000 + 1):
        # Batch data
        batch_inputs, batch_desired_outputs, batch_sequence_lengths = training.iterate_batch(brain_parameters
                                                                                             ["batch_dim"])

        # Train
        loss = agent.learn({"inputs": batch_inputs, "desired_outputs": batch_desired_outputs,
                            "sequence_length": batch_sequence_lengths})

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Loss": loss})

        # Display performance
        performance.output_performance(run_through=episode, special_aggregation={"Episode": lambda x: x[-1]})

    # Stop agent
    agent.stop_brain()
