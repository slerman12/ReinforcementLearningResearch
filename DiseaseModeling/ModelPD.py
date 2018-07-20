from __future__ import division
import Performance
import Agent
import Brains
from DiseaseModeling.Data import Data
import numpy as np


# Data reader
reader = Data.ReadPD("Data/Processed/encoded.csv", targets=["UPDRS_I", "UPDRS_II", "UPDRS_III"], train_test_split=0.7,
                     valid_eval_split=0.33)

# Validation data
validation_data = reader.read(reader.validation_data)

# Brain parameters
brain_parameters = dict(batch_dim=20, input_dim=reader.input_dim, hidden_dim=650, output_dim=reader.desired_output_dim,
                        max_time_dim=reader.max_num_records, num_layers=2, dropout=[0.2, 0, 0.5], mode="block",
                        max_gradient_clip_norm=5)

# Validation parameters
validation_parameters = brain_parameters.copy()
validation_parameters["dropout"] = [0, 0, 0]
validation_parameters["batch_dim"] = len(reader.validation_data)

# Agent
agent = Agent.Regressor(brain=Brains.LSTM(brain_parameters))

# Validation
validate = Agent.Regressor(brain=Brains.LSTM(validation_parameters, "validating"))

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Learning Rate", "RMSE", 
                                                    "Loss (MSE)", "Validation (MSE)"], description=brain_parameters,
                                      run_throughs_per_epoch=len(reader.training_data) // brain_parameters["batch_dim"])

# Main method
if __name__ == "__main__":
    # Start agent
    session = agent.start_brain()

    # Build graph for evaluation
    validate.start_brain(session)

    # Training iterations
    for episode in range(1, 100000000000 + 1):
        # Batch data
        batch_inputs, batch_desired_outputs, batch_time_dims = reader.iterate_batch(brain_parameters["batch_dim"])

        # Learning rate
        learning_rate = 0.0001

        # Train
        loss = agent.learn({"inputs": batch_inputs, "desired_outputs": batch_desired_outputs,
                            "time_dims": batch_time_dims, "learning_rate": learning_rate})

        # Validate
        validation_mse = validate.brain.run(validation_data, validate.loss) if performance.is_epoch(episode) else None

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Learning Rate": learning_rate,
                                         "RMSE": np.sqrt(loss), "Loss (MSE)": loss, "Validation (MSE)": validation_mse})

        # Display performance
        performance.output_performance(run_through=episode, special_aggregation={"Episode": lambda x: x[-1],
                                                                                 "Validation (MSE)": lambda x: x[-1]})

    # Stop agent
    agent.stop_brain()
