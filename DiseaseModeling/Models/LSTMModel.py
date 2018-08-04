from __future__ import division
import Performance
import Agent
import Vision
import Brains
from DiseaseModeling.Data import Data
import numpy as np

# Restore saved agent
restore = False

# Data reader
reader = Data.ReadPD("Data/Processed/encoded.csv", targets=["UPDRS_I", "UPDRS_II", "UPDRS_III"], train_test_split=0.7,
                     valid_eval_split=1, sequence_dropout=0.2)

# Brain parameters
brain_parameters = dict(batch_dim=32, input_dim=reader.input_dim, hidden_dim=128, output_dim=reader.desired_output_dim,
                        max_time_dim=reader.max_num_records, num_layers=1, dropout=[0.2, 0, 0.65], mode="fused",
                        max_gradient_clip_norm=5, time_ahead=True)

# Validation data
validation_data = reader.read(reader.validation_data, time_ahead=brain_parameters["time_ahead"])

# Validation parameters
validation_parameters = brain_parameters.copy()
validation_parameters["dropout"] = [0, 0, 0]
validation_parameters["batch_dim"] = len(reader.validation_data)

# Agent
agent = Agent.Regressor(vision=Vision.Vision(brain=Brains.PD_LSTM_Model(brain_parameters)))

# Validation
validate = Agent.Regressor(vision=Vision.Vision(brain=Brains.PD_LSTM_Model(validation_parameters)),
                           session=agent.session, scope_name="validating")

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Learning Rate", "RMSE",
                                                    "Loss (MSE)", "Validation (MSE)"], description=brain_parameters,
                                      run_throughs_per_epoch=len(reader.training_data) // brain_parameters["batch_dim"])

# TensorBoard
agent.start_tensorboard(scalars={"Loss MSE": agent.loss}, gradients=agent.gradients, variables=agent.variables,
                        logging_interval=100, directory_name="Models/Logs/LSTMModel/time_ahead_and_sequence_dropout")
validate.start_tensorboard(scalars={"Validation MSE": validate.loss}, tensorboard_writer=agent.tensorboard_writer,
                           directory_name="Models/Logs/LSTMModel/time_ahead_and_sequence_dropout")

# Main method
if __name__ == "__main__":
    # Load agent
    if restore:
        agent.load("Models/Saved/LSTMModel/time_ahead_and_sequence_dropout/brain")

    # Training iterations
    for episode in range(1, 100000000000 + 1):
        # Learning rate
        learning_rate = 0.0001

        # Batch
        inputs, desired_outputs, time_dims, time_ahead = reader.iterate_batch(brain_parameters["batch_dim"])

        # Train
        loss = agent.learn({"inputs": inputs, "desired_outputs": desired_outputs, "time_dims": time_dims,
                            "learning_rate": learning_rate, "time_ahead": time_ahead})

        # Validate
        validation_mse = validate.measure_loss(validation_data) if performance.is_epoch(episode) else None

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Learning Rate": learning_rate,
                                         "RMSE": np.sqrt(loss), "Loss (MSE)": loss, "Validation (MSE)": validation_mse})

        # Display performance
        performance.output_performance(run_through=episode, special_aggregation={"Episode": lambda x: x[-1],
                                                                                 "Validation (MSE)": lambda x: x[-1]})

        # Save agent
        if performance.is_epoch(episode):
            agent.save("Models/Saved/LSTMModel/time_ahead_and_sequence_dropout/brain")
