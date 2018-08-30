from __future__ import division
import Performance
import Agent
import Vision
import Brains
from DiseaseModeling.Data import Data
import numpy as np

# Restore saved agent
restore = False

# Model directory
path = "/Users/sam/Documents/Programming/ReinforcementLearningResearch/DiseaseModeling/Models"
model_directory = "LSTMModel/time_ahead_downstream_with_dropout_0.2_0_0.65"

# Data reader
reader = Data.ReadPD("Data/Processed/encoded.csv", targets=["UPDRS_I", "UPDRS_II", "UPDRS_III", "MSEADLG"],
                     train_test_split=0.8, valid_eval_split=1, sequence_dropout=0)

# Brain parameters
brain_parameters = dict(batch_dim=32, input_dim=reader.input_dim, hidden_dim=128, output_dim=reader.desired_output_dim,
                        max_time_dim=reader.max_num_records, num_layers=1, dropout=[0.2, 0, 0.65],
                        mode="block", max_gradient_clip_norm=5, time_ahead_downstream=True)

# Validation data
validation_data = reader.read(reader.validation_data)

# Vision
vision = Vision.Vision(brain=Brains.PD_LSTM_Model(brain_parameters))

# Agent
agent = Agent.Regressor(vision=vision)

# Validation
validate = agent.adapt(vision=vision.adapt({"dropout": [0, 0, 0], "batch_dim": len(reader.validation_data)}),
                       scope_name="validating")

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Learning Rate", "RMSE",
                                                    "Loss (MSE)", "Validation (MSE)"], description=brain_parameters,
                                      run_throughs_per_epoch=len(reader.training_data) // brain_parameters["batch_dim"])

# TensorBoard
# agent.start_tensorboard(scalars={"Loss MSE": agent.loss}, gradients=agent.gradients, variables=agent.variables,
#                         logging_interval=100, directory_name="{}/Logs/{}".format(path, model_directory))
# validate.start_tensorboard(scalars={"Validation MSE": validate.loss}, tensorboard_writer=agent.tensorboard_writer,
#                            directory_name="{}/Logs/{}".format(path, model_directory))

# Main method
if __name__ == "__main__":
    # Load agent
    if restore:
        agent.load("{}/Saved/{}".format(path,model_directory))

    # Training iterations
    for episode in range(1, 250000 + 1):
        # Learning rate
        learning_rate = 0.0001

        # Batch
        inputs, desired_outputs, time_dims, time_ahead = reader.iterate_batch(brain_parameters["batch_dim"])

        # Train
        loss = agent.learn({"inputs": inputs, "desired_outputs": desired_outputs, "time_dims": time_dims,
                            "learning_rate": learning_rate, "time_ahead": time_ahead})

        # Validate
        validation_mse = "processing"
        if performance.is_epoch(episode, interval=10):
            validation_mse = validate.measure_errors(validation_data) if performance.is_epoch(episode) else None

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Learning Rate": learning_rate,
                                         "RMSE": np.sqrt(loss), "Loss (MSE)": loss, "Validation (MSE)": validation_mse})

        # Display performance
        performance.output_performance(run_through=episode, interval=10,
                                       special_aggregation={"Episode": lambda x: x[-1],
                                                            "Validation (MSE)": lambda x: x[-1]})

        # Save agent
        if performance.is_epoch(episode):
            agent.save("{}/Saved/{}".format(path, model_directory))
