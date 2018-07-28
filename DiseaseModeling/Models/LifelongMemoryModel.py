from __future__ import division
import Performance
import Agent
import Vision
import Memories
import Brains
from DiseaseModeling.Data import Data
import numpy as np

# Restore saved agent
restore = True

# Data reader
reader = Data.ReadPD("Data/Processed/encoded.csv", targets=["UPDRS_I", "UPDRS_II", "UPDRS_III"], train_test_split=0.7,
                     valid_eval_split=0.33, sequence_dropout=False)

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

# Attributes
attributes = {"concepts": brain_parameters["output_dim"], "attributes": reader.desired_output_dim}

# Memory
memory = Memories.Memories(capacity=len(reader.evaluation_data), attributes=attributes)

# Agent
agent = Agent.Regressor(vision=Vision.Vision(brain=Brains.PD_LSTM_Model(brain_parameters)), long_term_memory=memory,
                        attributes=attributes)

# Validation
validate = Agent.Regressor(vision=Vision.Vision(brain=Brains.PD_LSTM_Model(validation_parameters)),
                           session=agent.session, scope_name="validating")

# Memory representation parameters
memory_representation_parameters = brain_parameters.copy()
memory_representation_parameters["dropout"] = [0, 0, 0]
memory_representation_parameters["batch_dim"] = memory.capacity

# Memory data
memory_data = reader.read(reader.evaluation_data)

# Memory representation
memory_representation = Agent.Regressor(session=agent.session, scope_name="memory_representation", vision=Vision.Vision(
    brain=Brains.PD_LSTM_Model(memory_representation_parameters)))

# Initial memories TODO untile the time dims
memories = memory_representation.see(memory_data)

# Populate memory with memories
memory.reset(population=memories)

# Consolidate memory
memory.consolidate()

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Learning Rate", "RMSE",
                                                    "Loss (MSE)", "Validation (MSE)"], description=brain_parameters,
                                      run_throughs_per_epoch=len(reader.training_data) // brain_parameters["batch_dim"])

# TensorBoard
agent.start_tensorboard(scalars={"Loss MSE": agent.loss}, gradients=agent.gradients, variables=agent.variables)
validate.start_tensorboard(scalars={"Validation MSE": validate.loss}, tensorboard_writer=agent.tensorboard_writer)

# Main method
if __name__ == "__main__":
    # Load agent
    if restore:
        agent.load()

    # Training iterations
    for episode in range(1, 100000000000 + 1):
        # Learning rate
        learning_rate = 0.0001

        # Batch
        inputs, desired_outputs, time_dims, time_ahead = reader.iterate_batch(brain_parameters["batch_dim"])

        # See
        scene = agent.see({"inputs": inputs, "time_dims": time_dims})

        # Remember
        remember_concepts, remember_attributes = agent.remember(scene, brain_parameters["batch_dims"],
                                                                brain_parameters["time_dims"])

        # Train
        loss = agent.learn({"remember_concepts": remember_concepts, "remember_attributes": remember_attributes,
                            "desired_outputs": desired_outputs, "time_ahead": time_ahead,
                            "learning_rate": learning_rate})

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Learning Rate": learning_rate,
                                         "RMSE": np.sqrt(loss), "Loss (MSE)": loss})

        # Display performance
        performance.output_performance(run_through=episode, special_aggregation={"Episode": lambda x: x[-1],
                                                                                 "Validation (MSE)": lambda x: x[-1]})

        # End of epoch
        if performance.is_epoch(episode):
            # Save agent
            agent.save()

            # Update memories
            memories = memory_representation.see(memory_data)

            # Populate memory with memories
            memory.reset(population=memories)

            # Consolidate memory
            memory.consolidate()
