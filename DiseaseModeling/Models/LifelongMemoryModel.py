from __future__ import division
import Performance
import Agent
import Vision
import Memories
import Brains
from DiseaseModeling.Data import Data
import numpy as np

# Restore saved agent
restore = False

# Model directory
model_directory = "LifelongMemoryModel/no_final_dense_layer"

# Data reader
reader = Data.ReadPD("../Data/Processed/encoded.csv", targets=["UPDRS_I", "UPDRS_II", "UPDRS_III"],
                     train_test_split=0.8, train_memory_split=0.5, valid_eval_split=1, sequence_dropout=0)

# Validation data
validation_data = reader.read(reader.validation_data)

# Memories
agent_memories = reader.read(reader.memory_data)
validation_memories = reader.read(reader.training_memory_data)

# Brain parameters
brain_parameters = dict(batch_dim=32, input_dim=reader.input_dim, output_dim=128,
                        max_time_dim=reader.max_num_records, num_layers=1, dropout=[0, 0, 0], mode="block",
                        max_gradient_clip_norm=5, time_ahead_downstream=True, time_ahead_upstream=True)

# Attributes
attributes = {"concepts": brain_parameters["output_dim"], "attributes": reader.desired_output_dim}

# Vision
vision = Vision.Vision(brain=Brains.PD_LSTM_Memory_Model(brain_parameters))

# Agent memory
agent_memory = Memories.Memories(capacity=reader.separate_time_dims(agent_memories["inputs"]).shape[0],
                                 attributes=attributes,
                                 vision=vision.adapt({"dropout": [0, 0, 0],
                                                      "batch_dim": agent_memories["inputs"].shape[0]}))

# Validation memory
validation_memory = agent_memory.adapt(capacity=reader.separate_time_dims(validation_memories["inputs"]).shape[0],
                                       vision=vision.adapt({"dropout": [0, 0, 0],
                                                            "batch_dim": len(reader.training_memory_data)}))

# Agent
agent = Agent.LifelongMemory(vision=vision, long_term_memory=agent_memory, attributes=attributes, k=10)

# Validation
validate = agent.adapt(vision=agent.vision.adapt({"dropout": [0, 0, 0], "batch_dim": len(reader.validation_data)}),
                       long_term_memory=validation_memory, scope_name="validating")

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Learning Rate",
                                                    "RMSE", "Loss (MSE)", "Validation (MSE)"],
                                      run_throughs_per_epoch=len(reader.training_data) // brain_parameters["batch_dim"],
                                      description=brain_parameters)

print(agent.gradients)

# TensorBoard
agent.start_tensorboard(scalars={"Loss MSE": agent.loss}, gradients=agent.gradients, variables=agent.variables,
                        logging_interval=10, directory_name="Logs/{}".format(model_directory))
validate.start_tensorboard(scalars={"Validation MSE": validate.loss}, tensorboard_writer=agent.tensorboard_writer,
                           directory_name="Logs/{}".format(model_directory))

# Main method
if __name__ == "__main__":
    # Load agent
    if restore:
        agent.load("Saved/{}".format(model_directory))

    # Populate memory with representations
    agent_memory.reset(
        population={"concepts": reader.separate_time_dims(agent_memory.represent(agent_memories)),
                    "attributes": reader.separate_time_dims(agent_memories["desired_outputs"])})
    validation_memory.reset(
        population={"concepts": reader.separate_time_dims(validation_memory.represent(validation_memories)),
                    "attributes":  reader.separate_time_dims(validation_memories["desired_outputs"])})

    # Consolidate memories (build KD tree for them)
    agent_memory.consolidate()
    validation_memory.consolidate()

    # Training iterations
    for episode in range(1, 100000000000 + 1):
        # Learning rate
        learning_rate = 0.0001

        # Batch
        inputs, desired_outputs, time_dims, time_ahead = reader.iterate_batch(brain_parameters["batch_dim"])

        # See
        scene = agent.see({"inputs": inputs, "time_dims": time_dims})

        # Remember
        remember_concepts, remember_attributes = agent.remember(scene, time_dims=time_dims)

        # Train
        loss = agent.learn({"remember_concepts": remember_concepts, "remember_attributes": remember_attributes,
                            "desired_outputs": desired_outputs, "learning_rate": learning_rate,
                            "time_ahead": time_ahead})

        # Validation
        validation_mse = "processing..."
        if performance.is_epoch(episode, interval=5):
            # See
            scene = validate.see(validation_data)

            # Remember
            remember_concepts, remember_attributes = validate.remember(scene, time_dims=validation_data["time_dims"])

            # Validate
            validation_mse = validate.measure_loss({"remember_concepts": remember_concepts,
                                                    "remember_attributes": remember_attributes,
                                                    "desired_outputs": validation_data["desired_outputs"],
                                                    "time_ahead": validation_data["time_ahead"]})

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Learning Rate": learning_rate,
                                         "RMSE": np.sqrt(loss), "Loss (MSE)": loss, "Validation (MSE)": validation_mse})

        # Display performance
        performance.output_performance(run_through=episode, special_aggregation={"Episode": lambda x: x[-1],
                                                                                 "Validation (MSE)": lambda x: x[-1]})

        # End of epoch
        if performance.is_epoch(episode, interval=5):
            # Save agent
            agent.save("Saved/{}".format(model_directory))

            # Shuffle training/memory split
            reader.shuffle_training_memory_split()

            # Read new memory data
            agent_memories = reader.read(reader.memory_data)

            # Populate memory with representations
            agent_memory.reset(
                population={"concepts": reader.separate_time_dims(agent_memory.represent(agent_memories)),
                            "attributes": reader.separate_time_dims(agent_memories["desired_outputs"])})
            validation_memory.reset(
                population={"concepts": reader.separate_time_dims(validation_memory.represent(validation_memories)),
                            "attributes":  reader.separate_time_dims(validation_memories["desired_outputs"])})

            # Consolidate memories (build KD tree for them)
            agent_memory.consolidate()
            validation_memory.consolidate()
