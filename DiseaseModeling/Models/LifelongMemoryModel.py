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
path = "/Users/sam/Documents/Programming/ReinforcementLearningResearch/DiseaseModeling/Models"
model_directory = "LifelongMemoryModel/test"

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
                        max_time_dim=reader.max_num_records, num_layers=1, dropout=[0.2, 0, 0.5], mode="block",
                        max_gradient_clip_norm=5, time_ahead_downstream=False, time_ahead_midstream=True,
                        time_ahead_upstream=False, memory_embedding_dim=64, raw_input_context_vector=False,
                        visual_representation_context_vector=True)

# Attributes
attributes = {"concepts": brain_parameters["memory_embedding_dim"], "attributes": reader.desired_output_dim}

# Vision
vision = Vision.Vision(brain=Brains.PD_LSTM_Memory_Model(brain_parameters))

# Agent memory
agent_memory = Memories.Memories(capacity=reader.separate_time_dims(agent_memories["inputs"]).shape[0],
                                 attributes=attributes, vision=vision.adapt({"dropout": [0, 0, 0]}))

# Validation memory
validation_memory = agent_memory.adapt(capacity=reader.separate_time_dims(validation_memories["inputs"]).shape[0])

# Agent
agent = Agent.LifelongMemory(vision=vision, long_term_memory=agent_memory, attributes=attributes, k=32)

# Validation
validate = agent.adapt(vision=agent_memory.vision.adapt(), long_term_memory=validation_memory, scope_name="validating")

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Learning Rate",
                                                    "RMSE", "Loss (MSE)", "Validation (MSE)"],
                                      run_throughs_per_epoch=len(reader.training_data) // brain_parameters["batch_dim"],
                                      description=brain_parameters)

# TensorBoard
agent.start_tensorboard(scalars={"Loss MSE": agent.loss}, gradients=agent.gradients, variables=agent.variables,
                        logging_interval=10, directory_name="{}/Logs/{}".format(path, model_directory))
validate.start_tensorboard(scalars={"Validation MSE": validate.loss}, tensorboard_writer=agent.tensorboard_writer,
                           directory_name="{}/Logs/{}".format(path, model_directory))

# Main method
if __name__ == "__main__":
    # Load agent
    if restore:
        agent.load("{}/Saved/{}".format(path, model_directory))

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

        # Remember
        remember_concepts, remember_attributes = agent.remember({"inputs": inputs, "time_dims": time_dims,
                                                                 "time_ahead": time_ahead})

        # Train
        loss = agent.learn({"remember_concepts": remember_concepts, "remember_attributes": remember_attributes,
                            "desired_outputs": desired_outputs, "learning_rate": learning_rate})

        # Validation
        validation_mse = "processing..."
        if performance.is_epoch(episode, interval=10):
            # Remember
            remember_concepts, remember_attributes = validate.remember({"inputs": validation_data["inputs"],
                                                                        "time_dims": validation_data["time_dims"],
                                                                        "time_ahead": validation_data["time_ahead"]})

            # Validate
            validation_mse = validate.measure_errors({"remember_concepts": remember_concepts,
                                                    "remember_attributes": remember_attributes,
                                                    "desired_outputs": validation_data["desired_outputs"]})

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Learning Rate": learning_rate,
                                         "RMSE": np.sqrt(loss), "Loss (MSE)": loss, "Validation (MSE)": validation_mse})

        # Display performance
        performance.output_performance(run_through=episode, interval=10,
                                       special_aggregation={"Episode": lambda x: x[-1],
                                                            "Validation (MSE)": lambda x: x[-1]})

        # End of epoch
        if performance.is_epoch(episode, interval=10):
            # Save agent
            agent.save("{}/Saved/{}".format(path, model_directory))

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
