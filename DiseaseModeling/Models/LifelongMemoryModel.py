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

# Model directory
model_directory = "LifelongMemoryModel/time_ahead"

# Data reader
reader = Data.ReadPD("../Data/Processed/encoded.csv", targets=["UPDRS_I", "UPDRS_II", "UPDRS_III"],
                     train_test_split=0.8, train_memory_split=0.75, valid_eval_split=1, sequence_dropout=0)

# Brain parameters
brain_parameters = dict(batch_dim=32, input_dim=reader.input_dim, output_dim=128,
                        max_time_dim=reader.max_num_records, num_layers=1, dropout=[0.2, 0, 0.65], mode="block",
                        max_gradient_clip_norm=5, time_ahead=True)

# Attributes
attributes = {"concepts": brain_parameters["output_dim"], "attributes": reader.desired_output_dim}

# Memory data
memory_data = reader.read(reader.memory_data)

# Memory representation parameters
memory_representation_parameters = brain_parameters.copy()
memory_representation_parameters["dropout"] = [0, 0, 0]
memory_representation_parameters["batch_dim"] = memory_data["inputs"].shape[0]

# Memory
memory = Memories.Memories(capacity=reader.separate_time_dims(memory_data["inputs"]).shape[0], attributes=attributes)

# TODO
# memory = Memories.Memories(capacity, attributes, brain=vision.adapt(bla)) (vision just calls brain.adapt)
# No memory_represent; initialize memory brain session in Agent
# memories = memory.represent(memory_data)

# Agent
agent = Agent.LifelongMemory(vision=Vision.Vision(brain=Brains.PD_LSTM_Memory_Model(brain_parameters)),
                             long_term_memory=memory, attributes=attributes, k=2)

# Memory representation
memory_represent = Agent.Agent(
    vision=Vision.Vision(brain=Brains.PD_LSTM_Memory_Model(memory_representation_parameters)), attributes=attributes,
    session=agent.session, scope_name="memory_representation")

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Learning Rate", "RMSE", "Loss (MSE)"],
                                      run_throughs_per_epoch=len(reader.training_data) // brain_parameters["batch_dim"],
                                      description=brain_parameters)

# TensorBoard
agent.start_tensorboard(scalars={"Loss MSE": agent.loss}, gradients=agent.gradients, variables=agent.variables,
                        logging_interval=10, directory_name="Logs/LifelongMemoryModel/time_ahead")

# Main method
if __name__ == "__main__":
    # Load agent
    if restore:
        agent.load("Saved/LifelongMemoryModel/time_ahead/brain")

    # Memory representations
    memories = memory_represent.see(memory_data)

    # Populate memory with representations
    memory.reset(population={"concepts": reader.separate_time_dims(memories),
                             "attributes": reader.separate_time_dims(memory_data["desired_outputs"])})

    # Consolidate memories (build KD tree for them)
    memory.consolidate()

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
                            "desired_outputs": desired_outputs, "time_ahead": time_ahead,
                            "learning_rate": learning_rate})

        # Measure performance
        performance.measure_performance({"Episode": episode, "Learn Time": agent.timer, "Learning Rate": learning_rate,
                                         "RMSE": np.sqrt(loss), "Loss (MSE)": loss})

        # Display performance
        performance.output_performance(run_through=episode, special_aggregation={"Episode": lambda x: x[-1]})

        # End of epoch
        if performance.is_epoch(episode):
            # Save agent
            agent.save("Saved/LifelongMemoryModel/time_ahead/brain")

            # Shuffle training/memory split
            reader.shuffle_training_memory_split()

            # Read new memory data
            memory_data = reader.read(reader.memory_data)

            # Memory representations
            memories = memory_represent.see(memory_data)

            # Populate memory with representations
            memory.reset(population={"concepts": reader.separate_time_dims(memories),
                                     "attributes": reader.separate_time_dims(memory_data["desired_outputs"])})

            # Consolidate memories (build KD tree for them)
            memory.consolidate()
