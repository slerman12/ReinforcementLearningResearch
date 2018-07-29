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
reader = Data.ReadPD("../Data/Processed/encoded.csv", targets=["UPDRS_I", "UPDRS_II", "UPDRS_III"],
                     train_test_split=0.7, valid_eval_split=0.33, sequence_dropout=False)

# Brain parameters
brain_parameters = dict(batch_dim=32, input_dim=reader.input_dim, hidden_dim=128, output_dim=reader.desired_output_dim,
                        max_time_dim=reader.max_num_records, num_layers=1, dropout=[0.2, 0, 0.65], mode="fused",
                        max_gradient_clip_norm=5, time_ahead=True)

# Attributes
attributes = {"concepts": brain_parameters["output_dim"], "attributes": reader.desired_output_dim}

# Memory data
memory_data = reader.read(reader.evaluation_data, time_dims_separated=True)

# Memory representation parameters
memory_representation_parameters = brain_parameters.copy()
memory_representation_parameters["dropout"] = [0, 0, 0]
memory_representation_parameters["batch_dim"] = len(memory_data["inputs"])
del memory_representation_parameters["max_time_dim"]

# Memory
memory = Memories.Memories(capacity=len(reader.evaluation_data), attributes=attributes)

# Agent
agent = Agent.LifelongMemory(vision=Vision.Vision(brain=Brains.PD_LSTM_Memory_Model(brain_parameters)),
                             long_term_memory=memory, attributes=attributes)

# Memory representation
memory_represent = Agent.LifelongMemory(
    vision=Vision.Vision(brain=Brains.PD_LSTM_Memory_Model(memory_representation_parameters)), attributes=attributes,
    session=agent.session, scope_name="memory_representation")

# Initialize metrics for measuring performance
performance = Performance.Performance(metric_names=["Episode", "Learn Time", "Learning Rate", "RMSE", "Loss (MSE)"],
                                      run_throughs_per_epoch=len(reader.training_data) // brain_parameters["batch_dim"],
                                      description=brain_parameters)

# TensorBoard
agent.start_tensorboard(scalars={"Loss MSE": agent.loss}, gradients=agent.gradients, variables=agent.variables)

# Main method
if __name__ == "__main__":
    # Load agent
    if restore:
        agent.load()

    # Memory representations (capacity x time x representation)
    memories = memory_represent.see(memory_data)

    # Populate memory with representations
    memory.reset(population=memories)

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
        remember_concepts, remember_attributes = agent.remember(scene, brain_parameters["batch_dims"],
                                                                brain_parameters["max_time_dim"], time_dims)

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
            agent.save()

            # Memory representations
            memories = memory_represent.see(memory_data)

            # Replace memories with new representations
            memory.reset(population=memories)

            # Consolidate memories (build KD tree for them)
            memory.consolidate()
