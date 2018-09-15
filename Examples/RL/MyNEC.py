from __future__ import division
import Performance
import Brains
import Vision
import Memories
import Agent
import numpy as np
import gym
import time
import datetime

# Restore saved agent
restore = False

# Model directory
path = "/Users/sam/Documents/Programming/ReinforcementLearningResearch/ReinforcementLearning/Models"
model_directory = "LifelongMemoryModel/testing"

# Environment
env_name = 'CartPole-v0'
env = gym.make(env_name)
actions = np.arange(env.action_space.n)
state_dim = env.observation_space.shape[0]

# Runner parameters
max_episode_length = 200
max_run_through_length = 200
epoch = 100

# Model parameters
num_similar_memories = 50
reward_discount = 0.999
representation_dim = 5
batch_dim = 128
trace_length = 200
memory_length = 400000
experience_replay_length = 100000
learning_rate = 0.0001

# Brain parameters
brain_parameters = dict(input_dim=state_dim, output_dim=10, max_gradient_clip_norm=5)

# Memory attributes
memory_attributes = dict(representation=representation_dim, action=1, reward=1, expectation=1, duplicate=1, terminal=1,
                         time_accessed=1, value=1)

# Experience replay attributes
experience_replay_attributes = dict(state=state_dim, value=1, time_accessed=1,
                                    remembered_representations=(1, num_similar_memories, representation_dim),
                                    remembered_attributes=(1, num_similar_memories, 1))

# Vision
vision = Vision.Vision(brain=Brains.FullyConnected(brain_parameters))

# Memories TODO replace target attribute with parameters for all start_brains when re-implementing brains
long_term_memory = [Memories.MFEC(capacity=memory_length, attributes=memory_attributes, target_attribute="value",
                                  num_similar_memories=num_similar_memories) for _ in actions]
short_term_memory = [Memories.MFEC(capacity=max_episode_length, attributes=memory_attributes) for _ in actions]

# Experience replay
experience_replay = Memories.Memories(capacity=experience_replay_length, attributes=experience_replay_attributes)

# Include all attributes
attributes = memory_attributes.copy()
attributes.update(experience_replay_attributes)

# Reward traces
traces = Memories.Traces(capacity=trace_length, attributes=attributes, reward_discount=reward_discount,
                         memories=short_term_memory, experience_replay=experience_replay)

# Agent
agent = Agent.NEC(vision=vision, long_term_memory=long_term_memory, short_term_memory=short_term_memory, traces=traces,
                  attributes=attributes, exploration_rate=1, actions=actions)

# File name
filename_prefix = "Agent"
filename = "{}_{}_{}___{}.csv".format(filename_prefix, env_name, datetime.datetime.today().strftime('%m_%d_%y'),
                                      datetime.datetime.now().strftime('%H_%M'))

# Initialize metrics for measuring performance
performance = Performance.Performance(['Run-Through', 'Episode', 'State', 'Number of Steps', 'Memory Size',
                                       'Number of Duplicates', 'K', 'Gamma', 'Epsilon', 'Max Episode Length',
                                       'Trace Length', 'Mean See Time', 'Mean Act Time', 'Mean Remember Time',
                                       'Mean Experience Time', 'Mean Learn Time', 'Mean Episode Time', 'Mean Reward'],
                                      epoch, filename=None)

# TensorFlow Partial Run (Note: leaks memory if any fetches un-used)
agent.start_tensorflow_partial_run(["representation", "expectation"],
                                   ["inputs", "remembered_representations", "remembered_attributes"])

# Main method
if __name__ == "__main__":
    # Initialize environment and measurement variables TODO move measure performance parameters and calls to performance
    state = env.reset()
    done = False
    run_through = 0
    total_steps = 0
    run_through_step = 0
    run_through_reward = 0
    see_times = []
    remember_times = []
    act_times = []
    experience_times = []
    learn_times = []
    episode_times = []

    # Train in batches of episodes
    for episode in range(100000):
        episode_start = time.time()

        # For every time step in episode
        for _ in range(max_episode_length):
            # Increment step
            total_steps += 1
            run_through_step += 1

            # Represent TODO add timers to other bodies
            representation = long_term_memory[0].represent(np.expand_dims(state, 0), do_partial_run=True)

            # Remember
            remembered = [(action_memories["representation"], action_memories["value"]) for action_memories in
                          [memory.retrieve(representation, agent.perception_of_time) for memory in long_term_memory]]
            remembered_representations = np.squeeze(np.stack([row[0] for row in remembered]), 1)
            remembered_attributes = np.squeeze(np.stack([row[1] for row in remembered]), 1)

            # Expect
            expectation = long_term_memory[0].expect({"remembered_representations": remembered_representations,
                                                      "remembered_attributes": remembered_attributes},
                                                     do_partial_run=True)

            # Set likelihood of picking a random action
            agent.exploration_rate = max(min(100000 / (episode + 1) ** 3, 1), 0.001)

            # Act
            action = agent.act(expectation)

            # Measure performance
            act_times += [agent.timer]

            # Transition
            state, reward, done, _ = env.step(action)

            # Episode done at terminal state or at max run-through length
            done = done or run_through_step > max_run_through_length

            # Measure performance
            run_through_reward += reward

            # Experience
            experience = {"state": state, "representation": representation, "expectation": expectation[action],
                          "remembered_representations": remembered_representations[action],
                          "remembered_attributes": remembered_attributes[action], "action": action, "duplicate": None,
                          "reward": reward, "value": None, "terminal": done, "time_accessed": None, "time": None}

            # Observe the experience such that traces and short term memory are updated
            agent.experience(experience)

            # Measure performance
            experience_times += [agent.timer]

            # Break at end of run-through
            if done:
                # Finish episode
                break

        # Experiences
        experiences = experience_replay.random_batch(batch_dim=batch_dim)
        experiences.update({"learning_rate": learning_rate,
                            "inputs": experiences["state"],
                            "desired_outputs": np.expand_dims(np.expand_dims(experiences["value"], 1), 1)})

        # Store short term memories in long term memory, re-build KD tree, and update agent's brain
        agent.learn(experiences)

        # Measure performance
        learn_times += [agent.timer]
        episode_times += [time.time() - episode_start]

        # Output and reset performance measures at the end of a run-through
        if done:
            # Increment run-through
            run_through += 1

            # Measure performance  TODO no means of empty slices
            metrics = {'Run-Through': run_through, 'Episode': episode + 1, 'State': total_steps,
                       'Number of Steps': run_through_step,
                       'Memory Size': sum([long_term_memory[a].length for a in actions]),
                       'Number of Duplicates': sum([long_term_memory[a].num_duplicates for a in actions]),
                       "K": long_term_memory[0].num_similar_memories,
                       "Gamma": traces.reward_discount, "Epsilon": round(agent.exploration_rate, 3),
                       'Max Episode Length': max_episode_length, 'Trace Length': trace_length,
                       'Mean See Time': np.mean(see_times), 'Mean Remember Time': np.mean(remember_times),
                       'Mean Act Time': np.mean(act_times), 'Mean Experience Time': np.mean(experience_times),
                       'Mean Learn Time': np.mean(learn_times), 'Mean Episode Time': np.mean(episode_times),
                       'Mean Reward': run_through_reward}
            performance.measure_performance(metrics)

            # Output performance per epoch
            performance.output_performance(run_through, env_name,
                                           special_aggregation={metric: lambda x: x[-1] for metric in
                                                                ["Run-Through", "Episode", "State", "Number of Steps",
                                                                 "Memory Size", "Number of Duplicates", "K", "Gamma",
                                                                 "Epsilon", "Max Episode Length", "Trace Length"]})

            # Reset environment and measurement variables
            state = env.reset()
            run_through_step = 0
            run_through_reward = 0
            see_times = []
            remember_times = []
            act_times = []
            experience_times = []
            learn_times = []
            episode_times = []
