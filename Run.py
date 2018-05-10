from Vision import Vision, RandomProjection
from Memories import Memories, Traces
from Agent import Agent
import numpy as np
import pandas as pd
import gym
import time
import datetime
import sys


# Display progress in console
class Progress:
    # Initialize progress measures
    progress_complete = 0.00
    progress_total = 0.00
    name = ""
    show = True

    def __init__(self, pc, pt, name, show):
        # Initialize variables
        self.progress_complete = pc
        self.progress_total = pt
        self.name = name
        self.show = show
        if self.show:
            sys.stdout.write("\rProgress: {:.2%} [{}]".format(0, name))
            sys.stdout.flush()

    def update_progress(self):
        # Update progress
        self.progress_complete += 1.00
        if self.show:
            sys.stdout.write("\rProgress: {:.2%} [{}]".format(self.progress_complete / self.progress_total, self.name))
            sys.stdout.flush()
        if (self.progress_complete == self.progress_total) and self.show:
            print("")


# Environment
# env_name = 'CartPole-v0'
# env = gym.make(env_name)
# action_space = np.arange(env.action_space.n)
# objects = None
# properties = None
# state_space = env.observation_space.shape[0]
# epoch = 100
# episode_length = 200
# trace_length = 200

# Environment
# env_name = 'Pong-v0'
# env = gym.make(env_name)
# action_space = np.arange(env.action_space.n)
# objects = 3
# scene_crop = [35, 18, 0, 0]
# scene_size = (80, 80)
# scene_scale = 10000.0
# scene_sigma = 0.001
# scene_min_size = 1
# epoch = 100
# episode_length = 200
# trace_length = 500
# state_space = objects * 5

# Environment
env_name = 'SpaceInvaders-v0'
env = gym.make(env_name)
action_space = np.arange(env.action_space.n)
objects = 44
scene_crop = [20, 15, 0, 0]
scene_size = (80, 80)
scene_scale = 10000.0
scene_sigma = 0.001
scene_min_size = 1
epoch = 100
episode_length = 200
trace_length = 500
state_space = objects * 5

# Attributes
attributes = {"num_attributes": 7, "action": -7, "reward": -6, "value": -5, "expected": -4, "duplicate": -3,
              "time_accessed": -2, "terminal": -1}

# Memory width
memory_width = state_space + attributes["num_attributes"]

# Initialize metrics for measuring performance
metrics = {'episode': [], 'memory_size': [], 'num_duplicates': [], 'see_time': [], 'act_time': [],
           'experience_time': [], 'learn_time': [], 'episode_time': [], 'reward': []}

# Initialize progress variable
progress = None

# Visual model
vision = Vision(object_capacity=objects, params=[scene_scale, scene_sigma, scene_min_size],
                crop=scene_crop, size=scene_size)
# vision = RandomProjection(64, None, True, True)

# Memories
long_term_memory = [Memories(capacity=1000000, width=memory_width, attributes=attributes) for _ in action_space]
short_term_memory = [Memories(capacity=episode_length, width=memory_width, attributes=attributes) for _ in action_space]

# Reward traces
traces = Traces(capacity=trace_length, width=memory_width, attributes=attributes, memories=short_term_memory,
                gamma=0.999)

# Agent
agent = Agent(vision=vision, long_term_memory=long_term_memory, short_term_memory=short_term_memory, traces=traces,
              attributes=attributes, actions=action_space, epsilon=1, k=50)

# Main method
if __name__ == "__main__":
    # Initialize environment
    state = env.reset()
    t = 0

    # Train in batches of episodes
    for episode in range(10000):
        # Performance metrics
        rewards = 0
        see_times = 0
        act_times = 0
        learn_times = 0
        experience_times = 0
        episode_start = time.time()

        # For every time step in episode
        for _ in range(episode_length):
            # Increment run-through time
            t += 1

            # Display environment
            # if 10 < t < 500:
            #     env.render()

            # Get scene from visual model
            scene = agent.see(state)

            # Show segmentation
            # if t > 150:
            #     agent.model.plot()

            # Measure performance
            see_times += agent.timer

            # Set likelihood of picking a random action
            agent.epsilon = max(min(100000 / (episode + 1) ** 3, 1), 0.001)

            # Get action
            action, expected, duplicate = agent.act(scene=scene)

            # Measure performance
            act_times += agent.timer

            # Transition
            state, reward, done, _ = env.step(action)

            # Measure performance
            rewards += reward

            # Observe the experience such that traces and short term memory are updated
            agent.experience(scene, action, reward, expected, duplicate, done)

            # Measure performance
            experience_times += agent.timer

            # Break at end of run-through
            if done:
                # Re-initialize environment
                t = 0
                state = env.reset()
                break

        # Store short term memories in long term memory and re-build KD tree
        agent.learn()

        # Measure performance
        learn_time = agent.timer
        episode_time = time.time() - episode_start
        metrics['episode'].append(episode)
        metrics['memory_size'].append(sum([long_term_memory[action].length for action in action_space]))
        metrics["num_duplicates"].append(sum([long_term_memory[action].num_duplicates for action in action_space]))
        metrics['see_time'].append(see_times)
        metrics['act_time'].append(act_times)
        metrics['experience_time'].append(experience_times)
        metrics['learn_time'].append(learn_time)
        metrics['episode_time'].append(episode_time)
        metrics['reward'].append(rewards)

        # Update progress
        if progress is not None:
            progress.update_progress()

        # Print and output performance metrics for epoch
        if not episode % epoch:
            # File name
            filename_suffix = "Run"
            filename = "Env_{}_Date_{}_{}".format(env_name, datetime.datetime.today().strftime('%m_%d_%y'),
                                                  filename_suffix)

            # Create file if first episode, else append metrics to existing file
            if episode > 0:
                # Print metrics
                print("Epoch {}, last {} episode reward average: {}".format(episode / epoch, epoch,
                                                                            np.mean(metrics['reward'][-epoch:])))
                print("* {} memories stored".format(metrics['memory_size'][-1]))
                print("* {} duplicates".format(metrics['num_duplicates'][-1]))
                print("* k is {}, gamma is {}, epsilon is {}, episode length is {}, trace length is {}".format(
                    agent.k, traces.gamma, agent.epsilon, episode_length, trace_length))
                print("* mean seeing time per episode: {}".format(np.mean(metrics['see_time'][-epoch:])))
                print("* mean acting time per episode: {}".format(np.mean(metrics['act_time'][-epoch:])))
                print("* mean experiencing time per episode: {}".format(np.mean(metrics['experience_time'][-epoch:])))
                print("* mean learning time per episode: {}".format(np.mean(metrics['learn_time'][-epoch:])))
                print("* mean episode time: {}\n".format(np.mean(metrics['episode_time'][-epoch:])))

                # Append metrics to existing file
                with open('Data/{}.csv'.format(filename), 'a') as data_file:
                    pd.DataFrame(data=metrics).to_csv(data_file, index=False, header=False,
                                                      columns=['episode', 'memory_size', 'num_duplicates', 'see_time',
                                                               'act_time', 'experience_time',
                                                               'learn_time', 'episode_time',
                                                               'reward'])
            else:
                # Create file
                pd.DataFrame(data=metrics).to_csv('Data/{}.csv'.format(filename), index=False,
                                                  columns=['episode', 'memory_size', 'num_duplicates', 'see_time',
                                                           'act_time', 'experience_time',
                                                           'learn_time', 'episode_time',
                                                           'reward'])

            # Reset metrics
            metrics = {'episode': [], 'memory_size': [], 'num_duplicates': [], 'see_time': [], 'act_time': [],
                       'experience_time': [], 'learn_time': [], 'episode_time': [], 'reward': []}

            # Initiate progress
            progress = Progress(0, epoch, "Epoch", True)
