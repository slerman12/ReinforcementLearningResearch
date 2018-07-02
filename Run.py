from __future__ import division
import Performance
import Vision
import Memories
import Agent
import numpy as np
import gym
import time
import datetime

# Environment
env_name = 'CartPole-v0'
env = gym.make(env_name)
action_space = np.arange(env.action_space.n)
state_space = env.observation_space.shape[0]
max_episode_length = 200
max_run_through_length = 200
trace_length = 200
epoch = 100
vision = None

# Environment
# env_name = 'Pong-v0'
# env = gym.make(env_name)
# action_space = np.arange(env.action_space.n)
# objects = 3
# crop = [35, 18, 0, 0]
# size = (80, 80)
# scale = 10000.0
# sigma = 0.001
# min_size = 1
# epoch = 5
# max_run_through_length = 100000
# max_episode_length = 10000
# trace_length = 250
# trajectory = True
# state_space = objects * 5 if trajectory else objects * 3

# Environment
# env_name = 'Riverraid-v0'
# env = gym.make(env_name)
# action_space = np.arange(env.action_space.n)
# objects = 30
# crop = [5, 50, 10, 10]
# size = (70, 90)
# scale = 10000.0
# sigma = 0.001
# min_size = 1
# epoch = 5
# max_run_through_length = 1000000
# episode_length = 250
# trace_length = 250
# trajectory = True
# state_space = objects * 5 if trajectory else objects * 3

# Environment
# env_name = 'Breakout-v0'
# env = gym.make(env_name)
# action_space = np.arange(env.action_space.n)
# objects = 30
# crop = [40, 20, 10, 10]
# size = (80, 80)
# scale = 10000.0
# sigma = 0.001
# min_size = 1
# epoch = 5
# max_run_through_length = 1000000
# episode_length = 250
# trace_length = 250
# trajectory = True
# state_space = objects * 5 if trajectory else objects * 3

# Environment
# env_name = 'SpaceInvaders-v0'
# env = gym.make(env_name)
# action_space = np.arange(env.action_space.n)
# objects = 44
# crop = [20, 15, 0, 0]
# size = (80, 80)
# scale = 10000.0
# sigma = 0.001
# min_size = 1
# epoch = 3
# max_run_through_length = 10000
# episode_length = 250
# trace_length = 250
# trajectory = True
# state_space = objects * 5 if trajectory else objects * 3

# Environment
# env_name = 'Bowling-v0'
# env = gym.make(env_name)
# action_space = np.arange(env.action_space.n)
# objects = 20
# crop = [110, 40, 0, 0]
# size = (100, 40)
# scale = 900
# sigma = 0.03
# min_size = 1
# epoch = 5
# max_run_through_length = 1000000
# episode_length = 250
# trace_length = 250
# trajectory = False
# state_space = objects * 5 if trajectory else objects * 3

# Environment
# env_name = 'MsPacman-v0'
# env = gym.make(env_name)
# action_space = np.arange(env.action_space.n)
# objects = 300
# crop = [5, 40, 0, 0]
# size = (80, 80)
# scale = 10000.0
# sigma = 0.001
# min_size = 1
# epoch = 3
# max_run_through_length = 10000
# episode_length = 250
# trace_length = 250
# state_space = objects * 5

# Visual model
# vision = Vision(object_capacity=objects, params=[scale, sigma, min_size], crop=crop, size=size, trajectory=trajectory)
# vision = RandomProjection(dimension=64, flatten=True, size=size, greyscale=True, crop=crop)
# state_space = 64

# Attributes
attributes = dict(state=state_space, action=1, reward=1, value=1, expected=1, duplicate=1, terminal=1, time_accessed=1)

# Memories
long_term_memory = [Memories.MFEC(capacity=400000, attributes=attributes) for _ in action_space]
short_term_memory = [Memories.MFEC(capacity=max_episode_length, attributes=attributes) for _ in action_space]

# Reward traces
traces = Memories.Traces(capacity=trace_length, attributes=attributes, memories=short_term_memory, gamma=0.999)

# Agent TODO: add Memory Agent (policy-based memory rather than value), prioritized experience, & value updates
agent = Agent.Agent(vision=vision, long_term_memory=long_term_memory, short_term_memory=short_term_memory,
                    traces=traces, attributes=attributes, actions=action_space, epsilon=1, k=50)

# File name
filename_prefix = "Segmentation"
filename = "Results/{}_{}_{}___{}.csv".format(filename_prefix, env_name, datetime.datetime.today().strftime('%m_%d_%y'),
                                              datetime.datetime.now().strftime('%H_%M'))

# Initialize metrics for measuring performance TODO: Add model saving
performance = Performance.Performance(['Run-Through', 'Episode', 'State', 'Number of Steps', 'Memory Size',
                                       'Number of Duplicates', 'K', 'Gamma', 'Epsilon', 'Max Episode Length',
                                       'Trace Length', 'Mean See Time', 'Mean Act Time', 'Mean Experience Time',
                                       'Mean Learn Time', 'Mean Episode Time', 'Reward'], epoch, filename)

# Main method
if __name__ == "__main__":
    # Initialize environment and measurement variables
    state = env.reset()
    done = False
    run_through = 0
    total_steps = 0
    run_through_step = 0
    run_through_reward = 0
    see_times = []
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

            # Display environment
            # if run_through > 5:
            # env.render()

            # Get scene from visual model
            scene = agent.see(state)

            # Show segmentation
            # if t > 90:
            #     agent.vision.plot()

            # Measure performance
            see_times += [agent.timer]

            # Set likelihood of picking a random action
            agent.epsilon = max(min(100000 / (episode + 1) ** 3, 1), 0.001)
            # agent.epsilon = max(1 / (episode + 1), 0.001)

            # Get action
            action, expected, duplicate = agent.act(scene=scene)

            # Measure performance
            act_times += [agent.timer]

            # Transition
            state, reward, done, _ = env.step(action)

            # Episode done at terminal state or at max run-through length
            done = done or run_through_step > max_run_through_length

            # Measure performance
            run_through_reward += reward

            # Experience
            experience = {"state": scene, "action": action, "reward": reward, "value": None, "expected": expected,
                          "duplicate": duplicate, "terminal": done, "time_accessed": None, "time": None}

            # Observe the experience such that traces and short term memory are updated
            agent.experience(experience)

            # Measure performance
            experience_times += [agent.timer]

            # Break at end of run-through
            if done:
                # Finish episode
                break

        # Store short term memories in long term memory and re-build KD tree
        agent.learn()

        # Measure performance
        learn_times += [agent.timer]
        episode_times += [time.time() - episode_start]

        # Output and reset performance measures at the end of a run-through
        if done:
            # Increment run-through
            run_through += 1

            # Measure performance
            metrics = {'Run-Through': run_through, 'Episode': episode + 1, 'State': total_steps,
                       'Number of Steps': run_through_step,
                       'Memory Size': sum([long_term_memory[a].length for a in action_space]),
                       'Number of Duplicates': sum([long_term_memory[a].num_duplicates for a in action_space]),
                       "K": agent.k, "Gamma": traces.gamma, "Epsilon": round(agent.epsilon, 3),
                       'Max Episode Length': max_episode_length, 'Trace Length': trace_length,
                       'Mean See Time': np.mean(see_times), 'Mean Act Time': np.mean(act_times),
                       'Mean Experience Time': np.mean(experience_times), 'Mean Learn Time': np.mean(learn_times),
                       'Mean Episode Time': np.mean(episode_times), 'Reward': run_through_reward}
            performance.measure_performance(metrics)

            # Output performance per epoch
            performance.output_performance(run_through, env_name)

            # Reset environment and measurement variables
            state = env.reset()
            run_through_step = 0
            run_through_reward = 0
            see_times = []
            act_times = []
            experience_times = []
            learn_times = []
            episode_times = []
