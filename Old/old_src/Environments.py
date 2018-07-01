import gym
import numpy as np


class Environments:
    def __init__(self, env_name):
        self.env_name = env_name

        # Environment
        if env_name == 'CartPole-v0':
            self.env = gym.make(env_name)
            self.action_space = np.arange(self.env.action_space.n)
            self.objects = None
            self.properties = None
            self.state_space = self.env.observation_space.shape[0]
            self.episode_length = 200
            self.max_run_through_length = 200
            self.trace_length = 200
            self.vision = None
            self.epoch = 100

        # Environment
        if env_name == 'Pong-v0':
            self.env = gym.make(env_name)
            self.action_space = np.arange(self.env.action_space.n)
            self.objects = 3
            self.crop = [35, 18, 0, 0]
            self.size = (80, 80)
            self.scale = 10000.0
            self.sigma = 0.001
            self.min_size = 1
            self.epoch = 5
            self.max_run_through_length = 100000
            self.max_episode_length = 10000
            self.trace_length = 250
            self.trajectory = True
            self.state_space = self.objects * 5 if self.trajectory else self.objects * 3

        # Environment
        if env_name == 'Riverraid-v0':
            self.env = gym.make(env_name)
            self.action_space = np.arange(self.env.action_space.n)
            self.objects = 30
            self.crop = [5, 50, 10, 10]
            self.size = (70, 90)
            self.scale = 10000.0
            self.sigma = 0.001
            self.min_size = 1
            self.epoch = 5
            self.max_run_through_length = 1000000
            self.episode_length = 250
            self.trace_length = 250
            self.trajectory = True
            self.state_space = self.objects * 5 if self.trajectory else self.objects * 3

        # Environment
        env_name = 'Breakout-v0'
        env = gym.make(env_name)
        action_space = np.arange(env.action_space.n)
        objects = 30
        crop = [40, 20, 10, 10]
        size = (80, 80)
        scale = 10000.0
        sigma = 0.001
        min_size = 1
        epoch = 5
        max_run_through_length = 1000000
        episode_length = 250
        trace_length = 250
        trajectory = True
        state_space = objects * 5 if trajectory else objects * 3

        # Environment
        env_name = 'SpaceInvaders-v0'
        env = gym.make(env_name)
        action_space = np.arange(env.action_space.n)
        objects = 44
        crop = [20, 15, 0, 0]
        size = (80, 80)
        scale = 10000.0
        sigma = 0.001
        min_size = 1
        epoch = 3
        max_run_through_length = 10000
        episode_length = 250
        trace_length = 250
        trajectory = True
        state_space = objects * 5 if trajectory else objects * 3

        # Environment
        env_name = 'Bowling-v0'
        env = gym.make(env_name)
        action_space = np.arange(env.action_space.n)
        objects = 20
        crop = [110, 40, 0, 0]
        size = (100, 40)
        scale = 900
        sigma = 0.03
        min_size = 1
        epoch = 5
        max_run_through_length = 1000000
        episode_length = 250
        trace_length = 250
        trajectory = False
        state_space = objects * 5 if trajectory else objects * 3

        # Environment
        env_name = 'MsPacman-v0'
        env = gym.make(env_name)
        action_space = np.arange(env.action_space.n)
        objects = 300
        crop = [5, 40, 0, 0]
        size = (80, 80)
        scale = 10000.0
        sigma = 0.001
        min_size = 1
        epoch = 3
        max_run_through_length = 10000
        episode_length = 250
        trace_length = 250
        state_space = objects * 5