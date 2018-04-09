import numpy as np


def expected_reward(self, scene, action, memory, k):
    print()


class Memory:
    memory = np.array([])

    def Update(self, scene, action, reward, reward_horizon, memory_horizon):
        print()

    # Merge and clear
    def Merge(self, m):
        print()


class Agent:
    local_memory = Memory()
    model = SHVS()

    def __init__(self, global_memory, actions, reward_horizon, memory_horizon, k):
        self.global_memory = global_memory
        self.actions = actions
        self.reward_horizon = reward_horizon
        self.memory_horizon = memory_horizon
        self.k = k

    def Model(self, state):
        return self.model.Update(state)

    def Act(self, scene):
        print()

    def Learn(self, scene, action, reward):
        self.local_memory.Update(scene, action, reward, self.reward_horizon, self.memory_horizon)

    def Finish(self):
        self.global_memory.Merge(self.local_memory)
        self.model.Finish()


# Streaming hierarchical video segmentation
class SHVS:
    def __init__(self):
        print()

    def Update(self, state):
        return 10

    def Finish(self):
        print()


# Main method
if __name__ == "__main__":
    print()
