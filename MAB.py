import numpy as np

class Epsilon_Greedy():
    def __init__(self, bandit, eps):
        self.bandit = bandit
        self.num_arms = bandit.num_arms
        self.Q_a = np.zeros(self.num_arms)
        self.N_a = np.zeros(self.num_arms)
        self.total_reward = 0
        self.eps = eps

    def play(self, max_steps):
        self.rewards = np.zeros(max_steps)
        num_steps = 0
        while num_steps < max_steps:
            arm_to_pull = self.choose(self.eps)
            reward = self.bandit.pull(arm_to_pull)
            self.total_reward += reward
            self.rewards[num_steps] = reward
            self.N_a[arm_to_pull] += 1
            self.Q_a[arm_to_pull] = self.Q_a[arm_to_pull] + (1 / self.N_a[arm_to_pull]) * (reward - self.Q_a[arm_to_pull])
            num_steps += 1
        return self.rewards, self.total_reward

    def choose(self, eps):
        p = np.random.uniform(0, 1)
        if(p > eps):
            arm_to_pull = np.argmax(self.Q_a)
        else:
            arm_to_pull = np.random.randint(0, self.bandit.num_arms, 1)[0]

        return arm_to_pull