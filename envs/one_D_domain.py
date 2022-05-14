import numpy as np
import itertools

class One_D_Domain(object):
    def __init__(self,domain_size,reward_grid,bound,states,actions,iterations,seed):
        self.domain_size=domain_size
        self.reward_grid=reward_grid
        self.bound=bound
        self.states=states
        self.actions=actions
        self.MC_iterations=iterations
        self.seed=seed

    def next_state(self,state, action):
        if state < 0 and state >= -self.bound + 2:  # lift should not be before reward
            next = state - 1
        elif state > 0 and state <= self.bound - 2:  # lift should not be before reward
            next = state + 1
        else:
            next = state + action
        return next

    def optimal_policy(self):
        """
        immediately go to the closest corner
        :return:
        """

        policy = [[0.00, 1.00] for i in range(self.domain_size)]
        return policy

    def monte_carlo_eval(self,policy):
        G = 0
        trajectories = []

        for k in range(self.MC_iterations):
            if k % 10000 == 0:
                print("iteration ", k, "/", self.MC_iterations)
            trajectory, G_ = self.run_MDP(policy, seed=k+self.seed)
            G += G_
            trajectories.append(trajectory)
        return trajectories, G / self.MC_iterations

    def run_MDP(self,policy, seed):
        """
        run the MDP and return the cost
        :param policy:
        :return:
        """
        G = 0
        np.random.seed(seed)
        state = 0
        trajectory = []
        while True:
            a = np.random.choice(list(range(len(self.actions))), p=policy[state])
            action = self.actions[a]
            # print("state ", state)
            # print("action ", action)

            next_s = self.next_state(state, action)
            state_index = self.states.index(next_s)
            reward = self.reward_grid[state_index]
            # print("reward ", reward)
            G += reward

            trajectory.append((state, a, reward))
            if np.abs(next_s) == self.domain_size // 2:
                # print("terminate at ",state)
                break
            state = next_s

        return trajectory, G
    def candidate_statesets(self):
        nonterminal_states=self.states[1:-1]
        return list(itertools.combinations(nonterminal_states, 0)) + list(itertools.combinations(nonterminal_states, 1)) + \
                  list(itertools.combinations(nonterminal_states, 2))