import numpy as np
import itertools
import random

class One_D_Domain(object):
    def __init__(self,domain_size,reward_grid,bound,states,next_states,actions,stochastic):
        self.domain_size=domain_size
        self.reward_grid=reward_grid
        self.bound=bound
        self.states=states
        self.next_states=next_states
        self.actions=actions
        self.stochastic = stochastic

    def next_state(self,state, action):
        if state < 0 and state >= -self.bound + 2:  # lift should not be before reward
            next = state - 1
        elif state > 0 and state <= self.bound - 2:  # lift should not be before reward
            next = state + 1
        else:
            next = state + action
        return next

    def next_state_stochastic(self,state, action):
        r = random.random()
        if state < 0 and state >= -self.bound + 2:  # lift should not be before reward
            if r < 0.95:
                next = state - 1
            else:
                next = state + 1
        elif state > 0 and state <= self.bound - 2:  # lift should not be before reward
            if r < 0.95:
                next = state + 1
            else:
                next = state - 1
        else:
            if r < 0.95:
                next = state + action
            else:
                next = state - action
        return next

    def optimal_policy(self):
        """
        go to the right
        :return:
        """
        if self.stochastic:  # suboptimal
            policy = [[0.05, 0.95] for i in range(len(self.states))]
        else:   # optimal
            policy = [[0.00, 1.00] for i in range(len(self.states))]
        return policy

    def policy_to_theta(self,policy,filename): # write in the format of the MAGIC c++ code
        file=open(filename,"w")
        S = len(self.states)
        for a in range(len(self.actions)):
            for s in range(S):
                file.write("%.10f\n"%(np.log(policy[s][a]+10**(-20)),))   # needs to be logarithmic


    def monte_carlo_eval(self,policy,seed,MC_iterations):
        G = 0
        trajectories = []
        self.seed = seed
        self.MC_iterations=MC_iterations
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
            state_index = self.states.index(state)
            a = np.random.choice(list(range(len(self.actions))), p=policy[state_index])
            action = self.actions[a]
            # print("state ", state)
            # print("action ", action)
            if self.stochastic:
                next_s = self.next_state_stochastic(state, action)
            else:
                next_s = self.next_state(state, action)
            next_state_index = self.next_states.index(next_s)
            reward = self.reward_grid[next_state_index]
            # print("reward ", reward)
            G += reward
            trajectory.append((state_index, a, reward))
            if np.abs(next_s) == self.bound:
                #print("terminate at ",next_s, " reward ", self.reward_grid[state_index])
                break
            state = next_s

        return trajectory, G
    def candidate_statesets(self):
        nonterminal_states=range(len(self.states))
        return list(itertools.combinations(nonterminal_states, 0)) + list(itertools.combinations(nonterminal_states, 1)) + \
                  list(itertools.combinations(nonterminal_states, 2))

    def lift_stateset(self):
        l=[]
        for i in range(-self.bound+1,self.bound): # go over all non-terminal states
            if i < 0 and i >=  -self.bound + 2 or i > 0 and i <= self.bound - 2:
                l.append(self.bound + i - 1) # index 0 is left most non-terminal state
        return l