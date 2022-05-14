
import numpy as np
import itertools

class Two_D_Cross_Domain(object):
    def __init__(self,domain_size,reward_grid_x,reward_grid_y,bound,actions,iterations,seed):
        self.domain_size=domain_size
        self.reward_grid_x=reward_grid_x
        self.reward_grid_y = reward_grid_y
        self.bound=bound
        self.actions=actions
        self.MC_iterations=iterations
        self.seed=seed

    def next_state(self,state,action):
        if state[0] < 0 and state[0] >= -self.bound + 2: # lift should not be before reward
            next = (state[0]-1,state[1])
        elif state[0] > 0 and state[0] <= self.bound -2: # lift should not be before reward
            next = (state[0]+1,state[1])
        elif state[1] < 0 and state[1] >= -self.bound + 2: # lift should not be before reward
            next = (state[0],state[1]-1)
        elif state[1] > 0 and state[1] <= self.bound -2: # lift should not be before reward
            next = (state[0],state[1]+1)
        else:
            if state[0] != 0:
                next=(state[0] + action[0], state[1])
            elif state[1] != 0:
                next = (state[0], state[1] + action[1])
            else:
                next = (state[0] + action[0], state[1] + action[1])
        return next
    def policies(self):
        """
        immediately go to the closest corner
        :return:
        """
        policy = {}
        behav_policy = {}
        for i in range(-self.bound+1, +self.bound):    # no decision needed for terminal states
            policy[(i,0)] = [0,1,0,0]  # W, E, S, N --> go east
            behav_policy[(i,0)] = [0.25,0.25,0.25,0.25]
        for j in range(-self.bound+1, +self.bound):    # no decision needed for terminal states
            policy[(0,j)] = [0,0,0,1]  # S, N, W, E --> go north
            behav_policy[(0,j)] = [0.25, 0.25, 0.25, 0.25]


        return policy, behav_policy

    def monte_carlo_eval(self,policy):
        G = 0
        trajectories=[]

        for k in range(self.MC_iterations):
            if k % 10000 == 0 :
                print("iteration ",k,"/",self.MC_iterations)
            trajectory, G_ =self.run_MDP(policy,seed=k+self.seed)
            G+=G_
            trajectories.append(trajectory)
        return trajectories, G / self.MC_iterations
    def run_MDP(self,policy,seed):
        """
        run the MDP and return the cost
        :param policy:
        :return:
        """
        G = 0
        np.random.seed(seed)
        state = (0,0)
        trajectory = []
        while True:
            # print(actions)
            # print(state)
            # print(policy[state])
            a = np.random.choice(list(range(len(self.actions))),p=policy[state])
            action = self.actions[a]
            #print("state ", state)
            #print("action ", action)

            next_s = self.next_state(state,action)
            if next_s[0] != 0:
                state_index = next_s[0] + self.bound
                reward = self.reward_grid_x[state_index]
            elif state[1] != 0:
                state_index = next_s[1] + self.bound
                reward = self.reward_grid_y[state_index]
            else:
                reward = 0

            #print("reward ", reward)
            G+=reward

            trajectory.append((state, a, reward))

            if np.abs(next_s[0]) == self.domain_size//2 or np.abs(next_s[1]) == self.domain_size//2:
                #print("terminate at ",state)
                break

            state = next_s

        return trajectory,G
    def candidate_statesets(self,policy):
        return list(itertools.combinations(policy.keys(), 0)) + list(itertools.combinations(policy.keys(), 1)) + \
        list(itertools.combinations(policy.keys(), 2)) + list(itertools.combinations(policy.keys(), 3)) + \
        list(itertools.combinations(policy.keys(), 4))