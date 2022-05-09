
from importance_sampling.baselines import *
from importance_sampling.SIS import *
from importance_sampling.INCRIS import *
import matplotlib.pyplot as plt


domain_size=9
reward_grid = [-1] + [0 for i in range(domain_size-2)] + [+1]
bound=domain_size//2
states=list(range(-bound,+bound+1))
actions = [-1,+1]
MC_iterations=10000


def manhattan_dist(a,b):
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])


def next_state(state,action):
    if state < 0 and state >= -bound + 2: # lift should not be before reward
        state-=1
    elif state > 0 and state <= bound -2: # lift should not be before reward
        state+=1
    else:
        state+=action
    return state
def optimal_policy():
    """
    immediately go to the closest corner
    :return:
    """

    policy=[[0.00,1.00] for i in range(domain_size)]
    return policy

def monte_carlo_eval(policy):
    G = 0
    trajectories=[]

    for k in range(MC_iterations):
        if k % 10000 == 0 :
            print("iteration ",k,"/",MC_iterations)
        trajectory, G_ =run_MDP(policy,seed=k)
        G+=G_
        trajectories.append(trajectory)
    return trajectories, G / MC_iterations
def run_MDP(policy,seed):
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
        a = np.random.choice(list(range(len(actions))),p=policy[state])
        action = actions[a]
        #print("state ", state)
        #print("action ", action)

        state = next_state(state,action)
        state_index = states.index(state)
        reward = reward_grid[state_index]
        #print("reward ", reward)
        G+=reward

        trajectory.append((state,a,reward))
        if np.abs(state) == domain_size//2:
            #print("terminate at ",state)
            break


    return trajectory,G

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    policy = optimal_policy()
    # behaviour policy
    behav = [[0.50,0.50] for i in range(domain_size)]
    _ , eval_score = monte_carlo_eval(policy)

    trajectories, behav_score = monte_carlo_eval(behav)

    #print("eval policy ", eval_score)
    print("behav policy ", behav_score)

    #data = trajectories,behav_score,eval_score
    #pickle.dump(data,open("data.pkl","wb"))
    #trajectories, behav_score, eval_score = pickle.load(open("data.pkl","rb")
    period=5000
    num_plotpoints = MC_iterations // period
    x = [i * period for i in range(num_plotpoints+1)]

    # note: this is intuitive but does not reflect how the algorithm would work for smaller number of trajectories; alternative is to recompute for each subset of
    # trajectories until then
    H = max([len(traj) for traj in trajectories])
    best_G, best_ks = INCRIS(trajectories, p_e=policy, p_b=behav, H=H, weighted=False)
    print("INCRIS",best_G)
    INCRIS_score = INCRIS_scores(trajectories, p_e=policy, p_b=behav, H=H, best_ks=best_ks, weighted=False, period=period)
    print("INCRIS")

    print("WPDIS")
    WPDIS_scores = WPDIS(trajectories, p_e=policy, p_b=behav, period=period)

    print("WIS")
    WIS_scores = WIS(trajectories, p_e=policy, p_b=behav, period=period)

    print("PDIS")
    PDIS_scores = PDIS(trajectories, p_e=policy, p_b=behav, period=period)

    print("IS")
    IS_scores =  IS(trajectories, p_e=policy, p_b=behav, period=period)

    print("Exhaustive SIS")
    SA_sets=[[]] + [[-1,1]] + [[0]] + [[1]] + [[2]] + [[3]] + [[-2,2]] + [[-3,3]]
    #SA_sets=[[]] + [[(s,a)] for s,a in lifts_int.items()] + [[(s,a) for s,a in lifts_int.items()]] + [[(s,a)] for s in states for a,act in enumerate(actions)]
    best_G, best_s_set = Exhaustive_SIS(trajectories, SA_sets, p_e=policy, p_b=behav, weighted=False)
    SIS_scores = SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=False,period=period)

    best_G, best_s_set = Exhaustive_SIS(trajectories, SA_sets, p_e=policy, p_b=behav,weighted=True)
    WSIS_scores = SIS(trajectories,best_s_set,p_e=policy,p_b=behav,weighted=True,period=period)
    #


    #Exhaustive_SPDIS(trajectories, SA_sets, p_e=policy, p_b=behav)
    line1, = plt.plot(x,IS_scores,marker="v")
    line2, = plt.plot(x,WIS_scores,marker="o")
    line3, = plt.plot(x,PDIS_scores,marker="x")
    line4, = plt.plot(x,SIS_scores,marker="D")
    line5, = plt.plot(x, WSIS_scores, marker="X")
    line6, = plt.plot(x, INCRIS_score, marker="O")
    plt.legend([line1,line2,line3,line4,line5],["IS","WIS","PDIS","SIS","WSIS","INCRIS"])
    plt.savefig("convergence.pdf")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
