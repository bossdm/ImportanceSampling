
from importance_sampling.baselines import *
from importance_sampling.SIS import *
from importance_sampling.INCRIS import *
import matplotlib.pyplot as plt
import itertools
from envs.two_D_cross_domain import *



def convergence():
    domain_size = 9
    reward_grid_x = [-1] + [0 for i in range(domain_size - 2)] + [+1]
    reward_grid_y = reward_grid_x
    bound = domain_size // 2
    actions = [(-1, 0), (1, 0), (0, -1), (0, +1)]  # W,E,S,N
    MC_iterations = 100000
    env = Two_D_Cross_Domain(domain_size,reward_grid_x,reward_grid_y,bound,actions,MC_iterations,seed=10*MC_iterations)
    policy, behav = env.policies()
    S_sets = env.candidate_statesets(policy)

    trajectories, behav_score = env.monte_carlo_eval(behav)
    _, eval_score = env.monte_carlo_eval(policy)
    print("eval policy ", eval_score)
    print("behav policy ", behav_score)

    # data = trajectories,behav_score,eval_score
    # pickle.dump(data,open("data.pkl","wb"))
    # trajectories, behav_score, eval_score = pickle.load(open("data.pkl","rb")
    period = 5000
    num_plotpoints = MC_iterations // period
    x = [i * period for i in range(num_plotpoints + 1)]

    # note: this is intuitive but does not reflect how the algorithm would work for smaller number of trajectories; alternative is to recompute for each subset of
    # trajectories until then
    H = max([len(traj) for traj in trajectories])
    best_G, best_ks = INCRIS(trajectories, p_e=policy, p_b=behav, H=H, weighted=False)
    print("INCRIS", best_G)
    INCRIS_score = INCRIS_Gs(trajectories, p_e=policy, p_b=behav, H=H, best_ks=best_ks, weighted=False,
                                 period=period)
    print("INCRIS")

    print("WPDIS")
    WPDIS_scores = WPDIS(trajectories, p_e=policy, p_b=behav, period=period)

    print("WIS")
    WIS_scores = WIS(trajectories, p_e=policy, p_b=behav, period=period)

    print("PDIS")
    PDIS_scores = PDIS(trajectories, p_e=policy, p_b=behav, period=period)

    print("IS")
    IS_scores = IS(trajectories, p_e=policy, p_b=behav, period=period)

    print("Exhaustive SIS")
    # SA_sets=[[]] + [[(s,a)] for s,a in lifts_int.items()] + [[(s,a) for s,a in lifts_int.items()]] + [[(s,a)] for s in states for a,act in enumerate(actions)]
    best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=False)
    SIS_scores = SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=False, period=period)

    best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=True)
    WSIS_scores = SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=True, period=period)
    #

    # Exhaustive_SPDIS(trajectories, SA_sets, p_e=policy, p_b=behav)
    line1, = plt.plot(x, IS_scores, marker="v")
    line2, = plt.plot(x, WIS_scores, marker="o")
    line3, = plt.plot(x, PDIS_scores, marker="x")
    line4, = plt.plot(x, SIS_scores, marker="D")
    line5, = plt.plot(x, WSIS_scores, marker="X")
    line6, = plt.plot(x, INCRIS_score, marker="O")
    plt.legend([line1, line2, line3, line4, line5, line6], ["IS", "WIS", "PDIS", "SIS", "WSIS", "INCRIS"])
    plt.savefig("convergence.pdf")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    convergence()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
