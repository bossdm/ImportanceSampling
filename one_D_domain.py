
from importance_sampling.baselines import *
from importance_sampling.SIS import *
from importance_sampling.INCRIS import *
import matplotlib.pyplot as plt
from envs.one_D_domain import *





def convergence():
    domain_size = 7
    reward_grid = [-1] + [0 for i in range(domain_size - 2)] + [+1]
    bound = domain_size // 2
    states = list(range(-bound, +bound + 1))
    actions = [-1, +1]
    MC_iterations = 100000
    env = One_D_Domain(domain_size,reward_grid,bound,states,actions,MC_iterations,seed=10*MC_iterations)
    policy = env.optimal_policy()
    # behaviour policy
    behav = [[0.50, 0.50] for i in range(domain_size)]
    _, eval_score = env.monte_carlo_eval(policy)
    trajectories, behav_score = env.monte_carlo_eval(behav)

    # print("eval policy ", eval_score)
    print("behav policy ", behav_score)

    # data = trajectories,behav_score,eval_score
    # pickle.dump(data,open("data.pkl","wb"))
    # trajectories, behav_score, eval_score = pickle.load(open("data.pkl","rb")
    period = 5000
    num_plotpoints = MC_iterations // period
    x = [i * period for i in range(1,num_plotpoints+1)]

    # note: this is intuitive but does not reflect how the algorithm would work for smaller number of trajectories; alternative is to recompute for each subset of
    # trajectories until then
    H = max([len(traj) for traj in trajectories])
    best_G, best_ks = INCRIS(trajectories, p_e=policy, p_b=behav, H=H, weighted=False)
    print("INCRIS", best_G)
    INCRIS_scores,best_ks = INCRIS_Gs(trajectories, p_e=policy, p_b=behav, H=H, best_ks=best_ks, weighted=False,
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

    S_sets=env.candidate_statesets()

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
    line6, = plt.plot(x, INCRIS_scores, marker="^")
    plt.legend([line1, line2, line3, line4, line5, line6], ["IS", "WIS", "PDIS", "SIS", "WSIS", "INCRIS"])
    plt.savefig("convergence.pdf")

def variance_test():
    actions = [-1, +1]
    MC_iterations = 10000
    repetitions=4
    sizes=[7,9,11,13,15,17]
    IS_score_l=[[] for i in sizes]
    WIS_score_l = [[] for i in sizes]
    PDIS_score_l = [[] for i in sizes]
    SIS_score_l = [[] for i in sizes]
    WSIS_score_l = [[] for i in sizes]
    INCRIS_score_l = [[] for i in sizes]
    IS_score_u = [[] for i in sizes]
    WIS_score_u = [[] for i in sizes]
    PDIS_score_u = [[] for i in sizes]
    SIS_score_u = [[] for i in sizes]
    WSIS_score_u = [[] for i in sizes]
    INCRIS_score_u = [[] for i in sizes]
    IS_score_m = [[] for i in sizes]
    WIS_score_m = [[] for i in sizes]
    PDIS_score_m = [[] for i in sizes]
    SIS_score_m = [[] for i in sizes]
    WSIS_score_m = [[] for i in sizes]
    INCRIS_score_m = [[] for i in sizes]
    for idx, domain_size in enumerate(sizes):  # [terminal, empty, lift(s), start, lift(s), empty, terminal] --> 1 or more lifts, horizon increasing
        print("doing domain size ",domain_size)
        bound = domain_size // 2
        reward_grid = [-bound] + [-1.0 for i in range(domain_size - 2)] + [+bound]  # penalise length of the path
        states = list(range(-bound, +bound + 1))
        IS_scores=[]
        SIS_scores=[]
        WIS_scores = []
        WSIS_scores = []
        PDIS_scores = []
        INCRIS_scores = []
        for run in range(repetitions):
            print("doing run ", run)
            env = One_D_Domain(domain_size, reward_grid, bound, states, actions, MC_iterations, seed=run*MC_iterations)
            policy = env.optimal_policy()
            # behaviour policy
            behav = [[0.50, 0.50] for i in range(domain_size)]
            _, eval_score = env.monte_carlo_eval(policy)
            print("true score ", eval_score)
            trajectories, behav_score = env.monte_carlo_eval(behav)

            period = 5000
            num_plotpoints = MC_iterations // period
            x = [i * period for i in range(num_plotpoints + 1)]

            # note: this is intuitive but does not reflect how the algorithm would work for smaller number of trajectories; alternative is to recompute for each subset of
            # trajectories until then
            H = max([len(traj) for traj in trajectories])
            best_G, best_ks = INCRIS(trajectories, p_e=policy, p_b=behav, H=H, weighted=False)
            print("INCRIS", best_G)
            print("INCRIS")
            INCRIS_scores.append(best_G)

            #print("WPDIS")
            #WPDIS_score = WPDIS(trajectories, p_e=policy, p_b=behav, period=period)

            print("WIS")
            WIS_scores.append(WIS(trajectories, p_e=policy, p_b=behav)[-1])

            print("PDIS")
            PDIS_scores.append(PDIS(trajectories, p_e=policy, p_b=behav)[-1])

            print("IS")
            IS_scores.append(IS(trajectories, p_e=policy, p_b=behav)[-1])

            print("Exhaustive SIS")
            #S_sets=env.candidate_statesets()
            best_s_set=[-1,1]
            #best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=False)
            SIS_scores.append(SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=False, period=period)[0])
            print(SIS_scores[-1])

            #best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=True)
            WSIS_scores.append(SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=True, period=period)[0])
            print(WSIS_scores[-1])

        # IS
        m=np.mean(IS_scores)
        s=np.std(IS_scores)/np.sqrt(len(IS_scores))
        IS_score_l[idx]=m-s
        IS_score_u[idx] = m + s
        IS_score_m[idx] = m
        # WIS
        m=np.mean(WIS_scores)
        s=np.std(WIS_scores)/np.sqrt(len(WIS_scores))
        WIS_score_l[idx]=m-s
        WIS_score_u[idx] = m + s
        WIS_score_m[idx] = m
        # PDIS
        m=np.mean(PDIS_scores)
        s=np.std(PDIS_scores)/np.sqrt(len(PDIS_scores))
        PDIS_score_l[idx]=m-s
        PDIS_score_u[idx] = m + s
        PDIS_score_m[idx] = m

        # SIS
        m=np.mean(SIS_scores)
        s=np.std(SIS_scores)/np.sqrt(len(SIS_scores))
        SIS_score_l[idx]=m-s
        SIS_score_u[idx] = m + s
        SIS_score_m[idx] = m
        # WSIS
        m=np.mean(WSIS_scores)
        s=np.std(WSIS_scores)/np.sqrt(len(WSIS_scores))
        WSIS_score_l[idx]=m-s
        WSIS_score_u[idx] = m + s
        WSIS_score_m[idx] = m
        # INCRIS
        m=np.mean(INCRIS_scores)
        s=np.std(INCRIS_scores)/np.sqrt(len(INCRIS_scores))
        INCRIS_score_l[idx]=m-s
        INCRIS_score_u[idx] = m + s
        INCRIS_score_m[idx] = m



    line1, = plt.plot(sizes,IS_score_m,marker="v")
    b1 = plt.fill_between(sizes,  IS_score_l,  IS_score_u,alpha=0.25)
    line2, = plt.plot(sizes, WIS_score_m,marker="o")
    b2 = plt.fill_between(sizes,  WIS_score_l,  WIS_score_u,alpha=0.25)
    line3, = plt.plot(sizes, PDIS_score_m,marker="x")
    b3 = plt.fill_between(sizes,  PDIS_score_l,  PDIS_score_u,alpha=0.25)
    line4, = plt.plot(sizes, SIS_score_m,marker="D")
    b4 = plt.fill_between(sizes,  SIS_score_l,  SIS_score_u,alpha=0.25)
    line5, = plt.plot(sizes, WSIS_score_m,marker="X")
    b5 = plt.fill_between(sizes,  WSIS_score_l,  WSIS_score_u,alpha=0.25)
    line6, = plt.plot(sizes, INCRIS_score_m,marker="^")
    b6 = plt.fill_between(sizes,  INCRIS_score_l,  INCRIS_score_u,alpha=0.25)
    plt.legend([line1, line2, line3, line4, line5, line6], ["IS", "WIS", "PDIS", "SIS", "WSIS", "INCRIS"])
    plt.savefig("variance_test.pdf")

    # table
    writefile=open("variance_test.txt","w")
    writefile.write("IS & WIS & PDIS & SIS & WSIS & INCRIS \\ \n")
    for idx, size in enumerate(sizes):
        se1 = IS_score_m[idx] - IS_score_l[idx]
        se2 = WIS_score_m[idx] - IS_score_l[idx]
        se3 = PDIS_score_m[idx] - IS_score_l[idx]
        se4 = SIS_score_m[idx] - IS_score_l[idx]
        se5=WSIS_score_m[idx] - IS_score_l[idx]
        se6 = INCRIS_score_m[idx] - IS_score_l[idx]

        writefile.write(str(IS_score_m[idx]) + "\pm" + str(se1) +" &" +\
                        str(WIS_score_m[idx]) + "\pm" +  str(se2) +\
                        str(PDIS_score_m[idx]) + "\pm" + str(se3) + " &"+\
                        str(SIS_score_m[idx]) + "\pm" + str(se4) +" &"+\
                        str(WSIS_score_m[idx]) + "\pm" + str(se5) +" &"+\
                        str(INCRIS_score_m[idx]) + "\pm" + str(se6) +" \\ \n")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #convergence()
    variance_test()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
