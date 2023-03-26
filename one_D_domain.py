
from importance_sampling.baselines import *
from importance_sampling.SIS import *
from importance_sampling.INCRIS import *
import matplotlib.pyplot as plt
from envs.one_D_domain import *
from importance_sampling.DoublyRobust import *




#
# def convergence():
#     domain_size = 7
#     bound = domain_size // 2
#     reward_grid = [-bound] + [0 for i in range(domain_size - 2)] + [+bound]
#     states = list(range(-bound, +bound + 1))
#     actions = [-1, +1]
#     MC_iterations = 100
#     env = One_D_Domain(domain_size,reward_grid,bound,states,actions,MC_iterations,seed=10*MC_iterations)
#     policy = env.optimal_policy()
#     # behaviour policy
#     behav = [[0.50, 0.50] for i in range(domain_size)]
#     _, eval_score = env.monte_carlo_eval(policy)
#     trajectories, behav_score = env.monte_carlo_eval(behav)
#
#     # print("eval policy ", eval_score)
#     print("behav policy ", behav_score)
#
#     # data = trajectories,behav_score,eval_score
#     # pickle.dump(data,open("data.pkl","wb"))
#     # trajectories, behav_score, eval_score = pickle.load(open("data.pkl","rb")
#     period = 50
#     num_plotpoints = MC_iterations // period
#     x = [i * period for i in range(1,num_plotpoints+1)]
#
#     # note: this is intuitive but does not reflect how the algorithm would work for smaller number of trajectories; alternative is to recompute for each subset of
#     # trajectories until then
#     H = max([len(traj) for traj in trajectories])
#     best_G, best_ks = INCRIS(trajectories, p_e=policy, p_b=behav, H=H, weighted=False)
#     print("INCRIS", best_G)
#     INCRIS_scores,best_ks = INCRIS_Gs(trajectories, p_e=policy, p_b=behav, H=H, best_ks=best_ks, weighted=False,
#                                  period=period)
#     print("INCRIS")
#
#     #print("WPDIS")
#     #WPDIS_scores = WPDIS(trajectories, p_e=policy, p_b=behav, period=period)
#
#     #print("WIS")
#     #WIS_scores = WIS(trajectories, p_e=policy, p_b=behav, period=period)
#
#     print("PDIS")
#     PDIS_scores = PDIS(trajectories, p_e=policy, p_b=behav, period=period)
#
#     print("IS")
#     IS_scores = IS(trajectories, p_e=policy, p_b=behav, period=period)
#
#     print("Exhaustive SIS")
#
#     S_sets=env.candidate_statesets()
#
#     best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=False)
#     SIS_scores = SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=False, period=period)
#
# #    best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=True)
#   #  WSIS_scores = SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=True, period=period)
#     #
#
#     # Exhaustive_SPDIS(trajectories, SA_sets, p_e=policy, p_b=behav)
#     line1, = plt.plot(x, IS_scores, marker="v")
#     #line2, = plt.plot(x, WIS_scores, marker="o")
#     line2, = plt.plot(x, PDIS_scores, marker="x")
#     line3, = plt.plot(x, SIS_scores, marker="D")
#     #line5, = plt.plot(x, WSIS_scores, marker="X")
#     line4, = plt.plot(x, INCRIS_scores, marker="^")
#     plt.legend([line1, line2, line3, line4], ["IS", "PDIS", "SIS", "INCRIS"])
#     plt.savefig("convergence.pdf")

def variance_test(stochastic,store_results):
    actions = [-1, +1]
    MC_iterations_list = [1000]
    repetitions=25
    sizes=[7,9,11,13,15,17]

    for MC_iterations in MC_iterations_list:
        IS_score_l = [[] for i in sizes]
        WIS_score_l = [[] for i in sizes]
        PDIS_score_l = [[] for i in sizes]
        SIS_score_l = [[] for i in sizes]
        SIS_score_search_l = [[] for i in sizes]
        QSIS_score_l = [[] for i in sizes]
        WSIS_score_l = [[] for i in sizes]
        INCRIS_score_l = [[] for i in sizes]
        IS_score_u = [[] for i in sizes]
        WIS_score_u = [[] for i in sizes]
        PDIS_score_u = [[] for i in sizes]
        SIS_score_u = [[] for i in sizes]
        SIS_score_search_u = [[] for i in sizes]
        QSIS_score_u = [[] for i in sizes]
        WSIS_score_u = [[] for i in sizes]
        INCRIS_score_u = [[] for i in sizes]
        IS_score_m = [[] for i in sizes]
        WIS_score_m = [[] for i in sizes]
        PDIS_score_m = [[] for i in sizes]
        SIS_score_m = [[] for i in sizes]
        SIS_score_search_m = [[] for i in sizes]
        WSIS_score_m = [[] for i in sizes]
        INCRIS_score_m = [[] for i in sizes]
        QSIS_score_m = [[] for i in sizes]

        IS_MSEs = []
        PDIS_MSEs = []
        SIS_MSEs = []
        SIS_search_MSEs = []
        QSIS_MSEs = []
        INCRIS_MSEs = []
        for idx, domain_size in enumerate(sizes):  # [terminal, empty, lift(s), start, lift(s), empty, terminal] --> 1 or more lifts, horizon increasing
            print("doing domain size ",domain_size)
            bound = domain_size // 2
            reward_grid = [-bound] + [-1.0 for i in range(domain_size - 2)] + [+bound]  # penalise length of the path
            states = list(range(-bound, +bound + 1))
            IS_scores=[]
            SIS_scores=[]
            SIS_scores_search=[]
            QSIS_scores=[]
            DR_scores=[]
            #WIS_scores = []
            #WSIS_scores = []
            PDIS_scores = []
            INCRIS_scores = []
            #domain
            env = One_D_Domain(domain_size, reward_grid, bound, states, actions, stochastic=stochastic)
            # best policy
            policy = env.optimal_policy()
            _, eval_score, terminals = env.monte_carlo_eval(policy,seed=0,MC_iterations=100)
            print("true score ", eval_score)
            # behaviour policy
            behav = [[0.50, 0.50] for i in range(domain_size)]


            for run in range(repetitions):
                print("doing run ", run)
                trajectories, behav_score, terminals = env.monte_carlo_eval(behav,seed=run*MC_iterations,MC_iterations=MC_iterations)

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

                #print("WIS")
                #WIS_scores.append(WIS(trajectories, p_e=policy, p_b=behav)[-1])

                print("PDIS")
                PDIS_scores.append(PDIS(trajectories, p_e=policy, p_b=behav)[-1])

                print("IS")
                IS_scores.append(IS(trajectories, p_e=policy, p_b=behav)[-1])

                print("Exhaustive SIS")

                best_s_set=env.lift_stateset()
                print(best_s_set)
                SIS_scores.append(SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=False, period=period)[0])
                print(SIS_scores[-1])

                S_sets = env.candidate_statesets()
                best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=False)
                SIS_scores_search.append(SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=False, period=period)[0])
                print(SIS_scores_search[-1])
                score, S_A = Qvalue_SIS(H=H,r_max=1,gamma=1.0,epsilon=0.05, alpha=0.25, trajectories=trajectories,
                           terminals=terminals, state_space=states, action_space=actions,
                           p_e=policy, p_b=behav, weighted=False)
                QSIS_scores.append(score[0])
                print(QSIS_scores)

                score = DoublyRobust(trajectories, H, states, actions, terminals, weighted=False, gamma=1.0,p_e=policy,
                                     p_b=behav)
                DR_scores.append(score)
                #best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=True)
                #WSIS_scores.append(SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=True, period=period)[0])
                #print(WSIS_scores[-1])

            # IS
            m=np.mean(IS_scores) if not stochastic else np.abs(np.mean(IS_scores) - eval_score)
            s=np.std(IS_scores)/np.sqrt(len(IS_scores))
            IS_score_l[idx]=m-s
            IS_score_u[idx] = m + s
            IS_score_m[idx] = m
            # WIS
            #m=np.mean(WIS_scores)
            #s=np.std(WIS_scores)/np.sqrt(len(WIS_scores))
            #WIS_score_l[idx]=m-s
            #WIS_score_u[idx] = m + s
            #WIS_score_m[idx] = m
            # PDIS
            m=np.mean(PDIS_scores) if not stochastic else np.mean(PDIS_scores) - eval_score
            s=np.std(PDIS_scores)/np.sqrt(len(PDIS_scores))
            PDIS_score_l[idx]=m-s
            PDIS_score_u[idx] = m + s
            PDIS_score_m[idx] = m

            # SIS
            m=np.mean(SIS_scores) if not stochastic else np.mean(SIS_scores) - eval_score
            s=np.std(SIS_scores)/np.sqrt(len(SIS_scores))
            SIS_score_l[idx]=m-s
            SIS_score_u[idx] = m + s
            SIS_score_m[idx] = m

            m = np.mean(SIS_scores_search) if not stochastic else np.mean(SIS_scores_search) - eval_score
            s = np.std(SIS_scores_search) / np.sqrt(len(SIS_scores_search))
            SIS_score_search_l[idx]=m-s
            SIS_score_search_u[idx] = m + s
            SIS_score_search_m[idx] = m

            m = np.mean(QSIS_scores) if not stochastic else np.mean(QSIS_scores) - eval_score
            s = np.std(QSIS_scores) / np.sqrt(len(QSIS_scores))
            QSIS_score_l[idx]=m-s
            QSIS_score_u[idx] = m + s
            QSIS_score_m[idx] = m
            # WSIS
            #m=np.mean(WSIS_scores)
            #s=np.std(WSIS_scores)/np.sqrt(len(WSIS_scores))
            #WSIS_score_l[idx]=m-s
            #WSIS_score_u[idx] = m + s
            #WSIS_score_m[idx] = m
            # INCRIS
            m=np.mean(INCRIS_scores) if not stochastic else np.mean(INCRIS_scores) - eval_score
            s=np.std(INCRIS_scores)/np.sqrt(len(INCRIS_scores))
            INCRIS_score_l[idx]=m-s
            INCRIS_score_u[idx] = m + s
            INCRIS_score_m[idx] = m


            #add MSEs
            if stochastic: # square the residual
                IS_MSEs.append(np.mean([score ** 2 for score in IS_scores]))
                PDIS_MSEs.append(np.mean([score ** 2 for score in PDIS_scores]))
                SIS_MSEs.append(np.mean([score ** 2 for score in SIS_scores]))
                SIS_search_MSEs.append(np.mean([score ** 2 for score in SIS_scores_search]))
                QSIS_MSEs.append(np.mean([score ** 2 for score in QSIS_scores]))
                INCRIS_MSEs.append(np.mean([score ** 2 for score in INCRIS_scores]))
            else:
                IS_MSEs.append(np.mean([(score - eval_score)**2 for score in IS_scores]))
                PDIS_MSEs.append(np.mean([(score - eval_score) ** 2 for score in PDIS_scores]))
                SIS_MSEs.append(np.mean([(score - eval_score) ** 2 for score in SIS_scores]))
                SIS_search_MSEs.append(np.mean([(score - eval_score) ** 2 for score in SIS_scores_search]))
                QSIS_MSEs.append(np.mean([(score - eval_score) ** 2 for score in QSIS_scores]))
                INCRIS_MSEs.append(np.mean([(score - eval_score) ** 2 for score in INCRIS_scores]))
        if store_results:
            line1, = plt.plot(sizes,IS_score_m,marker="v")
            b1 = plt.fill_between(sizes,  IS_score_l,  IS_score_u,alpha=0.25)
            #line2, = plt.plot(sizes, WIS_score_m,marker="o")
            #b2 = plt.fill_between(sizes,  WIS_score_l,  WIS_score_u,alpha=0.25)
            line2, = plt.plot(sizes, PDIS_score_m,marker="x")
            b2 = plt.fill_between(sizes,  PDIS_score_l,  PDIS_score_u,alpha=0.25)
            line3, = plt.plot(sizes, SIS_score_m,marker="D")
            b3 = plt.fill_between(sizes,  SIS_score_l,  SIS_score_u,alpha=0.25)

            line4, = plt.plot(sizes, SIS_score_search_m,marker="D")
            b4 = plt.fill_between(sizes,  SIS_score_search_l,  SIS_score_search_u,alpha=0.25)

            line5, = plt.plot(sizes, QSIS_score_m,marker="D")
            b5 = plt.fill_between(sizes,  QSIS_score_l,  QSIS_score_u,alpha=0.25)
            #line5, = plt.plot(sizes, WSIS_score_m,marker="X")
            #b5 = plt.fill_between(sizes,  WSIS_score_l,  WSIS_score_u,alpha=0.25)
            line6, = plt.plot(sizes, INCRIS_score_m,marker="^")
            b6 = plt.fill_between(sizes,  INCRIS_score_l,  INCRIS_score_u,alpha=0.25)
            line7, = plt.plot(sizes, np.zeros((len(sizes))) + 1,linestyle="--")  if not stochastic else plt.plot(sizes, np.zeros((len(sizes))),linestyle="--")

            plt.legend([line1,line2,line3,line4,line5,line6,line7], [r"$\hat{G}_{IS}$", r"$\hat{G}_{PDIS}$", r"$\hat{G}_{SIS}$ (Lift-states)",
                                                           r"$\hat{G}_{SIS}$ (Search-based)", r"$\hat{G}_{SIS}$ (Q-based)","$\hat{G}_{INCRIS}$",r"$G$"])

            plt.xlabel('Domain size')
            if stochastic:
                plt.ylabel('Residual ($\hat{G} - G$)')
            else:
                plt.ylabel('Expected return estimate ($\hat{G}$)')
            stoch_string="_stochastic" if stochastic else ""
            plt.savefig("variance_test_"+str(MC_iterations)+"its_eps0.01"+stoch_string+"_nobarcheck_Q.pdf")

            plt.close()

            # table
            writefile=open("variance_test_"+str(MC_iterations)+"_eps0.01"+stoch_string+"_nobarcheck_Q.txt","w")
            writefile.write(r" & $\hat{G}_{IS}$ & $\hat{G}_{PDIS}$ & $\hat{G}_{SIS}$ (Lift-states) & $\hat{G}_{SIS}$ (Search-based) & $\hat{G}_{SIS}$ (Q-based) & $\hat{G}_{INCRIS}$ \\\ \n "\
                           "\textbf{Size} & & & & & \\ \n" )
            for idx, size in enumerate(sizes):
                writefile.write("%d & %.4f & %.4f  & %.4f & %.4f& %.4f %.4f\\ \n"%(size,IS_MSEs[idx],PDIS_MSEs[idx],SIS_MSEs[idx],SIS_search_MSEs[idx],QSIS_MSEs[idx],INCRIS_MSEs[idx]))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #convergence()
    variance_test(stochastic=False,store_results=True)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
