from importance_sampling.run_method import run_method
from importance_sampling.compute_value import compute_value
import matplotlib.pyplot as plt
from envs.one_D_domain import *
import pickle
from RCMDP.Utils import check_folder
import argparse

parser = argparse.ArgumentParser(
                    prog = 'One D domain',
                    description = 'run RL on a one D problem with lift states')
parser.add_argument('--method', dest='method',type=str,default="MC") #MC or DR
parser.add_argument('--stochastic', dest='stochastic',type=str,default="deterministic") # deterministic or stochastic
parser.add_argument('--tag',dest="tag",type=str,default="SIS_METHODS") #
parser.add_argument('--load_scores',dest="load_scores",type=bool)

args = parser.parse_args()


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
def print_MSE_rows(MSEList,writefile):
    SortedMSEList = sorted(MSEList)
    best_MSE = SortedMSEList[0]
    for index, MSE in enumerate(MSEList):
        r = SortedMSEList.index(MSE)
        if r == 0:
            writefile.write(r"& \underline{\textbf{%.4f}} " % (MSEList[index]))
        elif MSEList[index] == best_MSE:  # underline all ties
            writefile.write(r"& \underline{\textbf{%.4f}} " % (MSEList[index]))
        elif r == 1:  # second performance (in case no tie)
            writefile.write(r"& \textbf{%.4f} " % (MSEList[index]))
        else:
            writefile.write(r"& %.4f " % (MSEList[index]))
def get_method_scores(sizes,MC_iterations,stochastic,methods,repetitions,folder, trajectories_from_file,epsilon_c,epsilon_q, max_t):
    scores = {}
    for method in methods:
        scores[method] =  [[] for i in sizes]
    for idx, domain_size in enumerate(sizes):  # [terminal, empty, lift(s), start, lift(s), empty, terminal] --> 1 or more lifts, horizon increasing
        print("doing domain size ", domain_size)
        bound = domain_size // 2
        actions = [-1, +1]
        reward_grid = [-bound] + [-1.0 for i in range(domain_size - 2)] + [+bound]  # penalise length of the path
        states = range(-bound + 1, +bound)  # non-terminal states
        next_states = range(-bound, bound + 1)  # all states (including terminal for reward grid)

        # domain
        env = One_D_Domain(domain_size, reward_grid, bound, states, next_states, actions, stochastic=stochastic)
        # best policy
        policy = env.optimal_policy()
        # _, eval_score_MC = env.monte_carlo_eval(policy,seed=0,MC_iterations=1000)
        # print("true score ", eval_score_MC)
        gamma = 1.0
        _Q, _V, eval_score = compute_value(env.get_true_d0(), env.get_true_P(), env.get_true_R(), gamma, states,
                                           actions, H=1000, p_e=policy)
        print("true score ", eval_score)
        # behaviour policy
        behav = [[0.50, 0.50] for i in range(len(states))]

        env.policy_to_theta(policy, "pi_e.txt")
        env.policy_to_theta(behav, "pi_b.txt")
        for run in range(repetitions):
            print("doing run ", run)
            savefile = folder + "size" + str(domain_size) + "_run" + str(run) + ".pkl"
            if trajectories_from_file:
                trajectories = pickle.load(open(savefile, "rb"))
                trajectories = trajectories[:MC_iterations]
            else:
                trajectories, behav_score = env.monte_carlo_eval(behav, seed=run * MC_iterations,
                                                                 MC_iterations=MC_iterations)
                pickle.dump(trajectories, open(savefile, "wb"))

            # period = 5000
            # num_plotpoints = MC_iterations // period
            # x = [i * period for i in range(num_plotpoints + 1)]

            # note: this is intuitive but does not reflect how the algorithm would work for smaller number of trajectories; alternative is to recompute for each subset of
            # trajectories until then
            H = max([len(traj) for traj in trajectories])
            for method in methods:
                score = run_method(env, method, trajectories, policy, behav, H, epsilon_c, epsilon_q,max_t)
                scores[method][idx].append(score)
    return scores, eval_score

def variance_test(stochastic,store_results,methods,tag,scale,epsilon_c,epsilon_q,max_t,
                  trajectories_from_file,load_scores):
    stoch_string = "_"+stochastic if stochastic != "deterministic" else ""
    folder="1D"+stoch_string+"_trajectories/" # trajectory folder
    check_folder(folder)
    resultsfolder = "1D" + stoch_string + "_results/" # results folder
    check_folder(resultsfolder)
    MC_iterations_list = [1000]
    repetitions=200 if stochastic!="deterministic" else 50
    sizes=[7,9,11,13,15,17]

    for MC_iterations in MC_iterations_list:
        score_l = {}
        score_u = {}
        score_m = {}
        MSEs = {}
        for method in methods:
            score_l[method] = [[] for i in sizes]
            score_u[method] = [[] for i in sizes]
            score_m[method] = [[] for i in sizes]
            MSEs[method] = []
        scorefile = resultsfolder+"variance_test_"+str(MC_iterations)+stoch_string+tag+ "_scores.pkl"
        if load_scores:
            scores, eval_score = pickle.load(open(scorefile,"rb"))
            print("loaded scores ", scores)
            print("loaded eval score ", eval_score)
        else:
            scores, eval_score = get_method_scores(sizes, MC_iterations, stochastic, methods, repetitions, folder,
                                                   trajectories_from_file, epsilon_c,
                              epsilon_q,max_t)
            pickle.dump((scores, eval_score),open(scorefile,"wb"))
        # now get stats on the scores
        for method in methods:
            for idx, size in enumerate(sizes):
                sc = scores[method][idx]
                m = np.mean(sc) - eval_score
                s = np.std(sc) / np.sqrt(len(sc))
                score_l[method][idx] = m - s
                score_u[method][idx] = m + s
                score_m[method][idx] = m

                MSE = np.mean([(score - eval_score)**2 for score in scores[method][idx]])
                MSEs[method].append(MSE)
        if store_results:
            markers={"IS": "x","PDIS":"o","SIS (Lift states)":"s","SIS (Covariance testing)":"D","SIS (Q-based)": "v",
                     "SIS": "v",
                     "INCRIS":"^",
                     "DR": "x", "DRSIS (Lift states)": "s", "DRSIS (Covariance testing)": "D", "DRSIS (Q-based)": "v",
                     "DRSIS": "v",
                     "WIS": "x", "WPDIS": "o", "WSIS (Lift states)": "s", "WSIS (Covariance testing)": "D",
                     "WSIS (Q-based)": "v", "WSIS": "v",
                    "WINCRIS": "^",
                     "WDR": "x", "WDRSIS (Lift states)": "s", "WDRSIS (Covariance testing)": "D", "WDRSIS (Q-based)": "v",
                     "WDRSIS": "v",
                     "SPDIS":"v", "WSPDIS":"v", "SINCRIS":"D", "WSINCRIS":"D"
                     }
            colors={"IS": "tab:blue","PDIS":"tab:orange","SIS (Lift states)":"tab:green","SIS (Covariance testing)":"tab:red",
                    "SIS (Q-based)": "tab:purple","SIS": "tab:purple","INCRIS":"tab:brown",
                     "DR": "tab:blue", "DRSIS (Lift states)": "tab:green", "DRSIS (Covariance testing)": "tab:red",
                    "DRSIS (Q-based)": "tab:purple","DRSIS": "tab:purple",
                    "WIS": "tab:blue", "WPDIS": "tab:orange", "WSIS (Lift states)": "tab:green",
                    "WSIS (Covariance testing)": "tab:red", "WSIS (Q-based)": "tab:purple","WSIS": "tab:purple",
                    "WINCRIS": "tab:brown", "WDR": "tab:blue", "WDRSIS (Lift states)": "tab:green",
                    "WDRSIS (Covariance testing)": "tab:red",
                    "WDRSIS (Q-based)": "tab:purple", "WDRSIS": "tab:purple",
                    "SPDIS": "tab:green", "WSPDIS": "tab:red", "SINCRIS": "tab:grey", "WSINCRIS": "tab:grey"
                    }

            lines=[]
            betweens=[]
            for method in methods:
                line, = plt.plot(sizes,score_m[method],marker=markers[method],color=colors[method],scaley=scale)
                b = plt.fill_between(sizes,  score_l[method],  score_u[method],alpha=0.25)
                lines.append(line)
                betweens.append(b)
            plt.legend(lines,methods)

            plt.xlabel('Domain size')
            plt.ylabel('Residual ($\hat{G} - G$)')
            plt.savefig(resultsfolder+"variance_test_"+str(MC_iterations)+tag+".pdf")

            plt.close()

            # table
            writefile = open(resultsfolder + "variance_test_" + str(MC_iterations) + tag + ".txt", "w")
            for method in methods:
                writefile.write(r" & " + method)
            writefile.write("\n \\textbf{Domain size}")
            for method in methods:
                writefile.write("& ")
            writefile.write("\n" )
            for idx, size in enumerate(sizes):
                writefile.write("%d " % (size,))
                MSEList = [MSEs[method][idx] for method in methods]
                print_MSE_rows(MSEList, writefile)
                writefile.write("\n")
            writefile.close()


if __name__ == '__main__':
    #convergence()
    # if args.method == "MC":
    #     methods = ["IS", "PDIS", "SIS (Lift states)", "SIS (Covariance testing)", "SIS (Q-based)", "INCRIS"]
    # else:
    #     methods = ["DR", "DRSIS (Lift states)", "DRSIS (Covariance testing)", "DRSIS (Q-based)"]
    # if args.stochastic.startswith("stochastic"): # use weighted
    #     methods = ["W"+method for method in methods]
    SIS_methods = ["IS", "SIS (Lift states)", "SIS (Covariance testing)", "SIS (Q-based)"]
    WSIS_methods = ["WIS", "WSIS (Lift states)", "WSIS (Covariance testing)", "WSIS (Q-based)"]
    DRSIS_methods = ["DR", "DRSIS (Lift states)", "DRSIS (Covariance testing)", "DRSIS (Q-based)"]
    WDRSIS_methods = ["WDR", "WDRSIS (Lift states)", "WDRSIS (Covariance testing)", "WDRSIS (Q-based)"]
    all_methods = ["IS","SIS","PDIS","SPDIS","INCRIS","SINCRIS"] # SIS variants use Q-based identification
    weighted_all_methods = ["W"+method for method in all_methods]


    variance_test(methods=SIS_methods, stochastic=args.stochastic, store_results=True, tag=args.tag,
                  scale="log",epsilon_c=0.01,epsilon_q=1.0,
                  max_t=float("inf"),trajectories_from_file=True,load_scores=False)
