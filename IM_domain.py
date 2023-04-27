from importance_sampling.run_method import run_method
from importance_sampling.compute_value import compute_value
from RCMDP_Benchmarks.InventoryManagement import InventoryManagement,args
from RCMDP.agent.agent_env_loop import agent_env_loop
from RCMDP.agent.set_agent import RandomAgent, set_agent
from RCMDP.Utils import check_folder
from one_D_domain import print_MSE_rows
import matplotlib.pyplot as plt
import pickle
import time
import numpy as np
import os
args.method_name="PG" # evaluation policy comes from training with policy gradient algorithm
def convert_trajectories(trajectories):
    sar_trajectories = []
    for traj in trajectories:
        sar_trajectory = []
        for (s, a, r, c, s_next, grad, actionProbs,grad_adv, probs_adv) in traj:
            sar_trajectory.append((s[0],a,r))
        sar_trajectories.append(sar_trajectory)
    return sar_trajectories

def get_scores(methods, repetitions, trajectories_from_file, MC_iterations,env,p_b_agent,policy,behav,epsilon_c,epsilon_q,max_t):
    scores = {}
    for method in methods:
        scores[method] = []
    for run in range(repetitions):
        print("doing run ", run)
        if trajectories_from_file:
            trajectories = pickle.load(open("IM_trajectories/run" + str(run) + ".pkl", "rb"))
            trajectories = trajectories[:MC_iterations]
        else:
            agent_env_loop(env, p_b_agent, args, episodeCount=0, episodeLimit=MC_iterations,
                           using_nextstate=False)
            trajectories = convert_trajectories(p_b_agent.trajectories)
            pickle.dump(trajectories, open("IM_trajectories/run" + str(run) + ".pkl", "wb"))
            p_b_agent.trajectories.clear()
        H = max([len(traj) for traj in trajectories])
        for method in methods:
            score = run_method(env, method, trajectories, policy, behav, H, epsilon_c, epsilon_q, max_t)
            scores[method].append(score)
    return scores
def variance_test(store_results,methods,tag,epsilon_c,epsilon_q,max_t,trajectories_from_file,load_scores):
    check_folder("IM_trajectories/")
    S=10
    MC_iterations_list = [100,250,500,1000]# #[100,1000]
    resultsfolder = "IM_results/"  # results folder
    check_folder(resultsfolder)
    repetitions=50
    gamma=1.0
    d = np.zeros(1)
    # domain
    env = InventoryManagement(gamma, d, S, using_nextstate=False)  # real samples
    env.stage = "test"
    p_b_agent = RandomAgent(len(env.actions), uncertainty_set=None)
    behav = [[1. / S for a in range(len(env.actions))] for i in range(S)]
    # trained policy
    p_e_agent = set_agent(args, env)
    p_e_agent.load_from_path("stored_policies/")
    policy = [p_e_agent.pi.select_action([i], deterministic=False)[2] for i in range(S)]  # third element
    _Q, _V, eval_score = compute_value(env.get_true_d0(), env.get_true_P(), env.get_true_R(), env.gamma, env.states,
                                       env.actions,
                                       H=env.stepsPerEpisode,
                                       p_e=policy)
    print("true score ", eval_score)
    score_l = {}
    score_u = {}
    score_m = {}
    MSEs = {}
    for method in methods:
        score_l[method] = [[] for i in MC_iterations_list]
        score_u[method] = [[] for i in MC_iterations_list]
        score_m[method] = [[] for i in MC_iterations_list]
        MSEs[method] = []
    MSEs = {}
    for method in methods:
        MSEs[method] = []
        # _, eval_score_MC = env.monte_carlo_eval(policy,seed=0,MC_iterations=1000)
        # print("true score ", eval_score_MC)
        # env.policy_to_theta(policy,"pi_e.txt")
        # env.policy_to_theta(behav, "pi_b.txt")


    for idx, MC_iterations in enumerate(MC_iterations_list):
        scorefile = resultsfolder + "variance_test_" + str(MC_iterations) + tag + "_scores.pkl"
        if load_scores:
            try:
                scores = pickle.load(open(scorefile, "rb"))
                print("loaded scores from scorefile", scorefile)
                print(scores)
                time.sleep(2)
            except:
                print("could NOT load scores from scorefile", scorefile)
                scores = get_scores(methods, repetitions, trajectories_from_file, MC_iterations, env, p_b_agent, policy,
                                    behav, epsilon_c,
                                    epsilon_q, max_t)
                pickle.dump(scores, open(scorefile, "wb"))
        else:
            scores = get_scores(methods, repetitions, trajectories_from_file, MC_iterations, env, p_b_agent, policy, behav, epsilon_c,
                       epsilon_q, max_t)
            pickle.dump(scores,open(scorefile, "wb"))

        #add MSEs
        for method in methods:
            sc = scores[method]
            m = np.mean(sc) - eval_score
            s = np.std(sc) / np.sqrt(len(sc))
            score_l[method][idx] = m - s
            score_u[method][idx] = m + s
            score_m[method][idx] = m
            MSE = np.mean([(score - eval_score)**2 for score in scores[method]])
            MSEs[method].append(MSE / eval_score**2) # divide by maximum for interpretability

    if store_results:
        # table
        writefile = open(resultsfolder + "variance_test_IM_" + tag + ".txt", "w")
        for method in methods:
            writefile.write(r" & " + method)
        writefile.write("\n")
        writefile.write(r" \textbf{Episodes}")
        for method in methods:
            writefile.write(r" & " + method)
        writefile.write("\n")
        for idx, MC_iterations in enumerate(MC_iterations_list):
            writefile.write(str(MC_iterations))
            MSEList = [MSEs[method][idx] for method in methods]
            print_MSE_rows(MSEList, writefile)
            writefile.write("\n")
        writefile.close()
        markers = {"IS": "x", "PDIS": "o", "SIS (Lift states)": "s", "SIS (Covariance testing)": "D",
                   "SIS (Q-based)": "v",
                   "SIS": "v",
                   "INCRIS": "^",
                   "DR": "x", "DRSIS (Lift states)": "s", "DRSIS (Covariance testing)": "D", "DRSIS (Q-based)": "v",
                   "DRSIS": "v",
                   "WIS": "x", "WPDIS": "o", "WSIS (Lift states)": "s", "WSIS (Covariance testing)": "D",
                   "WSIS (Q-based)": "v", "WSIS": "v",
                   "WINCRIS": "^",
                   "WDR": "x", "WDRSIS (Lift states)": "s", "WDRSIS (Covariance testing)": "D", "WDRSIS (Q-based)": "v",
                   "WDRSIS": "v",
                   "SPDIS": "v", "WSPDIS": "v", "SINCRIS": "D", "WSINCRIS": "D"
                   }
        colors = {"IS": "tab:blue", "PDIS": "tab:orange", "SIS (Lift states)": "tab:green",
                  "SIS (Covariance testing)": "tab:red",
                  "SIS (Q-based)": "tab:purple", "SIS": "tab:purple", "INCRIS": "tab:brown",
                  "DR": "tab:blue", "DRSIS (Lift states)": "tab:green", "DRSIS (Covariance testing)": "tab:red",
                  "DRSIS (Q-based)": "tab:purple", "DRSIS": "tab:purple",
                  "WIS": "tab:blue", "WPDIS": "tab:orange", "WSIS (Lift states)": "tab:green",
                  "WSIS (Covariance testing)": "tab:red", "WSIS (Q-based)": "tab:purple", "WSIS": "tab:purple",
                  "WINCRIS": "tab:brown", "WDR": "tab:blue", "WDRSIS (Lift states)": "tab:green",
                  "WDRSIS (Covariance testing)": "tab:red",
                  "WDRSIS (Q-based)": "tab:purple", "WDRSIS": "tab:purple",
                  "SPDIS": "tab:green", "WSPDIS": "tab:red", "SINCRIS": "tab:grey", "WSINCRIS": "tab:grey"
                  }
        lines = []
        betweens = []
        for method in methods:
            line, = plt.plot(MC_iterations_list, score_m[method], marker=markers[method], color=colors[method])
            b = plt.fill_between(MC_iterations_list, score_l[method], score_u[method], alpha=0.25)
            lines.append(line)
            betweens.append(b)
        plt.legend(lines, methods)

        plt.xlabel('Episodes')
        plt.ylabel('Residual ($\hat{G} - G$)')
        plt.savefig(resultsfolder + "variance_test_" + str(MC_iterations) + tag + ".pdf")

        plt.close()

if __name__ == '__main__':
    #convergence()
    # MC_methods=["WIS","WPDIS","WSIS (Covariance testing)","WSIS (Q-based)","WINCRIS"]
    # DR_methods = ["WDR","WDRSIS (Covariance testing)", "WDRSIS (Q-based)"]
    # methods = ["SPDIS", "WSPDIS", "SINCRIS", "WSINCRIS"]
    WDRSIS_methods = ["WDR", "WDRSIS (Covariance testing)", "WDRSIS (Q-based)"]
    all_methods = ["IS", "SIS", "PDIS", "SPDIS", "INCRIS", "SINCRIS"]  # SIS variants use Q-based identification
    weighted_all_methods = ["W" + method for method in all_methods]
    variance_test(methods=WDRSIS_methods, store_results=True,tag="WEIGHTED_DRSIS", epsilon_c=0.01,epsilon_q=50.0,max_t=10,
                  trajectories_from_file=True,load_scores=False)
    #variance_test(methods=DR_methods, store_results=True, tag="final_DR_methods_eps0.01_cardinality2", epsilon_c=0.01,epsilon_q=50.0,max_t=10,
    #              trajectories_from_file=True,load_scores=True)
