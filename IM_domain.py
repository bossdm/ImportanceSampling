from importance_sampling.run_method import run_method
from importance_sampling.compute_value import compute_value
import matplotlib.pyplot as plt
from RCMDP_Benchmarks.InventoryManagement import InventoryManagement,args
from RCMDP.agent.agent_env_loop import agent_env_loop
from RCMDP.agent.set_agent import RandomAgent, set_agent

import numpy as np
args.method_name="PG" # evaluation policy comes from training with policy gradient algorithm
def convert_trajectories(trajectories):
    sar_trajectories = []
    for traj in trajectories:
        sar_trajectory = []
        for (s, a, r, c, s_next, grad, actionProbs,grad_adv, probs_adv) in traj:
            sar_trajectory.append((s[0],a,r))
        sar_trajectories.append(sar_trajectory)
    return sar_trajectories
def variance_test(store_results,methods,tag,epsilon_c,epsilon_q):
    S=10
    MC_iterations_list = [100,1000] #[100,1000]
    repetitions=50
    gamma=1.0
    d = np.zeros(1)
    for MC_iterations in MC_iterations_list:
        MSEs = {}
        for method in methods:
            MSEs[method] = []

            #domain
            env = InventoryManagement(gamma, d, S, using_nextstate=False)  # real samples
            env.stage = "test"
            p_b_agent = RandomAgent(len(env.actions), uncertainty_set=None)
            behav = [[1./S for a in range(len(env.actions))] for i in range(S)]
            # trained policy
            p_e_agent = set_agent(args,env)
            policy = [p_e_agent.pi.select_action([i],deterministic=False)[2] for i in range(S)] # third element
            #_, eval_score_MC = env.monte_carlo_eval(policy,seed=0,MC_iterations=1000)
            #print("true score ", eval_score_MC)
            gamma=1.0
            _Q,_V,eval_score = compute_value(env.get_true_d0(), env.get_true_P(), env.get_true_R(), env.gamma, env.states, env.actions,
                                             H=env.stepsPerEpisode,
                                             p_e=policy)
            print("true score ", eval_score)

            # env.policy_to_theta(policy,"pi_e.txt")
            # env.policy_to_theta(behav, "pi_b.txt")
            scores={}
            for method in methods:
                scores[method] = []

            for run in range(repetitions):
                print("doing run ", run)
                agent_env_loop(env, p_b_agent, args, episodeCount=0, episodeLimit=MC_iterations,
                               using_nextstate=False)
                trajectories = convert_trajectories(p_b_agent.trajectories)

                H = max([len(traj) for traj in trajectories])
                for method in methods:
                    score = run_method(env,method,trajectories, policy, behav, H,epsilon_c,epsilon_q)
                    scores[method].append(score)

            #add MSEs]
            for method in methods:
                MSE = np.mean([(score - eval_score)**2 for score in scores[method]])
                MSEs[method].append(MSE)
        if store_results:
            markers={"IS": "x","PDIS":"o","SIS (Lift states)":"s","SIS (Covariance testing)":"D","SIS (Q-based)": "v","INCRIS":"^",
                     "DR": "x", "DRSIS (Lift states)": "s", "DRSIS (Covariance testing)": "D", "DRSIS (Q-based)": "v"}
            colors={"IS": "tab:blue","PDIS":"tab:orange","SIS (Lift states)":"tab:green","SIS (Covariance testing)":"tab:red","SIS (Q-based)": "tab:purple","INCRIS":"tab:brown",
                     "DR": "tab:blue", "DRSIS (Lift states)": "tab:green", "DRSIS (Covariance testing)": "tab:red", "DRSIS (Q-based)": "tab:purple"}

            # table
            writefile=open("variance_test_"+str(MC_iterations)+tag+".txt","w")
            for method in methods:
                writefile.write(r" & " + method)
            writefile.write("\n \\textbf(Size}")
            for method in methods:
                    writefile.write("& ")
            writefile.write("\n" )
            for idx, it in enumerate(MC_iterations_list):
                for method in methods:
                    writefile.write("%d "%(it,))
                    writefile.write("& %.4f "%(MSEs[method][idx]))
                writefile.write("\n")
            writefile.close()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #convergence()
    MC_methods=["IS","PDIS","SIS (Covariance testing)","SIS (Q-based)","INCRIS"]
    DR_methods = ["DR", "DRSIS (Covariance testing)", "DRSIS (Q-based)"]
    variance_test(methods=MC_methods, store_results=True,tag="MC_methods", epsilon_c=0.01,epsilon_q=1.0)
    variance_test(methods=DR_methods, store_results=True, tag="DR_methods", epsilon_c=0.01,epsilon_q=1.0)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
