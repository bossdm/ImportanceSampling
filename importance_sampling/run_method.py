
from importance_sampling.baselines import *
from importance_sampling.SIS import *
from importance_sampling.INCRIS import *
from importance_sampling.DoublyRobust import *
def run_method(env, method,trajectories, policy,behav, H, epsilon_c, epsilon_q, max_t,JiangStyle=False):
    gamma=1.0
    print(method)
    if method == "INCRIS":
        best_G, best_ks = INCRIS(trajectories, p_e=policy, p_b=behav, H=H, weighted=False,max_t=max_t)
        print(best_G)
        return best_G
    elif method == "WINCRIS":
        best_G, best_ks = INCRIS(trajectories, p_e=policy, p_b=behav, H=H, weighted=True,max_t=max_t)
        print(best_G)
        return best_G
    elif method == "PDIS":
        return PDIS(trajectories, p_e=policy, p_b=behav)[-1]
    elif method == "WPDIS":
        return WPDIS(trajectories, p_e=policy, p_b=behav)
    elif method == "IS":
        return IS(trajectories, p_e=policy, p_b=behav)[-1]
    elif method == "WIS":
        return WIS(trajectories, p_e=policy, p_b=behav)[-1]
    elif method == "SIS (Lift states)":
        best_s_set = env.lift_stateset()
        print(best_s_set)
        G = SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=False)[0]
        print("G ", G)
        return G
    elif method == "WSIS (Lift states)":
        best_s_set = env.lift_stateset()
        print(best_s_set)
        G = SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=True)[0]
        print("G ", G)
        return G
    elif method == "SIS (Covariance testing)":
        S_sets = env.candidate_statesets()
        #print("candidates ", S_sets)
        best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=False,epsilon=epsilon_c)
        G =  SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=False)[0]
        print("G ", G)
        return G
    elif method == "WSIS (Covariance testing)":
        S_sets = env.candidate_statesets()
        #print("candidates ", S_sets)
        best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=True,epsilon=epsilon_c)
        G =  SIS(trajectories, best_s_set, p_e=policy, p_b=behav, weighted=True)[0]
        print("G ", G)
        return G
    elif method == "SIS (Q-based)":
        w, rmin, rmax, d0, P, R, hat_q, hat_v, hat_G = get_model(trajectories, H, env.states, env.actions,
                                                                 weighted=False,
                                                                 gamma=gamma, p_e=policy, p_b=behav, JiangStyle=False)
        # epsilon 1.0 because that is one step difference
        G = Qvalue_SIS(hat_q, trajectories, epsilon=epsilon_q, states=env.states, actions=env.actions, p_e=policy, p_b=behav,
                           weighted=False)[0]
        print("G ", G)
        return G
    elif method == "WSIS (Q-based)":
        w, rmin, rmax, d0, P, R, hat_q, hat_v, hat_G = get_model(trajectories, H, env.states, env.actions,
                                                                 weighted=False,
                                                                 gamma=gamma, p_e=policy, p_b=behav, JiangStyle=False)
        # epsilon 1.0 because that is one step difference
        G = Qvalue_SIS(hat_q, trajectories, epsilon=epsilon_q, states=env.states, actions=env.actions, p_e=policy, p_b=behav,
                           weighted=True)[0]
        print("G ", G)
        return G
    elif method =="DR":
        w, rmin, rmax, d0, P, R, hat_q, hat_v, hat_G = get_model(trajectories, H, env.states, env.actions, weighted=False,
                                                                 gamma=gamma, p_e=policy, p_b=behav, JiangStyle=False)
        G = DoublyRobust(trajectories, gamma, p_e=policy, p_b=behav, w=w, hat_q=hat_q, hat_v=hat_v)
        print("G ", G)
        return G
    elif method =="WDR":
        w, rmin, rmax, d0, P, R, hat_q, hat_v, hat_G = get_model(trajectories, H, env.states, env.actions, weighted=True,
                                                                 gamma=gamma, p_e=policy, p_b=behav, JiangStyle=False)
        G = DoublyRobust(trajectories, gamma, p_e=policy, p_b=behav, w=w, hat_q=hat_q, hat_v=hat_v)
        print("G ", G)
        return G
    elif method =="DRSIS (Lift states)":
        best_s_set = env.lift_stateset()
        print(best_s_set)
        w, rmin, rmax, d0, P, R, hat_q, hat_v, hat_G = get_model(trajectories, H, env.states, env.actions, weighted=False,
                                                                 gamma=gamma, p_e=policy, p_b=behav, JiangStyle=False,
                                                                 negligible_states=best_s_set)
        G = DoublyRobust(trajectories, gamma, p_e=policy, p_b=behav, w=w, hat_q=hat_q, hat_v=hat_v)
        print("G ", G)
        return G
    elif method =="WDRSIS (Lift states)":
        best_s_set = env.lift_stateset()
        print(best_s_set)
        w, rmin, rmax, d0, P, R, hat_q, hat_v, hat_G = get_model(trajectories, H, env.states, env.actions, weighted=True,
                                                                 gamma=gamma, p_e=policy, p_b=behav, JiangStyle=False,
                                                                 negligible_states=best_s_set)
        G = DoublyRobust(trajectories, gamma, p_e=policy, p_b=behav, w=w, hat_q=hat_q, hat_v=hat_v)
        print("G ", G)
        return G
    elif method =="DRSIS (Covariance testing)":
        S_sets = env.candidate_statesets()
        #print("candidates ", S_sets)
        best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=False,epsilon=epsilon_c)
        w, rmin, rmax, d0, P, R, hat_q, hat_v, hat_G = get_model(trajectories, H, env.states, env.actions, weighted=False,
                                                                 gamma=gamma, p_e=policy, p_b=behav, JiangStyle=False,
                                                                 negligible_states=best_s_set)
        G = DoublyRobust(trajectories, gamma, p_e=policy, p_b=behav, w=w, hat_q=hat_q, hat_v=hat_v)
        print("G ", G)
        return G
    elif method =="WDRSIS (Covariance testing)":
        S_sets = env.candidate_statesets()
        #print("candidates ", S_sets)
        best_G, best_s_set = Exhaustive_SIS(trajectories, S_sets, p_e=policy, p_b=behav, weighted=True,epsilon=epsilon_c)
        w, rmin, rmax, d0, P, R, hat_q, hat_v, hat_G = get_model(trajectories, H, env.states, env.actions, weighted=True,
                                                                 gamma=gamma, p_e=policy, p_b=behav, JiangStyle=False,
                                                                 negligible_states=best_s_set)
        G = DoublyRobust(trajectories, gamma, p_e=policy, p_b=behav, w=w, hat_q=hat_q, hat_v=hat_v)
        print("G ", G)
        return G
    elif method =="DRSIS (Q-based)":
        _w, _rmin, _rmax, _d0, _P, _R, hat_q, hat_v, hat_G = get_model(trajectories, H, env.states, env.actions, weighted=False, gamma=gamma, p_e=policy, p_b=behav, JiangStyle=JiangStyle)
        S_A = get_Q_negligible_states(epsilon=epsilon_q, states=env.states, actions=env.actions, Q=hat_q)
        print("state set ", S_A)
        w,_rmin,_rmax = get_DR_params(trajectories,H,weighted=False,p_e=policy, p_b=behav,negligible_states=S_A)
        print("hatG from model", hat_G)
        G = DoublyRobust(trajectories, gamma, p_e=policy, p_b=behav, w=w, hat_q=hat_q, hat_v=hat_v)
        print("G ", G)
        return G
    elif method =="WDRSIS (Q-based)":
        _w, _rmin, _rmax, _d0, _P, _R, hat_q, hat_v, hat_G = get_model(trajectories, H, env.states, env.actions, weighted=True, gamma=gamma, p_e=policy, p_b=behav, JiangStyle=JiangStyle)
        S_A = get_Q_negligible_states(epsilon=epsilon_q, states=env.states, actions=env.actions, Q=hat_q)
        print("state set ", S_A)
        w,_rmin,_rmax = get_DR_params(trajectories,H,weighted=True,p_e=policy, p_b=behav,negligible_states=S_A) # use these weights
        print("hatG from model", hat_G)
        G = DoublyRobust(trajectories, gamma, p_e=policy, p_b=behav, w=w, hat_q=hat_q, hat_v=hat_v)
        print("G ", G)
        return G
    else:
        raise Exception("method ", method, " currently not supported.")
    # print("WPDIS")
    # WPDIS_score = WPDIS(trajectories, p_e=policy, p_b=behav, period=period)

    # print("WIS")
    # WIS_scores.append(WIS(trajectories, p_e=policy, p_b=behav)[-1])