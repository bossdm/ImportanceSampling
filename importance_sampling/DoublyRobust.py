
import numpy as np

def get_DR_params(trajectories,H,weighted,p_e,p_b):
    n = len(trajectories)
    w=np.zeros((H,n))
    ro_product = np.zeros((H,n))
    r_min=float("inf")
    r_max=-float("inf")
    for t in range(H):
        for i,traj in enumerate(trajectories):
            if t >= len(traj):
                ro_product[t, i] = ro_product[t-1,i]  # treated as absorbing state
            else:
                if p_e is None and p_b is None:
                    (s, a, r, ro) = traj[t]
                else:
                    (s, a, r) = traj[t]
                    ro = p_e[s][a]/p_b[s][a]
                ro_product[t,i] = ro_product[t - 1,i] * ro
            if r < r_min:
                r_min = r
            else:
                if r > r_max:
                    r_max = r

            if  weighted:
                s = sum(ro_product[t,i])
                w[t,:] = ro_product[t,i] / s
            else:
                w[t,:] = ro_product[t,i] / n
    return w, r_min, r_max

def get_DR_model(states, actions, trajectories, p_e,p_b,rmin):
    """
    """
    numStates=len(states)
    numNextStates =len(states) + 1  # treat both terminals as 1 state
    numActions=len(actions)
    stateActionCounts=np.zeros((numStates,numActions))
    stateActionCounts_includingHorizon=np.zeros((numStates,numActions))
    stateActionStateCounts=np.zeros((numStates,numActions,numNextStates))
    stateActionStateCounts_includingHorizon=np.zeros((numStates,numActions,numNextStates))
    P=np.zeros((numStates,numActions,numNextStates))
    R=np.zeros((numStates,numActions,numNextStates))
    d0=np.zeros(numStates) # starting distribution
    n=len(trajectories)
    # compute counts, cumulative reward, and initial distribution
    for i, traj in enumerate(trajectories):
        for j, step in enumerate(traj):
            if p_e is None and p_b is None:
                (s, a, r, ro) = step
                if j == len(traj) - 1:
                    s_next = numStates # single terminal state
                else:
                    (s_next, _a, _r, _ro) = traj[j+1]
            else:
                (s, a, r) = step
                if j == len(traj) - 1:
                    s_next = numStates # single terminal state
                else:
                    (s_next, _a, _r) = traj[j+1]
            if j == 0:
                d0[s] += 1. / n
            stateActionCounts[s,a]+=1
            stateActionStateCounts[s,a,s_next]+=1
            stateActionCounts_includingHorizon[s,a] +=1
            stateActionStateCounts_includingHorizon[s,a,s_next] +=1
            R[s,a,s_next] += r

    # Compute P and normalise R
    for s in range(numStates):
        for a in range(numActions):
            for s_next in range(numNextStates):
                if stateActionCounts[s,a] == 0:
                    P[s,a,s_next] = 1 if s_next == s else 0# use self transition
                else:
                    P[s,a,s_next] = stateActionStateCounts[s,a,s_next] / stateActionCounts[s][a]

                if stateActionStateCounts_includingHorizon[s,a,s_next] == 0:
                    R[s,a,s_next] = 0  # not using Jiang rmin style
                else:
                    R[s,a,s_next] /= stateActionStateCounts_includingHorizon[s,a,s_next]
    print("model computed ")
    #print("d0",d0)
    #print("P",P)
    #print("R",R)
    return d0, P, R

def get_DR_hatq_hatv(d0, P, R, gamma, states,actions,H,p_e):
    numStates = len(states)
    numNextStates = numStates + 1
    numActions = len(actions)

    if p_e is None:
        raise Exception("p_e is None. must provide p_e to evaluate it!")
    else:
        pi_e = np.array(p_e)
    # compute Q_t(s,a) = sum_{s_next} ( average reward + average next Q-val )
    Q = np.zeros((H,numNextStates,numActions))
    for t in range(H-1,-1,-1):
        for s in range(numStates):
            for a in range(numActions):
                for s_next in range(numNextStates):
                    Q[t,s,a] += P[s,a,s_next] * R[s,a,s_next]  # current average reward
                    if s_next != numStates and t != H - 1:   # add next average Q value
                        Q[t,s, a]+= gamma * P[s,a,s_next] * pi_e[s_next,:].dot(Q[t+1,s_next,:])  # gamma is different from MAGIC code here



    # compute V(s) = sum_a pi_e(a | s) * Q(s)
    V = np.zeros((H,numNextStates))
    for t in range(H):
        for s in range(numStates):
            V[t,s] = pi_e[s, :].dot(Q[t,s,:])
    # compute G (the expected return from the initial state distribution
    G=0
    for s in range(numStates):
        G+=d0[s]*V[0,s]

    # finished computing
    #print("values computed ")
    #print("Q",Q)
    #print("V",V)
    print("DR ",G)
    return Q, V, G

def DoublyRobust(trajectories, H, states, actions, weighted, gamma, p_e, p_b):

    w, rmin, rmax = get_DR_params(trajectories, H, weighted, p_e, p_b)
    d0, P, R = get_DR_model(states, actions, trajectories, p_e, p_b, rmin)
    hat_q,hat_v,_hat_G = get_DR_hatq_hatv(d0, P, R, gamma, states,actions,H,p_e)
    G = 0
    n=len(trajectories)
    w = np.zeros((H,n))
    for i,traj in enumerate(trajectories):
        curGamma = 1.0
        for t, step in enumerate(traj):
            if p_e is None and p_b is None:
                (s, a, r, ro) = step
            else:
                (s, a, r) = step
            G += curGamma * w[t, i] * r

            if t == 0:
                w2 = 1.0 / n
            else:
                w2 = w[t - 1, i]
            G -= curGamma * (w[t, i] * hat_q[t,s,a] - w2 * hat_v[t,s])  # correction with the discounted cumulative advantage based on reward model
            # based on model (up to that time)
            curGamma *= gamma

    return G