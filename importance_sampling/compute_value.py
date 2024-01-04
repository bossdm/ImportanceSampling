import numpy as np
def compute_value(d0, P, R, gamma, states,actions,H,p_e):
    numStates = len(states)
    numNextStates = numStates + 1
    numActions = len(actions)
    print("S=",numStates,"A=",numActions)
    if p_e is None:
        raise Exception("p_e is None. must provide p_e to evaluate it!")
    else:
        pi_e = np.array(p_e)
    # compute Q_t(s,a) = sum_{s_next} ( average reward + average next Q-val )
    Q = np.zeros((H,numStates,numActions))
    for t in range(H-1,-1,-1):
        if len(R.shape) == 2:
            Q[t, :,:] += R[:,:]  # current average reward
        for s_next in range(numNextStates):
            if len(R.shape) == 3:
                Q[t, :,:] += P[:,:, s_next] * R[:,:, s_next]  # current average reward
            if s_next != numStates and t != H - 1:  # add next average Q value
                Q[t, :,:] += gamma * P[:,:, s_next] * (pi_e[s_next, :].dot(
                    Q[t + 1, s_next, :]))  # gamma is different from MAGIC code here
        # for s in range(numStates):
        #     for a in range(numActions):
        #         if len(R.shape) == 2:
        #             Q[t, s, a] += R[s, a]  # current average reward
        #         for s_next in range(numNextStates):
        #             if len(R.shape) == 3:
        #                 Q[t, s, a] += P[s, a, s_next] * R[s, a, s_next]  # current average reward
        #             if s_next != numStates and t != H - 1:  # add next average Q value
        #                 Q[t, s, a] += gamma * P[s, a, s_next] * pi_e[s_next, :].dot(
        #                     Q[t + 1, s_next, :])  # gamma is different from MAGIC code here
        #print("Q[",t, "] = ",  Q[t])
        #print(t)

    # compute V(s) = sum_a pi_e(a | s) * Q(s)
    V = np.zeros((H,numStates))
    for t in range(H):
        for s in range(numStates):
            V[t,s] = pi_e[s, :].dot(Q[t,s,:])
        #print("V[",t,"]=", V[t])
    # compute G (the expected return from the initial state distribution
    G=0
    for s in range(numStates):
        G+=d0[s]*V[0,s]

    # finished computing
    print("values computed ")


    return Q, V, G