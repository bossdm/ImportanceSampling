import numpy as np


def get_A_B_r(k,t,trajectory,p_e,p_b,negligible_states):
    if t >= len(trajectory) :  # do not include terminal state TODO: what is done here?
        return 1, 1, 0
    else:
        A=1
        B=1
        if p_e is None and p_b is None:
            for (s, a, r, ro) in trajectory[0:t - k]:
                A *= ro
            for (s, a, r, ro) in trajectory[t - k:t+1]:
                if s in negligible_states:
                    A*=ro
                else:
                    B *= ro
        else:
            for (s, a, r) in trajectory[0:t - k]:
                ro = p_e[s][a] / p_b[s][a]
                A *=  ro
            for (s, a, r) in trajectory[t - k:t+1]:
                ro = p_e[s][a] / p_b[s][a]
                if s in negligible_states:
                    A *= ro
                else:
                    B *= ro

        return A, B, r  # takes the very last reward (r_t)

def get_MSE(k,t,trajectories,p_e,p_b,negligible_states):
    eps = 0.05
    # empty list for new k
    As = []
    Bs = []
    rs = []
    # k < t
    for i, trajectory in enumerate(trajectories):
        A, B, r = get_A_B_r(k, t, trajectory, p_e, p_b,negligible_states)
        As.append(A)
        Bs.append(B)
        rs.append(r)
    Br = np.array(Bs) * np.array(rs)
    V = np.var(Br)
    C = np.cov(As, Br)
    # print("Br",np.mean(Br))
    # print("avg A",np.mean(A))
    # print("C", np.mean(C))
    SW = sum(Bs)
    # if np.abs(np.mean(A) - 1.0) > eps:
    #     return float("inf"), Br, SW  # never choose these when mean is not close to 1.0
    MSE = V + C[0, 1] ** 2

    # print(MSE)
    # print(Br)
    # print(SW)
    return MSE, Br, SW
def INCRIS(trajectories,p_e,p_b,H,max_t=10,weighted=False,negligible_states=[]):
    """
    exhaustively search for the best drop across sets of SA-pairs
    :param trajectories:
    :param SAs:
    :param p_e:
    :param p_b:
    :param period:
    :return:
    """

    G = 0
    best_ks=[]
    for t in range(0,H+1):
        best_MSE = float("inf")
        best_k = None
        for k in range(0,min(max_t,t+1)): # loop over possible k
            MSE, Br, SW = get_MSE(k, t, trajectories,p_e,p_b,negligible_states)
            if MSE < best_MSE:
                best_k = k
                best_MSE = MSE
                if weighted:
                    r_t = np.sum(Br) / SW
                else:
                    r_t = np.mean(Br)
        G+=r_t
        best_ks.append(best_k)
        #print("best k", best_k)
    return G, best_ks

# def INCRIS_Gs(trajectories,p_e,p_b,H,best_ks,weighted=False,period=float("inf")):
#     G_scores= []
#     for num_traj in range(period,len(trajectories)+period,period):
#         G = 0
#         for t in range(H):
#             MSE, Br = get_MSE(best_ks[t], t, trajectories[0:num_traj], p_e, p_b)
#             G_t = np.mean(Br)
#             G+=G_t
#         G_scores.append(G)
#     print("INCRIS ",G_scores[-1])
#     return G_scores, best_ks