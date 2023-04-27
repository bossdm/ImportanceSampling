import numpy as np
# a*b*c = exp(log(a) + log(b) + log(c))
def get_trajectory_data(trajectory,p_e,p_b,Ss):
    # variance is the fluctuation in frequency of SAs between trajectories
    A = 1
    B = 1
    G_temp = 0
    if p_e is None and p_b is None:
        for (s, a, r, ro) in trajectory:
            # lefthand side term: variance is based on fluctuations in how often the set of SA-pairs occurs, this can be known from the trajectories
            if s in Ss:  # random variable
                A *= ro
            else:
                B *= ro
            G_temp += r
    else:
        for (s, a, r) in trajectory:
            ro = p_e[s][a] / p_b[s][a]
            # lefthand side term: variance is based on fluctuations in how often the set of SA-pairs occurs, this can be known from the trajectories
            if s in Ss:  # random variable
                A *= ro
            else:
                B *= ro
            G_temp += r
    return A,B,G_temp
def get_trajectory_data_log(trajectory,p_e,p_b,Ss):
    # variance is the fluctuation in frequency of SAs between trajectories
    A = 0
    B = 0
    G_temp = 0
    if p_e is None and p_b is None:
        for (s, a, r, ro) in trajectory:
            # lefthand side term: variance is based on fluctuations in how often the set of SA-pairs occurs, this can be known from the trajectories
            if s in Ss:  # random variable
                A += np.log(ro)
            else:
                B *= np.log(ro)
            G_temp += r
    else:
        for (s, a, r) in trajectory:
            ro = p_e[s][a] / p_b[s][a]
            # lefthand side term: variance is based on fluctuations in how often the set of SA-pairs occurs, this can be known from the trajectories
            if s in Ss:  # random variable
                A += np.log(ro)
            else:
                B += np.log(ro)
            G_temp += r
    A = np.exp(A)
    B = np.exp(B)
    return A,B,G_temp

def get_SIS_estimate(trajectories,p_e,p_b,Ss,weighted,period=float("inf")):
    As = []
    Bs = []
    rs = []
    scores = []
    SW = 0
    for i, trajectory in enumerate(trajectories):
        A,B,G_temp = get_trajectory_data(trajectory,p_e,p_b,Ss)
        As.append(A)
        SW += B
        Bs.append(B)
        rs.append(G_temp)

        Br = np.array(Bs) * np.array(rs)  # estimated returns at a (s,a)-set
        if i > 0 and i % period == 0:

            if weighted:
                E_G = np.sum(Br) / SW
            else:
                E_G = np.mean(Br)
            scores.append(E_G)

    return As,Bs,rs,Br,SW,scores


# def get_trajectory_data_SPDIS(subtrajectory,p_e,p_b,Ss):
#     A = 1
#     B = 1
#     r = subtrajectory[-1][2] # last reward of the subtrajectory
#     for (s, a, r) in subtrajectory: # trajectory until time t-1
#         ro = p_e[s][a] / p_b[s][a]
#         # lefthand side term: variance is based on fluctuations in how often the set of SA-pairs occurs, this can be known from the trajectories
#         if s in Ss:  # random variable
#             A *= ro
#         else:
#             B *= ro
#     return A,B,r
#
# def get_SPDIS_estimate(trajectories,p_e,p_b,Ss,weighted,H, max_t, period=float("inf")):
#     As = []
#     Bs = []
#     rs = []
#     scores = []
#     SW = 0
#     for t in range(0, H + 1):  # now weight associated with each time step
#         k = min(t,max_t)
#         for i, trajectory in enumerate(trajectories):
#             A,B,r = get_A_B_r(k,t,trajectory,p_e,p_b,Ss)
#             As.append(A)
#             Bs.append(B)
#             rs.append(r)
#         Br = np.array(Bs) * np.array(rs)
#         V = np.var(Br)
#         C = np.cov(As, Br)
#         # print("Br",np.mean(Br))
#         # print("avg A",np.mean(A))
#         # print("C", np.mean(C))
#         SW = sum(Bs)
#         # if np.abs(np.mean(A) - 1.0) > eps:
#         #     return float("inf"), Br, SW  # never choose these when mean is not close to 1.0
#         MSE = V + C[0, 1] ** 2
#
#         # print(MSE)
#         # print(Br)
#         # print(SW)
#         return MSE, Br, SW
#
def get_MSE(As,Br,weighted,SW):
    C = np.cov(As, Br)
    V = np.var(Br)
    MSE = V + C[0, 1] ** 2  # MSE is variance + bias^2
    if weighted:
        G = np.sum(Br)/SW
    else:
        G = np.mean(Br)
    hatA = np.mean(As)
    return G, MSE, hatA, C, V

