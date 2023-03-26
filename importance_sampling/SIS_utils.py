import numpy as np

def get_SIS_estimate(trajectories,p_e,p_b,Ss,weighted,period=float("inf")):
    As = []
    Bs = []
    rs = []
    scores = []
    SW = 0
    for i, trajectory in enumerate(trajectories):
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
        As.append(A)
        SW += B
        Bs.append(B)
        rs.append(G_temp)

        if weighted:
            for i, traj in enumerate(trajectories):
                Bs[i] /= SW

        Br = np.array(Bs) * np.array(rs)  # estimated returns at a (s,a)-set
        if i > 0 and i % period == 0:

            if weighted:
                E_G = np.sum(Br) / SW
            else:
                E_G = np.mean(Br)
            scores.append(E_G)

    return As,Bs,rs,Br,SW,scores

def get_MSE(As,Br,weighted):
    C = np.cov(As, Br)
    V = np.var(Br)
    MSE = V + C[0, 1] ** 2  # MSE is variance + bias^2
    if weighted:
        G = np.sum(Br)
    else:
        G = np.mean(Br)
    hatA = np.mean(As)
    return G, MSE, hatA, C, V

