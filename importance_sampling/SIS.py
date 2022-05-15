import numpy as np


def Exhaustive_SIS(trajectories,S_sets,p_e,p_b,weighted=False):
    """
    exhaustively search for the best drop across sets of SA-pairs
    :param trajectories:
    :param SAs:
    :param p_e:
    :param p_b:
    :param period:
    :return:
    """
    writefile=open("SIS_log.txt","w")
    writefile.write("S^A \t G \t V \t C \t hatA\n")
    epsilon=0.05
    best_setsize=-1
    best_MSE = float("inf")
    for Ss in S_sets:
        As = []
        Bs = []
        rs = []
        SW=0
        for i, trajectory in enumerate(trajectories):
            # variance is the fluctuation in frequency of SAs between trajectories
            A = 1
            B = 1
            G_temp = 0
            if p_e is None and p_b is None:
                for (s,a,r,ro) in trajectory:
                    # lefthand side term: variance is based on fluctuations in how often the set of SA-pairs occurs, this can be known from the trajectories
                    if s in Ss:     # random variable
                        A*=ro
                    else:
                        B*=ro
                    G_temp += r
            else:
                for (s,a,r) in trajectory:
                    ro = p_e[s][a]/p_b[s][a]
                    # lefthand side term: variance is based on fluctuations in how often the set of SA-pairs occurs, this can be known from the trajectories
                    if s in Ss:     # random variable
                        A*=ro
                    else:
                        B*=ro
                    G_temp += r
            As.append(A)
            SW += B
            Bs.append(B)
            rs.append(G_temp)

        if weighted:
            for i, traj in enumerate(trajectories):
                Bs[i]/=SW

        Br=np.array(Bs)*np.array(rs)   # estimated returns at a (s,a)-set
        C= np.cov(As,Br)
        V = np.var(Br)
        MSE = V + C[0,1]**2 # MSE is variance + bias^2
        if weighted:
            G = np.sum(Br)
        else:
            G = np.mean(Br)
        hatA = np.mean(As)
        writefile.write(str(Ss) + "\t" + str(G) + "\t" + str(V) + "\t" + str(C[0,1]) + "\t" + str(hatA) + "\n")
        if hatA > 1+epsilon or hatA < 1-epsilon:  # don't consider these
            continue
        # if C[0,1] >= 0.5*V:  # don't consider these
        #     continue

        if MSE < best_MSE or MSE < best_MSE*(1+epsilon) and len(Ss) > best_setsize:
                best_setsize=len(Ss)
                best_MSE = MSE
                best_s_set = Ss
                best_G = G
    print("best_G", best_G)
    print("best_state_set",best_s_set)
    return best_G, best_s_set


def SIS(trajectories,Ss,p_e,p_b,weighted=False,period=float("inf")):
    """
    SIS for a particular given state-set
    :param trajectories:
    :param SAs:
    :param p_e:
    :param p_b:
    :param period:
    :return:
    """

    As = []
    Bs = []
    rs = []
    scores=[]
    SW=0
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
        SW += B
        Bs.append(B)
        rs.append(G_temp)
        if i > 0 and i % period == 0:
            Br = np.array(Bs) * np.array(rs)  # estimated returns at a (s,a)-set
            if weighted:
                E_G = np.sum(Br) / SW
            else:
                E_G = np.mean(Br)
            scores.append(E_G)
    #final score
    Br = np.array(Bs) * np.array(rs)  # estimated returns at a (s,a)-set
    if weighted:
        E_G = np.sum(Br) / SW
    else:
        E_G = np.mean(Br)
    scores.append(E_G)
    return scores

def Exhaustive_SPDIS(trajectories,S_sets,p_e,p_b,period=float("inf")):
    """
    exhaustively search for the best drop across sets of SA-pairs
    :param trajectories:
    :param SAs:
    :param p_e:
    :param p_b:
    :param period:
    :return:
    """
    best_G = -float("inf")
    best_MSE = float("inf")
    for Ss in S_sets:
        As = []
        Brs = []
        rs = []
        for i, trajectory in enumerate(trajectories):
            # variance is the fluctuation in frequency of SAs between trajectories
            A = 0 #
            Br = 0 #
            for t in range(1,len(trajectory)+1):
                A_temp = 1
                B_temp = 1
                G_temp = trajectory[t-1][2]
                for (s,a,r) in trajectory[0:t]:
                    ro = p_e[s][a]/p_b[s][a]
                    # lefthand side term: variance is based on fluctuations in how often the set of SA-pairs occurs, this can be known from the trajectories
                    if s in Ss:     # random variable
                        A_temp*=ro
                    else:
                        B_temp*=ro
                A+=A_temp
                Br+=B_temp*G_temp
            As.append(A)
            Brs.append(Br)
            #rs.append(G_temp)


        #Br=np.array(Brs)*np.array(rs)   # estimated returns at a (s,a)-set
        C= np.cov(As,Brs)
        V = np.var(Brs)
        MSE = V + C[0,1]**2 # MSE is variance + bias^2
        G = np.mean(Brs)
        print("state-set",Ss)
        print("G =", G)
        print("V=",V)
        print("C=", C)
        hatA = np.mean(As)
        print("hatA=",hatA)
        if hatA > 2 or hatA < 0.50:  # don't consider these
            continue
        if C >= V:  # don't consider these
            continue
        if MSE < best_MSE:
            best_MSE = MSE
            best_s_set = Ss
            best_G = G
    print("best_G", best_G)
    print("best_state_set",best_s_set)
    return best_G, best_s_set



# def Exhaustive_SAIS(trajectories,SA_sets,p_e,p_b,period=float("inf")):
#     """
#     exhaustively search for the best drop across sets of SA-pairs
#     :param trajectories:
#     :param SAs:
#     :param p_e:
#     :param p_b:
#     :param period:
#     :return:
#     """
#     best_MSE = float("inf")
#     for SAs in SA_sets:
#         As = []
#         Bs = []
#         rs = []
#         for i, trajectory in enumerate(trajectories):
#             # variance is the fluctuation in frequency of SAs between trajectories
#             A = 1
#             B = 1
#             G_temp = 0
#             for (s,a,r) in trajectory:
#                 ro = p_e[s][a]/p_b[s][a]
#                 # lefthand side term: variance is based on fluctuations in how often the set of SA-pairs occurs, this can be known from the trajectories
#                 if (s,a) in SAs:     # random variable
#                     A*=ro
#                 else:
#                     B*=ro
#                 G_temp += r
#
#             As.append(A)
#             Bs.append(B)
#             rs.append(G_temp)
#
#
#         Br=np.array(Bs)*np.array(rs)   # estimated returns at a (s,a)-set
#         C= np.cov(As,Br)
#         V = np.var(Br)
#         MSE = V + C[0,1]**2 # MSE is variance + bias^2
#         G = np.mean(Br)
#         print("sa-set",SAs)
#         print("G =", G)
#         hatA = np.mean(As)
#         print("hatA=",hatA)
#         if MSE < best_MSE:
#             best_MSE = MSE
#             best_sa_set = SAs
#             best_G = G
#     print("best_G", best_G)
#     print("best_sa_set",best_sa_set)
#     return best_G, best_sa_set