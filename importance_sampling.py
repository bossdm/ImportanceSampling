import numpy as np

def IS(trajectories,p_e,p_b,period=float("inf")):
    printfile=open("importance_trajectories.txt","w")
    E_G = 0
    scores=[]
    for i, trajectory in enumerate(trajectories):
        G=0
        importance = 1
        for (s,a,r) in trajectory:
            importance*= p_e[s][a]/p_b[s][a]
            printfile.write("%.4f,%.4f,%.4f,"%(p_e[s][a],p_b[s][a],r))
            G += r
        printfile.write("\n")
        G = importance*G
        E_G+=G
        if i % period == 0:
            scores.append(E_G/(i+1))
    scores.append(E_G / len(trajectories))
    print("IS ", scores[-1])
    return scores

def WIS(trajectories,p_e,p_b,period=float("inf")):
    E_G = 0
    SW = 0
    scores=[]
    for i, trajectory in enumerate(trajectories):
        G=0
        importance = 1
        for (s,a,r) in trajectory:
            importance*= p_e[s][a]/p_b[s][a]
            G += r
        G = importance*G
        SW+=importance
        E_G+=G
        if i % period == 0:
            print(i)
            scores.append(E_G/SW)
    scores.append(E_G/SW)
    print("WIS ", scores[-1])
    return scores

def PDIS(trajectories,p_e,p_b,period=float("inf")):
    E_G = 0
    scores=[]
    for i, trajectory in enumerate(trajectories):
        G=0
        for t in range(1,len(trajectory)+1):
            importance_prod = 1
            for (s,a,r) in trajectory[0:t]:
                importance_prod = importance_prod * p_e[s][a]/p_b[s][a]
            # use the last r
            G += importance_prod * r
        E_G+=G
        if i % period == 0:
            scores.append(E_G/(i+1))
    scores.append(E_G / len(trajectories))
    print("PDIS ",scores[-1])
    return scores
#
def WPDIS(trajectories, p_e, p_b, period=float("inf")):
    ro_cum = np.ones(shape=(len(trajectories),)) # cumulant vector for ro_t
    r_t = np.zeros(shape=(len(trajectories),))  # cumulant vector for ro_t
    E_G =  0
    H=0
    for traj in trajectories:
        if len(traj) > H:
            H = len(traj)

    for t in range(H):
        # initialise ro-vector with all ones
        for i in range(len(trajectories)):
            traj = trajectories[i]
            if len(traj) > t:
                s,a,r = traj[t]
                ro = p_e[s][a] / p_b[s][a]
                ro_cum[i] *= ro
        SW = np.sum(ro_cum)
        for i in range(len(trajectories)):
            traj = trajectories[i]
            if len(traj) > t:
                s,a,r = traj[t]
                E_G += ro_cum[i] / SW  * r
    print("WPDIS ", E_G)
    return E_G
# def WPDIS(trajectories,p_e,p_b,period=float("inf")):
#     E_G = 0
#     SW = 0
#     scores=[]
#     for i, trajectory in enumerate(trajectories):
#         G=0
#         for t in range(1,len(trajectory)+1):
#             importance_prod = 1
#             for (s,a,r) in trajectory[0:t]:
#                 importance_prod = importance_prod * p_e[s][a]/p_b[s][a]
#             # use the last r
#             G += importance_prod * r
#             SW+=importance_prod
#         E_G += G
#         if i % period == 0:
#             scores.append(E_G / SW)
#     scores.append(E_G / SW)
#     print("WPDIS ", scores[-1])
#     return scores

# def Exhaustive_SIS(trajectories,SA_sets,p_e,p_b,period=float("inf")):
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

def Exhaustive_SIS(trajectories,S_sets,p_e,p_b,weighted=False,period=float("inf")):
    """
    exhaustively search for the best drop across sets of SA-pairs
    :param trajectories:
    :param SAs:
    :param p_e:
    :param p_b:
    :param period:
    :return:
    """
    epsilon=0.001
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
        print("G =", G)
        print("V=",V)
        print("C=", C)
        hatA = np.mean(As)
        print("hatA=",hatA)
        if hatA > 2 or hatA < 0.50:  # don't consider these
            continue
        # if C[0,1] >= 0.5*V:  # don't consider these
        #     continue
        print("Ss ",Ss)
        if MSE < best_MSE + epsilon:
            if len(Ss) > best_setsize:
                best_setsize=len(Ss)
                best_MSE = MSE
                best_s_set = Ss
                best_G = G
    print("best_G", best_G)
    print("best_state_set",best_s_set)
    return best_G, best_s_set


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