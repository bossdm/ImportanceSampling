import numpy as np
from importance_sampling.SIS_utils import *
from importance_sampling.compute_value import compute_value

def Exhaustive_SIS(trajectories,S_sets,p_e,p_b,weighted=False,epsilon=0.01):
    """
    exhaustively search for the best drop across sets of SA-pairs
    :param trajectories:
    :param SAs:
    :param p_e:
    :param p_b:
    :param period:
    :return:
    """
    epsilon_s = 0.01
    #writefile=open("SIS_log.txt","w")
    #writefile.write("S^A \t G \t V \t C \t hatA\n")
    best_setsize=-1
    best_MSE = float("inf")
    for Ss in S_sets:
        As, Bs, rs, Br, SW, scores = get_SIS_estimate(trajectories, p_e, p_b, Ss, weighted, period=float("inf"))
        G, MSE, hatA, C, V = get_MSE(As, Br, weighted, SW)
        #writefile.write(str(Ss) + "\t" + str(G) + "\t" + str(V) + "\t" + str(C[0, 1]) + "\t" + str(hatA) + "\n")
        # writefile.write(str(Ss) + "\t" + str(G) + "\t" + str(V) + "\t" + str(C[0,1]) + "\t" + str(hatA) + "\n")
        # if hatA > (1 + epsilon) or hatA < (1 - epsilon):  # don't consider these
        #     continue
        if np.abs(C[0, 1]) >= epsilon:  # don't consider these
            continue
        #     continue

        if MSE < best_MSE or MSE < best_MSE * (1 + epsilon_s) and len(Ss) > best_setsize:
            best_setsize = len(Ss)
            best_MSE = MSE
            best_s_set = Ss
            best_G = G
    print("best_G", best_G)
    print("best_state_set",best_s_set)
    return best_G, best_s_set

def get_Q_negligible_states(epsilon,states,actions,Q):
    Q_neg_states=[]
    H=len(Q)
    for s in range(len(states)):
        is_negligible=True
        for t in range(H):
            for i in range(1,len(actions)):
                for j in range(i):   #check all pair
                    if np.abs(Q[t,s,i] - Q[t,s,j]) >= epsilon:
                        is_negligible=False
                        break
                if not is_negligible:
                        break
            if not is_negligible:
                break
        if is_negligible:
            Q_neg_states.append(s)

    return Q_neg_states

def Qvalue_SIS(hat_q,trajectories,epsilon,states,actions,p_e,p_b,weighted=False):
    """
    search for epsilon-negligibility based on Q-values
    :return:
    """
    S_A =  get_Q_negligible_states(epsilon,states,actions,hat_q)
    print("state set ",S_A)
    scores = SIS(trajectories,S_A,p_e,p_b,weighted)
    return scores

def SIS(trajectories,Ss,p_e,p_b,weighted=False):
    """
    SIS for a particular given state-set
    :param trajectories:
    :param SAs:
    :param p_e:
    :param p_b:
    :param period:
    :return:
    """

    As,Bs,rs,Br,SW,scores = get_SIS_estimate(trajectories, p_e, p_b, Ss, weighted, period=float("inf"))
    if weighted:
        E_G = np.sum(Br) / SW
    else:
        E_G = np.mean(Br)
    scores.append(E_G)
    return scores
#
# def Exhaustive_SPDIS(trajectories,S_sets,p_e,p_b,H,max_t=10):
#     """
#     exhaustively search for the best drop across sets of SA-pairs
#     :param trajectories:
#     :param SAs:
#     :param p_e:
#     :param p_b:
#     :param period:
#     :return:
#     """
#     best_G = -float("inf")
#     best_MSE = float("inf")
#     for Ss in S_sets:
#         As = []
#         Brs = []
#         rs = []
#         for i, trajectory in enumerate(trajectories):
#
#             As.append(A)
#             Brs.append(Br)
#             #rs.append(G_temp)
#
#
#         #Br=np.array(Brs)*np.array(rs)   # estimated returns at a (s,a)-set
#         C= np.cov(As,Brs)
#         V = np.var(Brs)
#         MSE = V + C[0,1]**2 # MSE is variance + bias^2
#         G = np.mean(Brs)
#         print("state-set",Ss)
#         print("G =", G)
#         print("V=",V)
#         print("C=", C)
#         hatA = np.mean(As)
#         print("hatA=",hatA)
#         if MSE < best_MSE:
#             best_MSE = MSE
#             best_s_set = Ss
#             best_G = G
#     print("best_G", best_G)
#     print("best_state_set",best_s_set)
#     return best_G, best_s_set

# def SPDIS(trajectories,Ss,p_e,p_b,H,max_t=10,weighted=False):
#     """
#     SIS for a particular given state-set
#     :param trajectories:
#     :param SAs:
#     :param p_e:
#     :param p_b:
#     :param period:
#     :return:
#     """
#
#     As,Bs,rs,Br,SW,scores = get_SPDIS_estimate(trajectories, p_e, p_b, Ss, H, max_t, weighted)
#     if weighted:
#         E_G = np.sum(Br) / SW
#     else:
#         E_G = np.mean(Br)
#     scores.append(E_G)
#     return scores

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