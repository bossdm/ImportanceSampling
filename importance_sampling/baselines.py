import numpy as np

def IS(trajectories,p_e=None,p_b=None,period=float("inf")):
    printfile=open("../importance_trajectories.txt", "w")
    E_G = 0
    scores=[]
    for i, trajectory in enumerate(trajectories):
        G=0
        importance = 1
        if p_e is None and p_b is None:
            for (s,a,r, rho) in trajectory:
                importance*= rho
                #printfile.write("%.4f,%.4f,%.4f,"%(p_e[s][a],p_b[s][a],r))
                G += r
        else:
            for (s,a,r) in trajectory:
                importance*= p_e[s][a]/p_b[s][a]
                #printfile.write("%.4f,%.4f,%.4f,"%(p_e[s][a],p_b[s][a],r))
                G += r
        printfile.write("\n")
        G = importance*G
        E_G+=G
        if i > 0 and i % period == 0:
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
        if p_e is None and p_b is None:
            for (s,a,r, rho) in trajectory:
                importance*= rho
                G += r
        else:
            for (s,a,r) in trajectory:
                importance*= p_e[s][a]/p_b[s][a]
                G += r
        G = importance*G
        SW+=importance
        E_G+=G
        if i > 0 and i % period == 0:
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
            if p_e is None and p_b is None:
                for (s,a,r,rho) in trajectory[0:t]:
                    importance_prod = importance_prod * rho
            else:
                for (s,a,r) in trajectory[0:t]:
                    importance_prod = importance_prod * p_e[s][a]/p_b[s][a]
            # use the last r
            G += importance_prod * r
        E_G+=G
        if i > 0 and i % period == 0:
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
                if p_e is None and p_b is None:
                    s,a,r,ro = traj[t]
                else:
                    s,a,r = traj[t]
                    ro = p_e[s][a] / p_b[s][a]
                ro_cum[i] *= ro
        SW = np.sum(ro_cum)
        for i in range(len(trajectories)):
            traj = trajectories[i]
            if len(traj) > t:
                if p_e is None and p_b is None:
                    s, a, r, ro = traj[t]
                else:
                    s, a, r = traj[t]

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

