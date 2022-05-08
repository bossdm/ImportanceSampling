
from importance_sampling import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import pickle
    trajectories, behav_score = pickle.load(open("data_maze_behavpol.pkl","rb"))
    traj_eval, eval_score = pickle.load(open("data_maze_evalpol.pkl", "rb"))
    states = set([])
    for trajectory in trajectories:
        for (s,a,r,ro) in trajectory:
            states.add(s)

    MC_iterations = len(trajectories)
    print("eval policy ", eval_score)
    print("behav policy ", behav_score)

    #data = trajectories,behav_score,eval_score
    #pickle.dump(data,open("data.pkl","wb"))
    #trajectories, behav_score, eval_score = pickle.load(open("data.pkl","rb")
    period=1000
    num_plotpoints = MC_iterations // period
    x = [i * period for i in range(num_plotpoints+1)]

    print("WPDIS")
    WPDIS_scores = WPDIS(trajectories, p_e=None, p_b=None, period=period)

    print("WIS")
    WIS_scores = WIS(trajectories, p_e=None, p_b=None, period=period)

    print("PDIS")
    PDIS_scores = PDIS(trajectories, p_e=None, p_b=None, period=period)

    print("IS")
    IS_scores =  IS(trajectories, p_e=None, p_b=None, period=period)

    print("Exhaustive SIS")
    SA_sets=[[]] + [[-1,1]] + [[0]] + [[1]] + [[2]] + [[3]] + [[-2,2]] + [[-3,3]]
    #SA_sets=[[]] + [[(s,a)] for s,a in lifts_int.items()] + [[(s,a) for s,a in lifts_int.items()]] + [[(s,a)] for s in states for a,act in enumerate(actions)]
    best_G, best_s_set = Exhaustive_SIS(trajectories, SA_sets, p_e=None, p_b=None, weighted=False)
    SIS_scores = SIS(trajectories, best_s_set, p_e=None, p_b=None, weighted=False,period=period)

    best_G, best_s_set = Exhaustive_SIS(trajectories, SA_sets, p_e=None, p_b=None,weighted=True)
    WSIS_scores = SIS(trajectories,best_s_set,p_e=None,p_b=None,weighted=True,period=period)

    #Exhaustive_SPDIS(trajectories, SA_sets, p_e=policy, p_b=None)
    line1, = plt.plot(x,IS_scores,marker="v")
    line2, = plt.plot(x,WIS_scores,marker="o")
    line3, = plt.plot(x,PDIS_scores,marker="x")
    line4, = plt.plot(x,SIS_scores,marker="D")
    line5, = plt.plot(x, WSIS_scores, marker="X")
    plt.legend([line1,line2,line3,line4,line5],["IS","WIS","PDIS","SIS","WSIS"])