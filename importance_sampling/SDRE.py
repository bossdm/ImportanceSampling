from taxi.run_exp import off_policy_evaluation_density_ratio,train_density_ratio
from taxi.Density_Ratio_discrete import *

def SDRE(trajectories,p_e,p_b,n_states,negligible_states=[]):
    den_discrete = Density_Ratio_discounted(n_states, gamma=1.0)
    SASR = []
    for traj in trajectories:
        sasr = []
        for t in range(len(traj)):
            s,a,r = traj[t]
            if t == len(traj) - 1:
                s_n = s   # no terminal states
            else:
                s_n, _, _ = traj[t+1]
            sasr.append((s,a,s_n,r))
        SASR.append(sasr)
    H = max(len(sasr) for sasr in SASR)
    x, w = train_density_ratio(SASR, p_b, p_e, den_discrete, gamma=1.0)
    w = w.reshape(-1)
    est_SDRE = off_policy_evaluation_density_ratio(SASR, p_b, p_e, w, 1.0,negligible_states) *H
    return est_SDRE