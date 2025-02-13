# State-based Importance Sampling

This repository uses off-policy evaluation for reinforcement learning using importance sampling techniques.

The code allows running various estimators including
* ordinary importance sampling
* per-decision importance sampling
* incremental importance sampling
* stationary density ratio estimation
* doubly robust estimator

and introduces a new variance reduction technique called 
State-based Importance Sampling that is easily 
added to these as variants. The technique is based on removing 
"negligible states" from the
estimator, which are defined as states that have limited to no impact on the 
expected return.

For the most recent paper on state-based importance sampling, which also uses this code, please see:

David M. Bossens & Philip S. Thomas (2024). Low Variance Off-policy Evaluation with State-based Importance Sampling. 
IEEE Conference on Artificial Intelligence (CAI 2024). Available at 
https://arxiv.org/abs/2212.03932 and 
https://ieeexplore.ieee.org/abstract/document/10605477


There are currently three scripts:
* one_D_Domain.py : to run experiments on lift domains.
* IM_domain.py : to run experiments on inventory management. based on the RCMDP repository https://github.com/bossdm/RCMDP/blob/main/InventoryManagement.py
* taxi/run_exp.py : to run experiments on taxi. based on https://github.com/zt95/infinite-horizon-off-policy-estimation/tree/master/taxi
