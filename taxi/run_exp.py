import os
import sys
import argparse
import optparse
import subprocess
import numpy as np
from taxi.Density_Ratio_discrete import Density_Ratio_discrete, Density_Ratio_discounted
from taxi.Q_learning import Q_learning
from taxi.environment import random_walk_2d, taxi
from importance_sampling.compute_value import  compute_value
from importance_sampling.DoublyRobust import *

from importance_sampling.SIS import get_Q_negligible_states
from importance_sampling.INCRIS import INCRIS
from importance_sampling.baselines import WIS
from utils import print_MSE_rows
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# import seaborn as sns
# sns.set(style="white")
epsilon_Q = 2.0

def roll_out(state_num, env, policy, num_trajectory, truncate_size):
	SASR = []
	total_reward = 0.0
	frequency = np.zeros(state_num)
	for i_trajectory in range(num_trajectory):
		state = env.reset()
		sasr = []
		for i_t in range(truncate_size):
			#env.render()
			p_action = policy[state, :]
			action = np.random.choice(p_action.shape[0], 1, p = p_action)[0]
			next_state, reward = env.step(action)

			sasr.append((state, action, next_state, reward))
			frequency[state] += 1
			total_reward += reward
			#print env.state_decoding(state)
			#a = input()

			state = next_state
		SASR.append(sasr)
	return SASR, frequency, total_reward/(num_trajectory * truncate_size)

def train_density_ratio(SASR, policy0, policy1, den_discrete, gamma):
	for sasr in SASR:
		discounted_t = 1.0
		initial_state = sasr[0][0]
		for state, action, next_state, reward in sasr:
			discounted_t *= gamma
			policy_ratio = policy1[state][action]/policy0[state][action]
			#print(state)
			den_discrete.feed_data(state, next_state, initial_state, policy_ratio, discounted_t)
		den_discrete.feed_data(-1, initial_state, initial_state, 1, 1-discounted_t)
		
	x, w = den_discrete.density_ratio_estimate()
	return x, w

def off_policy_evaluation_density_ratio(SASR, policy0, policy1, density_ratio, gamma,negligible_states):
	total_reward = 0.0
	self_normalizer = 0.0
	for sasr in SASR:
		discounted_t = 1.0
		for state, action, next_state, reward in sasr:
			if state in negligible_states:
				policy_ratio = 1
			else:
				policy_ratio = policy1[state][action]/policy0[state][action]
			total_reward += density_ratio[state] * policy_ratio * reward * discounted_t
			self_normalizer += density_ratio[state] * policy_ratio * discounted_t
			discounted_t *= gamma
	return total_reward / self_normalizer

def on_policy(SASR, gamma):
	total_reward = 0.0
	self_normalizer = 0.0
	for sasr in SASR:
		discounted_t = 1.0
		for state, action, next_state, reward in sasr:
			total_reward += reward * discounted_t
			self_normalizer += discounted_t
			discounted_t *= gamma
	return total_reward / self_normalizer

def importance_sampling_estimator(SASR, policy0, policy1, gamma,negligible_states):
	mean_est_reward = 0.0
	for sasr in SASR:
		log_trajectory_ratio = 0.0
		total_reward = 0.0
		discounted_t = 1.0
		self_normalizer = 0.0
		for state, action, next_state, reward in sasr:
			if state not in negligible_states:
				log_trajectory_ratio += np.log(policy1[state, action]) - np.log(policy0[state, action])
			total_reward += reward * discounted_t
			self_normalizer += discounted_t
			discounted_t *= gamma
		avr_reward = total_reward / self_normalizer
		mean_est_reward += avr_reward * np.exp(log_trajectory_ratio)
	mean_est_reward /= len(SASR)
	return mean_est_reward

def importance_sampling_estimator_stepwise(SASR, policy0, policy1, gamma,negligible_states):
	mean_est_reward = 0.0
	for sasr in SASR:
		step_log_pr = 0.0
		est_reward = 0.0
		discounted_t = 1.0
		self_normalizer = 0.0
		for state, action, next_state, reward in sasr:
			if state not in negligible_states:
				step_log_pr += np.log(policy1[state, action]) - np.log(policy0[state, action])
			est_reward += np.exp(step_log_pr)*reward*discounted_t
			self_normalizer += discounted_t
			discounted_t *= gamma
		est_reward /= self_normalizer
		mean_est_reward += est_reward
	mean_est_reward /= len(SASR)
	return mean_est_reward

def weighted_importance_sampling_estimator(SASR, policy0, policy1, gamma,negligible_states):
	total_rho = 0.0
	est_reward = 0.0
	for sasr in SASR:
		total_reward = 0.0
		log_trajectory_ratio = 0.0
		discounted_t = 1.0
		self_normalizer = 0.0
		for state, action, next_state, reward in sasr:
			if state not in negligible_states:
				log_trajectory_ratio += np.log(policy1[state, action]) - np.log(policy0[state, action])
			total_reward += reward * discounted_t
			self_normalizer += discounted_t
			discounted_t *= gamma
		avr_reward = total_reward / self_normalizer
		trajectory_ratio = np.exp(log_trajectory_ratio)
		total_rho += trajectory_ratio
		est_reward += trajectory_ratio * avr_reward

	avr_rho = total_rho / len(SASR)
	return est_reward / avr_rho/ len(SASR)

def weighted_importance_sampling_estimator_stepwise(SASR, policy0, policy1, gamma,negligible_states):
	Log_policy_ratio = []
	REW = []
	for sasr in SASR:
		log_policy_ratio = []
		rew = []
		discounted_t = 1.0
		self_normalizer = 0.0
		for state, action, next_state, reward in sasr:
			if state in negligible_states:
				#print("negligible ", state)
				log_pr = 0
			else:
				#print("non-negligible ", state)
				log_pr = np.log(policy1[state, action]) - np.log(policy0[state, action])
				#print(log_pr)
			if log_policy_ratio:
				log_policy_ratio.append(log_pr + log_policy_ratio[-1])
			else:
				log_policy_ratio.append(log_pr)
			rew.append(reward * discounted_t)
			self_normalizer += discounted_t
			discounted_t *= gamma
		Log_policy_ratio.append(log_policy_ratio)
		REW.append(rew)
	est_reward = 0.0
	rho = np.exp(Log_policy_ratio)
	#print 'rho shape = {}'.format(rho.shape)
	REW = np.array(REW)
	for i in range(REW.shape[0]):
		est_reward += np.sum(rho[i]/np.mean(rho, axis = 0) * REW[i])/self_normalizer
	return est_reward/REW.shape[0]

def WIS_check(SASR, policy0, policy1, gamma):
	trajectories = [[(s, a, r) for s, a, ns, r in sasr] for sasr in SASR]
	H = max(len(sasr) for sasr in SASR)
	G= WIS(trajectories, policy1, policy0)
	return G[-1] /H
def INCRIS_estimator(SASR, policy0, policy1, gamma, negligible_states):
	# convert trajectories to sar trajectories
	trajectories=[[(s,a,r) for s, a, ns, r in sasr] for sasr in SASR]
	H = max(len(sasr) for sasr in SASR)
	G, _ = INCRIS(trajectories,policy1,policy0,H,max_t=10,weighted=False,negligible_states=negligible_states)
	return G/H

def weighted_INCRIS_estimator(SASR, policy0, policy1, gamma, negligible_states):
	# convert trajectories to sar trajectories
	trajectories=[[(s,a,r) for s, a, ns, r in sasr] for sasr in SASR]
	H = max(len(sasr) for sasr in SASR)
	G, _ = INCRIS(trajectories,policy1,policy0,H,max_t=10,weighted=True,negligible_states=negligible_states)
	return G/H

def DR_estimator(SASR, policy0, policy1, gamma, states, actions, negligible_states):
	# convert trajectories to sar trajectories
	trajectories=[[(s,a,r) for s, a, ns, r in sasr] for sasr in SASR]
	H = max(len(sasr) for sasr in SASR)

	if negligible_states:
		_w, _rmin, _rmax, d0, P, R, hat_q, hat_v, hat_G = get_model(trajectories, H,  states,  actions,
																 weighted=True,
																 gamma=gamma, p_e=policy1, p_b=policy0,
																 JiangStyle=False)
		w, _rmin, _rmax = get_DR_params(trajectories, H, weighted=True, p_e=policy1, p_b=policy0,
										negligible_states=negligible_states)  # use these weights
	else:
		w, rmin, rmax, d0, P, R, hat_q, hat_v, hat_G = get_model(trajectories, H,  states,  actions,
																 weighted=True,
																 gamma=gamma, p_e=policy1, p_b=policy0,
																 JiangStyle=False)
	G = DoublyRobust(trajectories, gamma, p_e=policy1, p_b=policy0, w=w, hat_q=hat_q, hat_v=hat_v)
	return G/H

def Q_learning(env, num_trajectory, truncate_size, temperature = 2.0):
	agent = Q_learning(n_state, n_action, 0.01, 0.99)

	state = env.reset()
	for k in range(20):
		print('Training for episode {}'.format(k))
		for i in range(50):
			for j in range(5000):
				action = agent.choose_action(state, temperature)
				next_state, reward = env.step(action)
				agent.update(state, action, next_state, reward)
				state = next_state
		pi = agent.get_pi(temperature)
		np.save('taxi-policy/pi{}.npy'.format(k), pi)
		SAS, f, avr_reward = roll_out(n_state, env, pi, num_trajectory, truncate_size)
		print ('Episode {} reward = {}'.format(k, avr_reward))
		heat_map(length, f, env, 'heatmap/pi{}.pdf'.format(k))

def heat_map(length, f, env, filename):
	p_matrix = np.zeros([length, length], dtype = np.float32)
	for state in range(env.n_state):
		x,y,_,_ = env.state_decoding(state)
		#x,y = env.state_decoding(state)
		p_matrix[x,y] = f[state]
	p_matrix = p_matrix / np.sum(p_matrix)
	
	sns.heatmap(p_matrix, cmap="YlGnBu")#, vmin = 0.0, vmax = 0.07)
	ppPDF = PdfPages(filename)
	ppPDF.savefig()
	ppPDF.close()
	plt.clf()

def model_based(n_state, n_action, SASR, pi, gamma):
	T = np.zeros([n_state, n_action, n_state], dtype = np.float32)
	R = np.zeros([n_state, n_action], dtype = np.float32)
	R_count = np.zeros([n_state, n_action], dtype = np.int32)
	for sasr in SASR:
		for state, action, next_state, reward in sasr:
			T[state, action, next_state] += 1
			R[state, action] += reward
			R_count[state, action] += 1
	d0 = np.zeros([n_state, 1], dtype = np.float32)

	for state in SASR[:,0,0].flat:
		d0[state, 0] += 1.0
	t = np.where(R_count > 0)
	t0 = np.where(R_count == 0)
	R[t] = R[t]/R_count[t]
	R[t0] = np.mean(R[t])
	#print(R)
	T = T + 1e-9	# smoothing
	T = T/np.sum(T, axis = -1)[:,:,None]
	Tpi = np.zeros([n_state, n_state])
	for state in range(n_state):
		for next_state in range(n_state):
			for action in range(n_action):
				Tpi[state, next_state] += T[state, action, next_state] * pi[state, action]
	dt = d0/np.sum(d0)
	dpi = np.zeros([n_state, 1], dtype = np.float32)
	truncate_size = SASR.shape[1]
	discounted_t = 1.0
	self_normalizer = 0.0
	for i in range(truncate_size):
		dpi += dt * discounted_t
		if i < 50:
			dt = np.dot(Tpi.T,dt)
		self_normalizer += discounted_t
		discounted_t *= gamma
	dpi /= self_normalizer
	Rpi = np.sum(R * pi, axis = -1)
	return np.sum(dpi.reshape(-1) * Rpi)



def compute_Q_value(n_state, n_action, SASR, pi, gamma):
	T = np.zeros([n_state, n_action, n_state+1], dtype = np.float32)
	R = np.zeros([n_state, n_action], dtype = np.float32)
	R_count = np.zeros([n_state, n_action], dtype = np.int32)
	H = max(len(sasr) for sasr in SASR)
	for sasr in SASR:
		for state, action, next_state, reward in sasr:
			T[state, action, next_state] += 1
			R[state, action] += reward
			R_count[state, action] += 1
	d0 = np.zeros([n_state, 1], dtype = np.float32)
	for state in SASR[:,0,0].flat:
		d0[state, 0] += 1.0
	d0 /= np.sum(d0)
	T = T +  1e-9
	T = T / np.sum(T, axis=-1)[:, :, None]
	t = np.where(R_count > 0)
	t0 = np.where(R_count == 0)
	R[t] = R[t]/R_count[t]
	R[t0] = np.mean(R[t])
	#print(R)
	Q, V, G = compute_value(d0, T, R, gamma, range(n_state),range(n_action),H,pi)
	# convert G to average reward case
	G = G / H
	return Q,V,G

def run_experiment(n_state, n_action, SASR, pi0, pi1, gamma):
	
	den_discrete = Density_Ratio_discounted(n_state, gamma)
	x, w = train_density_ratio(SASR, pi0, pi1, den_discrete, gamma)
	x = x.reshape(-1)
	w = w.reshape(-1)

	Q_hat,_,G= compute_Q_value(n_state,n_action,SASR,pi1,gamma)
	negligible_states = get_Q_negligible_states(epsilon=epsilon_Q,states=range(n_state),actions=range(n_action),Q=Q_hat)
	print("S_A=",negligible_states)
	print("len(S_A)=", len(negligible_states))
	#print("Q_hat=",Q_hat)
	#print("G=",G)

	est_model_based2 = G[0]
	est_model_based = model_based(n_state, n_action, SASR, pi1, gamma)
	est_SDRE = off_policy_evaluation_density_ratio(SASR, pi0, pi1, w, gamma,[])
	est_SSDRE = off_policy_evaluation_density_ratio(SASR, pi0, pi1, w, gamma,negligible_states)
	est_naive_average = on_policy(SASR, gamma)
	est_IS = importance_sampling_estimator(SASR, pi0, pi1, gamma, [])
	est_SIS = importance_sampling_estimator(SASR, pi0, pi1, gamma, negligible_states)
	est_PDIS = importance_sampling_estimator_stepwise(SASR, pi0, pi1, gamma, [])
	est_SPDIS = importance_sampling_estimator_stepwise(SASR, pi0, pi1, gamma, negligible_states)
	est_INCRIS = INCRIS_estimator(SASR, pi0, pi1, gamma,[])
	est_SINCRIS = INCRIS_estimator(SASR, pi0, pi1, gamma,negligible_states)
	est_WIScheck = WIS_check(SASR, pi0, pi1, gamma)
	est_WIS = weighted_importance_sampling_estimator(SASR, pi0, pi1, gamma,[])
	est_WSIS = weighted_importance_sampling_estimator(SASR, pi0, pi1, gamma,negligible_states)
	est_WPDIS = weighted_importance_sampling_estimator_stepwise(SASR, pi0, pi1, gamma,[])
	est_WSPDIS = weighted_importance_sampling_estimator_stepwise(SASR, pi0, pi1, gamma,negligible_states)
	est_WINCRIS = weighted_INCRIS_estimator(SASR, pi0, pi1, gamma,[])
	est_SWINCRIS = weighted_INCRIS_estimator(SASR, pi0, pi1, gamma,negligible_states)

	#return est_model_based
	return (est_naive_average, est_SDRE, est_SSDRE, est_IS, est_SIS, est_PDIS, est_SPDIS, est_INCRIS, est_SINCRIS, est_WIS,
			est_WSIS, est_WIScheck,est_WPDIS, est_WSPDIS, est_WINCRIS, est_SWINCRIS,est_model_based, est_model_based2)


def run_DR_experiment(n_state, n_action, SASR, pi0, pi1, gamma):
	Q_hat, _, G = compute_Q_value(n_state, n_action, SASR, pi1, gamma)
	negligible_states = get_Q_negligible_states(epsilon=epsilon_Q, states=range(n_state), actions=range(n_action), Q=Q_hat)
	print("S_A=", negligible_states)
	print("len(S_A)=", len(negligible_states))
	# print("Q_hat=",Q_hat)
	# print("G=",G)
	est_DR = DR_estimator(SASR, pi0, pi1, gamma, range(n_state),range(n_action),[])
	est_SDR = DR_estimator(SASR, pi0, pi1, gamma, range(n_state),range(n_action),negligible_states)

	# return est_model_based
	return (est_DR,est_SDR)

def plot(sizes,scores,methods,eval_scores, resultsfolder,MC_iterations,tag):
	markers = {
		# "IS": "tab:blue","PDIS":"tab:orange","SIS (Lift states)":"tab:green","SIS (Covariance testing)":"tab:red",
		#     "SIS (Q-based)": "tab:purple","SIS": "tab:purple","INCRIS":"tab:brown",
		#      "DR": "tab:blue", "DRSIS (Lift states)": "tab:green", "DRSIS (Covariance testing)": "tab:red",
		#     "DRSIS (Q-based)": "tab:purple","DRSIS": "tab:purple",
		"WIS": "o",
		"WSIS": "^",
		"WPDIS": "D",
		"WSPDIS": "x",
		# "SIS (Lift states)": "tab:green",
		# "SIS (Covariance testing)": "tab:red", "SIS (Q-based)": "tab:purple",

		"WINCRIS": "+",
		"WSINCRIS": "8",
		"WDR": "P",
		# "DRSIS (Lift states)": "tab:green",
		# "DRSIS (Covariance testing)": "tab:red",
		# "DRSIS (Q-based)": "tab:purple",
		"WDRSIS": "v",
		# "SPDIS": "tab:green",

		# "SINCRIS": "tab:grey",

		"SDRE": ">", "SSDRE": "<",
	}
	colors = {
		# "IS": "tab:blue","PDIS":"tab:orange","SIS (Lift states)":"tab:green","SIS (Covariance testing)":"tab:red",
		#     "SIS (Q-based)": "tab:purple","SIS": "tab:purple","INCRIS":"tab:bron",
		#      "DR": "tab:blue", "DRSIS (Lift states)": "tab:green", "DRSIS (Covariance testing)": "tab:red",
		#     "DRSIS (Q-based)": "tab:purple","DRSIS": "tab:purple",
		"WIS": "tab:red",
		"WSIS": "tab:orange",
		"WPDIS": "tab:blue",
		"WSPDIS": "tab:purple",
		# "SIS (Lift states)": "tab:green",
		# "SIS (Covariance testing)": "tab:red", "SIS (Q-based)": "tab:purple",

		"WINCRIS": "k",
		"WSINCRIS": "tab:grey",
		"WDR": "tab:brown",
		# "DRSIS (Lift states)": "tab:green",
		# "DRSIS (Covariance testing)": "tab:red",
		# "DRSIS (Q-based)": "tab:purple",
		"WDRSIS": "tab:olive",
		# "SPDIS": "tab:green",

		# "SINCRIS": "tab:grey",

		"SDRE": "tab:cyan", "SSDRE": "tab:pink"
	}
	plt.figure(figsize=(5, 5))
	MSEs = {}
	score_l={}
	score_u = {}
	score_m = {}
	for i, method in enumerate(methods):
		print(method)
		score_l[method] = []
		score_u[method] = []
		score_m[method] = []
		MSEs[method] = []
		for idx,size in enumerate(sizes):
			sc = scores[method][idx]*size
			m = (np.mean(sc) - eval_scores[idx]*size)
			s = np.std(sc) / np.sqrt(len(sc))
			score_l[method].append(m - s)
			score_u[method].append(m + s)
			score_m[method].append(m)

			MSE = np.mean([(score - eval_scores[idx]) ** 2 for score in scores[method][idx]])
			MSEs[method].append(MSE)
	lines = []
	betweens = []
	for method in methods:
		line, = plt.plot(sizes, score_m[method], marker=markers[method], color=colors[method])
		print(score_l[method])
		b = plt.fill_between(sizes, score_l[method], score_u[method], color=colors[method],alpha=0.25)
		lines.append(line)
		betweens.append(b)
	#plt.yscale('symlog')
	plt.legend(lines, methods)

	plt.xlabel('Effective horizon (' +r"$H$" + ')')
	plt.ylabel("Residual (" + r"$\hat{G} - \mathcal{G}$" +")")
	plt.tight_layout()
	plt.savefig(resultsfolder + "variance_test_" + str(MC_iterations) + tag + ".pdf")

	plt.close()

	# table
	writefile = open(resultsfolder + "variance_test_" + str(MC_iterations) + tag + ".txt", "w")
	for method in methods:
		writefile.write(r" & " + method)
	writefile.write("\n \\textbf{Domain size}")
	for method in methods:
		writefile.write("& ")
	writefile.write("\n")
	for idx, size in enumerate(sizes):
		writefile.write("%d " % (size,))
		MSEList = [MSEs[method][idx] for method in methods]
		print_MSE_rows(MSEList, writefile)
		writefile.write("\n")
	writefile.close()

if __name__ == '__main__':

	length = 10
	env = taxi(length)
	n_state = env.n_state
	n_action = env.n_action
	
	num_trajectory = 100
	#truncate_size = 400 # 400
	gamma = 1.0

	parser = argparse.ArgumentParser(description='taxi environment')
	parser.add_argument('--f', dest = "folder", type=str, required=False, default="ResultsEpsilon"+str(epsilon_Q)+"/")
	parser.add_argument('--nt', type = int, required = False, default = num_trajectory)
	#parser.add_argument('--ts', type = int, required = False, default = truncate_size)
	parser.add_argument('--gm', type = float, required = False, default = gamma)
	args = parser.parse_args()

	behavior_ID = 4
	target_ID = 5

	pi_target = np.load('taxi-policy/pi19.npy')
	alpha = 0.0 # mixture ratio
	nt = args.nt # num_trajectory
	#ts = args.ts # truncate_size
	gm = args.gm # gamma
	pi_behavior = np.load('taxi-policy/pi18.npy')

	pi_behavior = alpha * pi_target + (1-alpha) * pi_behavior
	runs=20
	experiment_type = "plot"   # IS, DR, plot
	estimator_name_DR=['On Policy',
						  'WDR',
						  'WDRSIS']
	estimator_name_IS = ['On Policy',
						  'Naive Average',
						  'SDRE', 'SSDRE',
						  'IS', 'SIS',
						  'PDIS', 'SPDIS',
						  'INCRIS', 'SINCRIS',
						  'WIS', 'WSIS', 'WIS (check)',
						  'WPDIS', 'WSPDIS',
						  'WINCRIS', 'WSINCRIS',
						  'Model Based', 'Model-based 2']
	if experiment_type == "DR":
		estimator_name = estimator_name_DR
		args.folder+="DR"
	elif experiment_type == "IS":
		estimator_name = estimator_name_IS
	elif experiment_type == "plot":
		estimator_name = ['WIS', 'WSIS',
						  'WPDIS', 'WSPDIS',
						  'WINCRIS', 'WSINCRIS',
						  'SDRE','SSDRE',
						  'WDR','WDRSIS']
		#IS_indices = [estimator_name_IS.index(name) for name in estimator_name[:-2]]
		#DR_indices = [estimator_name_DR.index(name) for name in estimator_name[-2:]]
	sizes = [10,50,250,1000]
	if experiment_type == "plot":
		eval_scores=[]
		scores={method:[] for method in estimator_name}
		for ts in sizes:
			print("ts ", ts)
			res_IS = np.load(args.folder + 'nt={}ts={}gm={}.npy'.format(nt, ts, gm))
			res_DR = np.load(args.folder + 'DRnt={}ts={}gm={}.npy'.format(nt, ts, gm))
			eval_scores.append(np.mean(res_IS[0,:]))
			for method in estimator_name:
				if method in estimator_name_IS:
					i = estimator_name_IS.index(method)
					scores[method].append(res_IS[i])
				elif method in estimator_name_DR:
					i = estimator_name_DR.index(method)
					scores[method].append(res_DR[i])
				#print(scores[method][-1])
				print(method," MSE:",np.mean(np.square(scores[method][-1] - eval_scores[-1])))
		plot(sizes, scores, estimator_name, eval_scores, resultsfolder=args.folder+"plot_", MC_iterations=num_trajectory, tag="")
		sys.exit(0)
	estimators = len(estimator_name)
	for ts in sizes:

		res = np.zeros((estimators, runs), dtype = np.float32)
		writefile = open(args.folder + "MSE_" + str(ts) + ".txt", "w")
		for k in range(runs):
			np.random.seed(k)
			SASR0, _, _ = roll_out(n_state, env, pi_behavior, nt, ts)
			np.random.seed(k)
			SASR, _, _ = roll_out(n_state, env, pi_target, nt, ts)
			res[0, k] = on_policy(np.array(SASR), gm)
			if experiment_type == "IS":
				res[1:,k] = run_experiment(n_state, n_action, np.array(SASR0), pi_behavior, pi_target, gm)
			elif experiment_type == "DR":
				res[1:, k] = run_DR_experiment(n_state, n_action, np.array(SASR0), pi_behavior, pi_target, gm)
			print('------seed = {}------'.format(k))
			for i in range(estimators):
				print('  ESTIMATOR: '+estimator_name[i]+ ', rewards = {}'.format(res[i,k]))
			print('----------------------')
			sys.stdout.flush()
		for i in range(1,estimators):
			MSE = np.mean(np.square(res[i, :] - res[0,:]))
			writefile.write(" %s & %.8f \n" % (estimator_name[i], MSE))
		np.save(args.folder+'nt={}ts={}gm={}.npy'.format(nt,ts,gm), res)


	
