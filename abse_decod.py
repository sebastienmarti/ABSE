""" Load libraries """
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from mne.io.meas_info import create_info
from mne.epochs import EpochsArray
from mne.decoding import GeneralizationAcrossTime
from mne import filter as flt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.joblib import Parallel, delayed
from fonctionsPython.sm_fieldtrip2mne import sm_fieldtrip2mne

""" Some variables """
subjects = ['am150105', 'bb130599', 'cd130323', 'jf140150', 'ql100269', 
	'sb150103', 'sl130503','ws140212', 'jl150086', 'fm100109', 
	'hb140194', 'sd150012', 'mk140057', 'xm140202', 'lr110094']

data_path = '/neurospin/meg/meg_tmp/ABSE_Marti_2014/mat/epoch/'
save_path = '/neurospin/meg/meg_tmp/ABSE_Marti_2014/mne/decod/'
minNumTrials = 2 # min number of trials to compute scores in a condition
decim = 10
lowpass_filter = 30
# classification parameters
n_jobs = -2
n_fold = 5
# step = .01
# length = .01

""" Main loop across subjects """
for isub in range(len(subjects)):
	# isub=1
	#------------------------- for single task -------------------------------------
	subject = subjects[isub]
	fname = op.join(data_path, 'abse_' + subject + '_train.mat')

	# import data
	epochs = sm_fieldtrip2mne(fname)
	n_trial = len(epochs)

	# additional preproc steps
	epochs.apply_baseline((0,.05))
	epochs._data = flt.low_pass_filter(epochs.get_data(), 1000, lowpass_filter)

	# import behavior and conditions
	mat = sio.loadmat(fname)
	behavior = mat['data']['behavior']
	stim = np.squeeze(behavior[0][0][0][0][0][0])
	response = np.squeeze(behavior[0][0][0][0][1][0])
	face = stim == 'f'
	obj = stim == 'o'
	body = stim == 'b'
	house = stim == 'h'
	square = stim == 's'
	stim_list = np.zeros((n_trial,)) # note that shape should not be ntrial X 1 like in matlab but instead (ntrial,)
	for itrl in range(n_trial):
		if face[itrl]:
			stim_list[itrl] = 1
		elif house[itrl]:
			stim_list[itrl] = 2
		elif obj[itrl]:
			stim_list[itrl] = 3
		elif body[itrl]:
			stim_list[itrl] = 4
		elif square[itrl]:
			stim_list[itrl] = 0

	# decoding
	epochs = epochs.decimate(decim)
	n_tr = len(epochs.times)
	X = epochs[stim_list>0]
	y = stim_list[stim_list>0]

	# SVM parameters
	scaler = StandardScaler() # centers data by removing the mean and scales to unit variance
	# model = svm.SVC(C=1, kernel='linear', class_weight='auto')
	model = svm.LinearSVC(C=1, multi_class='ovr', class_weight='auto')
	clf = make_pipeline(scaler, model)
	cv = StratifiedKFold(y, n_fold)
	gat = GeneralizationAcrossTime(
		cv=cv,
		clf=clf, 
		predict_mode='cross-validation', 
		n_jobs=n_jobs,
		# train_times=dict(step=step,	length=length)
		)

	gat.fit(X, y)
	# score = gat.score(X, y)
	# gat.plot(vmin=.2, vmax=.3)
	# gat.plot_diagonal(chance=.25)

	#######################################################################################################
	 # now for the rsvp data 
	fname = op.join(data_path, 'abse_' + subject + '_main.mat')

	epochs = sm_fieldtrip2mne(fname)
	n_trial = len(epochs)


	# additional preproc steps
	epochs.apply_baseline((-.5,0))
	epochs._data = flt.low_pass_filter(epochs.get_data(), 1000, lowpass_filter)

	# rsvp conditions and behavior
	fname = op.join(data_path, 'abse_' + subject + '_main_behavior.mat')
	mat = sio.loadmat(fname)
	n_stim = 12
	stim_list_rsvp = np.zeros((n_trial,n_stim))
	for itrl in range(n_trial):
		for istim in range(n_stim):
			rsvpfile = mat['output']['stimuli'][0][0][0][0][0][0][0][0][0][itrl][0][istim][0]
			if 'face' in rsvpfile:
				stim_list_rsvp[itrl,istim] = 1
			elif 'house' in rsvpfile:
				stim_list_rsvp[itrl,istim] = 2
			elif 'object' in rsvpfile:
				stim_list_rsvp[itrl,istim] = 3
			elif 'body' in rsvpfile:
				stim_list_rsvp[itrl,istim] = 4

	# get lag indices
	lag = mat['output']['target'][0][0][0][0][0][0]
	R1 = mat['output']['response'][0][0][0][0][3][0][0][0][0]
	R2 = mat['output']['response'][0][0][0][0][4][0][0][0]
	ulag = np.unique(lag)
	n_lag = len(ulag)
	# offset = [-1, 0, 1]

	# decoding
	epochs = epochs.decimate(decim)
	n_te = len(epochs.times)
	gat.predict_mode = 'mean-prediction' # generalization across conditions so no cross validation here.

	# compute scores for each stimulus
	rsvp_score_all = np.zeros((n_tr, n_te, n_stim))
	rsvp_score_lag = np.zeros((n_tr, n_te, n_stim, n_lag))
	rsvp_score_offset0_lag = np.zeros((n_tr, n_te, n_stim, n_lag))
	rsvp_score_offset_1_lag = np.zeros((n_tr, n_te, n_stim, n_lag))
	rsvp_score_offset1_lag = np.zeros((n_tr, n_te, n_stim, n_lag))
	for istim in range(n_stim):
		# compute scores across all trials
		s = gat.score(epochs, stim_list_rsvp[:,istim]) # compute predictions as well.
		s = np.array(s)
		rsvp_score_all[:,:,istim] = s

		# compute scores for lags
		for ilag in range(len(ulag)):
			sel = lag==ulag[ilag] # select one lag
			s = gat.score(epochs[sel], stim_list_rsvp[sel,istim])
			s = np.array(s)
			rsvp_score_lag[:,:,istim,ilag] = s

			if istim+1==ulag[ilag]: # if the stim is one of the lag, then scores according to report 
				# compute scores depending on accuracy
				# one lag, correct T1, correct T2
				# sel = [all(tup) for tup in zip(lag==ulag[ilag], R2[:,0]==ulag[ilag])]
				sel = lag==ulag[ilag]
				sel2 = R2[:,0]==ulag[ilag]
				sel = sel & sel2
				if sum(np.array(sel).astype(int)) > minNumTrials: 
					s = gat.score(epochs[sel], stim_list_rsvp[sel,istim])
					s = np.array(s)
					rsvp_score_offset0_lag[:,:,istim,ilag] = s
				else:
					rsvp_score_offset0_lag[:,:,istim,ilag] = None

				# one lag, correct T1, T2 with offset -1
				sel = lag==ulag[ilag]
				sel2 = R2[:,0]==ulag[ilag]-1
				sel = sel & sel2
				if sum(np.array(sel).astype(int)) > minNumTrials: 
					s = gat.score(epochs[sel], stim_list_rsvp[sel,istim])
					s = np.array(s)
					rsvp_score_offset_1_lag[:,:,istim,ilag] = s
				else:
					rsvp_score_offset_1_lag[:,:,istim,ilag] = None

				# one lag, correct T1, T2 with offset 1
				sel = lag==ulag[ilag]
				sel2 = R2[:,0]==ulag[ilag]+1
				sel = sel & sel2
				if sum(np.array(sel).astype(int)) > minNumTrials: 
					s = gat.score(epochs[sel], stim_list_rsvp[sel,istim])
					s = np.array(s)
					rsvp_score_offset1_lag[:,:,istim,ilag] = s
				else:
					rsvp_score_offset1_lag[:,:,istim,ilag] = None
			else:
				rsvp_score_offset0_lag[:,:,istim,ilag] = None
				rsvp_score_offset_1_lag[:,:,istim,ilag] = None
				rsvp_score_offset1_lag[:,:,istim,ilag] = None
			print('.')
		print(istim)

	# Save subject results
	fsave = op.join(save_path, subject + '_decod_main')
	np.save(fsave, rsvp_score_all)
	fsave = op.join(save_path, subject + '_decod_main_lag')
	np.save(fsave, rsvp_score_lag)
	fsave = op.join(save_path, subject + '_decod_main_offset0')
	np.save(fsave, rsvp_score_offset0_lag)
	fsave = op.join(save_path, subject + '_decod_main_offset_1')
	np.save(fsave, rsvp_score_offset_1_lag)
	fsave = op.join(save_path, subject + '_decod_main_offset1')
	np.save(fsave, rsvp_score_offset1_lag)






