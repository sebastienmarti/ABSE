# Load libraries -------------------------------------------------------------
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
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.joblib import Parallel, delayed
from fonctionsPython.sm_fieldtrip2mne import sm_fieldtrip2mne
from fonctionsPython.jr_subscore import subscore

#------------------------- for single task -------------------------------------
data_path = '/neurospin/meg/meg_tmp/ABSE_Marti_2014/mat/epoch/'
subject = 'am150105'
fname = op.join(data_path, 'abse_' + subject + '_train.mat')
decim = 10
n_jobs = 7

# import data
epochs = sm_fieldtrip2mne(fname)
n_trial = len(epochs)

# additional preproc steps
epochs.apply_baseline((0,.05))

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

# # plot ERFs 
# evoked_face = epochs[face].average()
# evoked_face.plot(show=False)
# evoked_house = epochs[house].average()
# evoked_house.plot(show=False)

# decoding
epochs = epochs.decimate(decim)
gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=n_jobs)
gat.fit(epochs, stim_list)
# y_pred = gat.predict(epochs)
# score = gat.score()



#######################################################################################################
 # now for the rsvp data 
fname = op.join(data_path, 'abse_' + subject + '_main.mat')

epochs = sm_fieldtrip2mne(fname)
n_trial = len(epochs)

# additional preproc steps
epochs.apply_baseline((-.5,0))
# epochs._data = flt.low_pass_filter(epochs.get_data(), 1000, 20)

# # check ERFs
# evoked = epochs.average()
# evoked.plot(show=False)

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
# offset = [-1, 0, 1]

# decoding
epochs = epochs.decimate(decim)
gat.predict_mode = 'mean-prediction' # generalization across conditions so no cross validation here.

# compute scores for each stimulus
rsvp_score_all = []
rsvp_score_lag = []
rsvp_score_offset0_lag = []
rsvp_score_offset_1_lag = []
rsvp_score_offset1_lag = []
for istim in range(n_stim):
	# compute scores across all trials
	s = gat.score(epochs, stim_list_rsvp[:,istim])
	s = np.array(s)
	rsvp_score_all.append(s)
	# compute scores for lags
	for ilag in range(len(ulag)):
		sel = lag==ulag[ilag]
		s = subscore(gat, sel=sel, y=stim_list_rsvp[sel,istim])
		s = np.array(s)
		rsvp_score_lag.append(s)

		# compute scores depending on accuracy
		# offset = 0 (correct report)
		sel = [all(tup) for tup in zip(lag==ulag[ilag], R1==1, R2[:,1]==ulag[ilag])] # select one lag, correct T1 response, T2 resp with an offset
		s = subscore(gat, sel=sel, y=stim_list_rsvp[sel,istim])
		s = np.array(s)
		rsvp_score_offset0_lag.append(s)

		# compute scores depending on accuracy
		# offset = 0 (correct report)
		sel = [all(tup) for tup in zip(lag==ulag[ilag], R1==1, R2[:,1]==ulag[ilag]-1)] # select one lag, correct T1 response, T2 resp with an offset
		s = subscore(gat, sel=sel, y=stim_list_rsvp[sel,istim])
		s = np.array(s)
		rsvp_score_offset_1_lag.append(s)

		# compute scores depending on accuracy
		# offset = 0 (correct report)
		sel = [all(tup) for tup in zip(lag==ulag[ilag], R1==1, R2[:,1]==ulag[ilag]+1)] # select one lag, correct T1 response, T2 resp with an offset
		s = subscore(gat, sel=sel, y=stim_list_rsvp[sel,istim])
		s = np.array(s)
		rsvp_score_offset1_lag.append(s)
		print('.')

	print(istim)

# gat.plot(vmin = .2, vmax = .3, title="Generalization Across Time", show=False)
# gat.plot_diagonal(show=False)
# plt.show()

# compute subscores


