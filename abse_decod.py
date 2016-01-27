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
from fonctionsPython import sm_fieldtrip2mne

#------------------------- for single task -------------------------------------
data_path = '/neurospin/meg/meg_tmp/ABSE_Marti_2014/mat/epoch/'
subject = 'am150105'
fname = op.join(data_path, 'abse_' + subject + '_train.mat')
decim = 10

# # Convert from Fieldtrip structure to mne Epochs
# mat = sio.loadmat(fname, squeeze_me=True, struct_as_record=False) # Load data 
# ft_data = mat['data']
# n_trial = len(ft_data.trial)
# n_chans, n_time = ft_data.trial[0].shape
# data = np.zeros((n_trial, n_chans, n_time))
# for trial in range(n_trial):
#     data[trial, :, :] = ft_data.trial[trial]
# sfreq = float(ft_data.fsample)  # sampling frequency
# coi = range(306)  # channels of interest:
# data = data[:, coi, :]
# chan_names = [l.encode('ascii') for l in ft_data.label[coi]]
# chan_types = ft_data.label[coi]
# chan_types[0:305:3] = 'grad'
# chan_types[1:305:3] = 'grad'
# chan_types[2:306:3] = 'mag'
# info = create_info(chan_names, sfreq, chan_types)
# events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
#                np.zeros(n_trial), np.zeros(n_trial)]
# epochs = EpochsArray(data, info, events=np.array(events, int),
#                      tmin=ft_data.time[0][0], verbose=False)
# epochs.times = ft_data.time[0]

epochs = sm_fieldtrip2mne.sm_fieldtrip2mne(fname)

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
gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=6)
gat.fit(epochs, stim_list)
y_pred = gat.predict(epochs)
score = gat.score()
# gat.plot(vmin = .1, vmax = .4, title="Generalization Across Time", show=False)
# gat.plot_diagonal()

#----------------------- now for the rsvp data -------------------------
fname = op.join(data_path, 'abse_' + subject + '_main.mat')
# Convert from Fieldtrip structure to mne Epochs
# mat = sio.loadmat(fname, squeeze_me=True, struct_as_record=False)
# ft_data = mat['data']
# n_trial = len(ft_data.trial)
# n_chans, n_time = ft_data.trial[0].shape
# data = np.zeros((n_trial, n_chans, n_time))
# for trial in range(n_trial):
#     data[trial, :, :] = ft_data.trial[trial]
# sfreq = float(ft_data.fsample)  # sampling frequency
# coi = range(306)  # channels of interest:
# data = data[:, coi, :]
# chan_names = [l.encode('ascii') for l in ft_data.label[coi]]
# chan_types = ft_data.label[coi]
# chan_types[0:305:3] = 'grad'
# chan_types[1:305:3] = 'grad'
# chan_types[2:306:3] = 'mag'
# info = create_info(chan_names, sfreq, chan_types)
# events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
#                np.zeros(n_trial), np.zeros(n_trial)]
# epochs = EpochsArray(data, info, events=np.array(events, int),
#                      tmin=ft_data.time[0][0], verbose=False)
# epochs.times = ft_data.time[0]

epochs = sm_fieldtrip2mne.sm_fieldtrip2mne(fname)

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

# decoding
epochs = epochs.decimate(decim)
gat.predict_mode = 'mean-prediction' # generalization across conditions so no cross validation here.
# y_pred = gat.predict(epochs)
# loop across rsvp stimuli
# loop version
rsvp_score = []
for istim in range(n_stim):
	s = gat.score(epochs, stim_list_rsvp[:,istim])
	s = np.array(s)
	rsvp_score.append(s)
	print(istim)

# joblib version
# out = Parallel(n_jobs=6)(delayed(gat.score)(epochs, stim_list_rsvp[:,istim]) for istim in range(n_stim))

# gat.plot(vmin = .2, vmax = .3, title="Generalization Across Time", show=False)
# gat.plot_diagonal(show=False)
# plt.show()

#  compute subscores.


