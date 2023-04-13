# Code inspired by MNE's documentation and adapted to fit our project
import mne
import os

import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
# from mne.datasets import eegbci
# from mne.io import concatenate_raws, read_raw_edf

# # For elimiating warnings
# from warnings import simplefilter

# import mne.viz


import numpy as np

from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns



def mine():
    #Define the parameters 
    subject = 1  # use data from subject 1
    runs = [6, 10, 14]  # use only hand and feet motor imagery runs
    k = 5

    #Get data and locate in to given path
    files = eegbci.load_data(subject, runs, '../Individual Project/datasets/')
    #Read raw data files where each file contains a run
    raws = [read_raw_edf(f, preload=True) for f in files]
    #Combine all loaded runs
    raw_obj = concatenate_raws(raws)

    raw_data = raw_obj._data

    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
    raw = mne.io.read_raw_fif(sample_data_raw_file)
    raw.crop(tmax=60).load_data()
    # print(raw_data)
    # plt.plot(raw_data)
    # plt.xlabel('Time(s)')
    # plt.ylabel('Voltage')
    # plt.show()

    print("Number of channels: ", str(len(raw_data)))
    print("Number of samples: ", str(len(raw_data[0])))

    #Extract events from raw data
    events, event_ids = mne.events_from_annotations(raw_obj, event_id='auto')

    tmin, tmax = -1, 4  # define epochs around events (in s)
    event_ids = dict(hands=2, feet=3)  # map event IDs to tasks

    epochs = mne.Epochs(raw_obj, events, event_ids, tmin, tmax, baseline=None, preload=True)

    # # Get the data and labels
    # X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    # y = epochs.events[:, 2]  # labels (2: left hand, 3: right hand)

    # # Flatten the data to 2D (for SVM)
    # X_flat = np.reshape(X, (X.shape[0], -1))  # shape: (n_epochs, n_channels * n_times)

    # # Train SVM
    # clf = SVC(kernel='linear')
    # scores = cross_val_score(clf, X_flat, y, cv=7)  # 7-fold cross validation
    # mean = scores.mean() * 100
    # print(f"Accuracy: {mean:.2f}")

    # clf.fit(X_flat, y)  # fit the classifier to the full data
    # coef = clf.coef_  # the weights of the SVM
    # coef_img = np.reshape(coef, (X.shape[1], X.shape[2]))  # reshape the weights back to 2D
    # evoked = mne.EvokedArray(coef_img, epochs.info, tmin=tmin)
    # fig, ax = plt.subplots()
    # im = evoked.plot_image()
    
    # ax.set_title('SVM classification weights')
    # plt.show()

    # X = epochs.get_data()
    # y = epochs.events[:, -1]
    # n_samples, n_features, _ = X.shape
    # train_size = int(n_samples * 0.8)
    # X_train, X_test = X[:train_size], X[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # # Define the KNN classifier
    # knn = KNeighborsClassifier(n_neighbors=k)

    # # Train the KNN classifier on the training set
    # knn.fit(X_train.reshape(train_size, -1), y_train)

    # # Evaluate the KNN classifier using 5-fold cross-validation
    # cv_scores = cross_val_score(knn, X.reshape(n_samples, -1), y, cv=5)
    # mean = scores.mean() * 100
    # print('Accuracy: {mean:.2f}%'.format(cv_scores.mean()*100))

    # # fit KNN to training data and predict on test data
    # y_pred = knn.predict(X_test.reshape(n_samples - train_size, -1))
    # cm = confusion_matrix(y_test, y_pred)

    # # Plot confusion matrix
    # sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.show()

    # root = mne.datasets.sample.data_path() / 'MEG' / 'sample'
    # evk_file = root / 'sample_audvis-ave.fif'
    # evokeds_list = mne.read_evokeds(evk_file, baseline=(None, 0), proj=True,
    #                                 verbose=False)

    # # Show condition names and baseline intervals
    # for e in evokeds_list:
    #     print(f'Condition: {e.comment}, baseline: {e.baseline}')

    # conds = ('aud/left', 'aud/right', 'vis/left', 'vis/right')
    # evks = dict(zip(conds, evokeds_list))
    # #      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is equivalent to:
    # # {'aud/left': evokeds_list[0], 'aud/right': evokeds_list[1],
    # #  'vis/left': evokeds_list[2], 'vis/right': evokeds_list[3]}

    # subjects_dir = root.parent.parent / 'subjects'
    # trans_file = root / 'sample_audvis_raw-trans.fif'
    
    # for ch_type in ('mag', 'grad', 'eeg'):
    #     evk = evks['aud/right'].copy().pick(ch_type)
    #     _map = mne.make_field_map(evk, trans=str(trans_file), subject='sample',
    #                             subjects_dir=subjects_dir, meg_surf='head')
    #     fig = evk.plot_field(_map, time=0.1)
    #     mne.viz.set_3d_title(fig, ch_type, size=20)

    # #Access to the data
    data = epochs._data

    # Get FFT

    # n_events = len(data) # or len(epochs.events)
    # print("Number of events: " + str(n_events)) 

    # n_channels = len(data[0,:]) # or len(epochs.ch_names)
    # print("Number of channels: " + str(n_channels))

    # n_times = len(data[0,0,:]) # or len(epochs.times)
    # print("Number of time instances: " + str(n_times))
    # spectrum = raw.compute_psd()
    # spectrum.plot(average=True)
    # spectrum.pick('eeg').plot_topo()
    plt.plot(data[16:2000,0,:].T)
    # print(data[14:20,0,:])
    plt.title("Exemplar single-trial epoched data, for electrode 0")
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage')
    raw.plot_sensors(ch_type='eeg')
    plt.show()

    # plt.plot(raw_data[0,:4999])
    # plt.title("Raw EEG, electrode 0, samples 0-4999")
    # plt.xlabel('Time(s)')
    # plt.ylabel('Voltage')
    # plt.show()
    # #Define the parameters 
    # subject = 1  # use data from subject 1
    # runs = [6, 10, 14]  # use only hand and feet motor imagery runs

    # #Get data and locate in to given path
    # files = eegbci.load_data(subject, runs, '../Individual Project/datasets/')
    # #Read raw data files where each file contains a run
    # raws = [read_raw_edf(f, preload=True) for f in files]
    # #Combine all loaded runs
    # raw_obj = concatenate_raws(raws)




    # # ignore all future warnings
    # simplefilter(action='ignore', category=FutureWarning)
    # #Load epoched data
    # data_file = '../study1/study1_eeg/epochdata/P-02'

    # # Read the EEG epochs:
    # epochs = mne.read_epochs(data_file + '.fif', verbose='error')



    # epochs = epochs['FP', 'FN', 'FU']
    # print('Percentage of Pleasant familiar events : ', np.around(len(epochs['FP'])/len(epochs), decimals=2))
    # print('Percentage of Neutral familiar events : ',np.around(len(epochs['FN'])/len(epochs), decimals=2))
    # print('Percentage of Unpleasant familiar : ', np.around(len(epochs['FU'])/len(epochs), decimals=2))
    # epochs.plot(scalings='auto')




if __name__ == "__main__":
    mine()