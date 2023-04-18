# Code inspired by MNE's documentation and adapted to fit our project's experiment
import mne
import os
import matplotlib.pyplot as plt
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns



def mine():
    # Define parameters 
    subject = 1  # use data from subject 1
    runs = [6, 10, 14]  # use only hand and feet motor imagery runs
    k = 5

    # Get EEG data
    files = eegbci.load_data(subject, runs, '../MNE_datasets/')
    # Read raw data files and combine all runs
    raws = [read_raw_edf(f, preload=True) for f in files]
    raw_obj = concatenate_raws(raws)

    raw_data = raw_obj._data

    # Extract events from raw data
    events, event_ids = mne.events_from_annotations(raw_obj, event_id='auto')

    tmin, tmax = -1, 4  # define epochs around events (in s)
    event_ids = dict(hands=2, feet=3)  # map event IDs to tasks

    epochs = mne.Epochs(raw_obj, events, event_ids, tmin, tmax, baseline=None, preload=True)

    # # Machine Learning SVM Classification

    # # Get the data and labels
    # X = epochs.get_data()
    # y = epochs.events[:, 2]  # labels (2: left hand, 3: right hand)

    # # Flatten the data to 2D (for SVM)
    # X_flat = np.reshape(X, (X.shape[0], -1))  # shape: (n_epochs, n_channels * n_times)

    # # Train SVM
    # clf = SVC(kernel='linear')
    # scores = cross_val_score(clf, X_flat, y, cv=7)  # 7-fold cross validation
    # mean = scores.mean() * 100
    # # Accuracy metric shown in Table 4.1
    # print(f"Accuracy: {mean:.2f}")

    # clf.fit(X_flat, y)  # fit the classifier to the full data
    # coef = clf.coef_  # the weights of the SVM
    # coef_img = np.reshape(coef, (X.shape[1], X.shape[2]))  # reshape the weights back to 2D
    # evoked = mne.EvokedArray(coef_img, epochs.info, tmin=tmin)
    # fig, ax = plt.subplots()
    # image = evoked.plot_image()
    
    # # Machine Learning Classification shown in Figure 2.6
    # plt.show()

    # Machine Learning KNN Classification 

    # # Get the data and labels
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
    # mean = cv_scores.mean() * 100
    # # Accuracy metric shown in Table 4.1
    # print(f"Accuracy: {mean:.2f}")

    # # fit KNN to training data and predict on test data
    # y_pred = knn.predict(X_test.reshape(n_samples - train_size, -1))
    # cm = confusion_matrix(y_test, y_pred)

    # # Plot confusion matrix
    # sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.show()

    #Access to the data
    data = epochs._data

    # Plot of raw EEG data shown in Figure 2.12
    plt.plot(data[16:2000,0,:].T)
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage')
    plt.show()

    # # Plot of single electrode EEG data shown in Figure 2.11
    # plt.plot(raw_data[0,:4999])
    # plt.title("Raw EEG, electrode 0, samples 0-4999")
    # plt.xlabel('Time(s)')
    # plt.ylabel('Voltage')
    # plt.show()
   
if __name__ == "__main__":
    mine()