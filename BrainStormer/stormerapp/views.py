from django.shortcuts import render
import sys, csv
import pandas, numpy, mne
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import plotly.graph_objects as gr
from plotly.subplots import make_subplots
from mne.preprocessing import ICA

import os

# Create your views here.
def mainPage(request):
    return render(request, 'index.html')

# Add check for data file
def generatorPage(request):
    print(request.FILES)
    if request.method == 'POST' and 'customFile' in request.FILES:
        file = request.FILES['customFile']
        # make error message
        if not isFileCsv(file):
            return render(request, 'generator.html')
        # reader = csv.DictReader(file)
        # for row in reader:
        #     pass
        object = pandas.read_csv(file)
        object = object[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']]
        # object = object[['timestamps', 'acc_1', 'acc_2', 'acc_3']]
        # object = object[['timestamps', 'alpha_session_score_1', 'alpha_session_score_2', 'alpha_session_score_3', 'alpha_session_score_4']]
        # object = object[['timestamps', 'blink']]
        object = object.dropna()
        newfile = object[['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']]
        # newfile = object[['acc_1', 'acc_2', 'acc_3']]
        # newfile = object[['alpha_session_score_1', 'alpha_session_score_2', 'alpha_session_score_3', 'alpha_session_score_4']]
        # newfile = object[['blink']]
        print()
        timestamps = object[['timestamps']].to_numpy()
        # timestamps = newfile[:1].to_numpy()
        # with open('names.csv', newline='') as csvfile:
            
        #     
        # file1 = csv.writer(file)
        # print(file)
        # 
        columns_list = list(newfile.columns)
        
        # timestamps = pandas.read_csv(file, usecols=lambda x: x.upper() in ["TIMESTAMPS"])
        
        data_types = []
        for i in range(0, len(columns_list)):
            data_types.append("eeg")
        data = newfile.transpose().to_numpy()
        sfreq = 256  # Hz
        info = mne.create_info(columns_list, sfreq, ch_types=data_types)
        raw = mne.io.RawArray(data, info)
        ica = ICA(n_components=len(columns_list), random_state=0)
        ica.fit(raw)
        ica.exclude = [0, 1, 2]  # select which ICs to remove
        ica.apply(raw)

        time = timestamps - timestamps[0]
        data1 = numpy.reshape(data, (len(data[0]), len(data)))
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(range(10))
        # fig.savefig('temp.png')
        plt.plot(time, data1)
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.title('Raw EEG Data')
        # plt.show()
        plt.savefig('stormerapp/static/images/output.png')



        plt.cla()
        plt.plot(time[0:1000], data1[0:1000])
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.title('Raw EEG Data')
        plt.savefig('stormerapp/static/images/output_filter1.png')
        plt.cla()
        plt.plot(time[0:5000], data1[0:5000])
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.title('Raw EEG Data')
        plt.savefig('stormerapp/static/images/output_filter2.png')
        plt.cla()
        plt.plot(time[0:10000], data1[0:10000])
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.title('Raw EEG Data')
        plt.savefig('stormerapp/static/images/output_filter3.png')

        if 'rangeData' in request.POST:
            rangeData = request.POST.get(rangeData)
            plt.cla()
            plt.plot(time[0:rangeData], data1[0:rangeData])
            plt.savefig('stormerapp/static/images/output.png')

        
        
        return render(request, 'upload.html', {'csvFile': file})
    return render(request, 'generator.html')

# helper functions
def isFileCsv(filePath):
    # Check if the file has a .csv extension
    fileExt = os.path.splitext(filePath.name)[-1].lower()
    if fileExt != '.csv':
        return False
    else:
        return True

def filterIntoFrequency(data):
    value = mne.filter.filter_data(data, 256, 8, 12)
    plt.plot(value)


