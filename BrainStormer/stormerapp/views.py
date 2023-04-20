from django.shortcuts import render
import pandas, numpy, mne
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import os

def mainPage(request):
    return render(request, 'index.html')

def descriptionPage(request):
    return render(request, 'description.html')

def generatorPage(request):
    print(request.FILES)
    if request.method == 'POST' and 'customFile' in request.FILES and 'channels' in request.POST:
        file = request.FILES['customFile']
        # Return if columns are not present in the file
        if not isFileCsv(file):
            return render(request, 'generator.html')

        object_csv = pandas.read_csv(file)
        column_names = object_csv.columns
        column_types = request.POST.getlist('channels')
        if not all(column in column_names for column in column_types):
            print("Error: Some of the columns are not present in the CSV file.")
            return
        
        object_csv = object_csv[['timestamps'] + column_types]
        object_csv = object_csv.dropna()
        eeg_data = object_csv[column_types]
        timestamps = object_csv[['timestamps']].to_numpy()

        columns_list = list(eeg_data.columns)
        data_types = ["eeg"] * len(columns_list)
        data = eeg_data.transpose().to_numpy()
        sfreq = 256  # Hz
        info = mne.create_info(columns_list, sfreq, ch_types=data_types)
        raw = mne.io.RawArray(data, info)
        
        data1 = numpy.reshape(raw.get_data(), (len(data[0]), len(data)))
        plt.plot(timestamps, data1)
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.title('Raw EEG Data')
        plt.savefig('stormerapp/static/images/output.png')

        # If ICA does not work, reset to original plot
        try:
            plt.cla()
            ica = ICA(n_components=len(columns_list), random_state=0)
            ica.fit(raw.copy())
            ica.exclude = [0, 1]
            component = ica.apply(raw.copy())
            data2 = numpy.reshape(component.get_data(), (len(data[0]), len(data)))
            plt.plot(timestamps, data2)
            plt.xlabel('Time(s)')
            plt.ylabel('Voltage')
            plt.title('EEG Data with ICA')
            plt.savefig('stormerapp/static/images/output_ica.png')
        except Exception as e:
            data1 = numpy.reshape(raw.get_data(), (len(data[0]), len(data)))
            plt.plot(timestamps, data1)
            plt.xlabel('Time(s)')
            plt.ylabel('Voltage')
            plt.title('Raw EEG Data')
            plt.savefig('stormerapp/static/images/output_ica.png')


        # Set of 200 samples
        plt.cla()
        plt.plot(timestamps[0:200], data1[0:200])
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.title('Raw EEG Data')
        plt.savefig('stormerapp/static/images/output_filter1.png')

        # Set of 500 samples
        plt.cla()
        plt.plot(timestamps[0:500], data1[0:500])
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.title('Raw EEG Data')
        plt.savefig('stormerapp/static/images/output_filter2.png')

        # Set of 1000 samples
        plt.cla()
        plt.plot(timestamps[0:1000], data1[0:1000])
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.title('Raw EEG Data')
        plt.savefig('stormerapp/static/images/output_filter3.png')

        # Set of custom range
        if 'rangeData' in request.POST:
            rangeData = request.POST.get(rangeData)
            plt.cla()
            plt.plot(timestamps[0:rangeData], data1[0:rangeData])
            plt.savefig('stormerapp/static/images/output.png')
        
        return render(request, 'upload.html', {'csvFile': file})

    # Return generator page if no valid file is provided
    return render(request, 'generator.html')

# helper functions
def isFileCsv(file_path):
    # Check if the file has a .csv extension
    file_ext = os.path.splitext(file_path.name)[-1].lower()
    if file_ext != '.csv':
        return False
    else:
        return True

# Not used domain transformer
def transform_into_frequency(data):
    value = mne.filter.filter_data(data, 256, 8, 12)
    plt.plot(value)


