import sys, csv
import pandas, numpy, mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import argparse

'''Default function for visualising EEG Data without sampling capabilities and static data columns'''
def visualise_default(file_cmd):
    object_csv = pandas.read_csv(file_cmd)
    object_csv = object_csv[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']]
    object_csv = object_csv.dropna()
    eeg_data = object_csv[['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']]
    timestamps = object_csv[['timestamps']].to_numpy()

    columns_list = list(eeg_data.columns)  

    data_types = ["eeg"] * len(columns_list)
    data = eeg_data.transpose().to_numpy()
    sfreq = 256  # Hz
    info = mne.create_info(columns_list, sfreq, ch_types=data_types)

    raw = mne.io.RawArray(data, info)

    data_reshaped = numpy.reshape(raw.get_data(), (len(data[0]), len(data)))
    plt.plot(timestamps[:300], data_reshaped[:300])
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage')
    plt.title('Raw EEG Data')
    plt.show()

''' Visualise EEG Data with options for ICA, PSD, band-pass filtering and choosing channels to plot
    Possible options to pass to parameter column_type:
        ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']
        ['acc_1', 'acc_2', 'acc_3']
        ['alpha_session_score_1', 'alpha_session_score_2', 'alpha_session_score_3', 'alpha_session_score_4']
        ['blink']
        Or a mixture of all. 
'''
def visualise_filtered(file_cmd, column_type):
    object_csv = pandas.read_csv(file_cmd)

    column_names = object_csv.columns
    if not all(column in column_names for column in column_type):
        print("Error: Columns are not present in the CSV file.")
        return

    object_csv = object_csv[['timestamps'] + column_type]
    object_csv = object_csv.dropna()
    eeg_data = object_csv[column_type]
    timestamps = object_csv[['timestamps']].to_numpy()

    columns_list = list(eeg_data.columns)
    data_types = ["eeg"] * len(columns_list)
    data = eeg_data.transpose().to_numpy()
    sfreq = 256  # Hz
    
    info = mne.create_info(columns_list, sfreq, ch_types=data_types)
    raw = mne.io.RawArray(data, info)

    # # Apply Independent Component Analysis (ICA)
    # ica = ICA(n_components=len(columns_list), random_state=0)
    # ica.fit(raw)
    # ica.apply(raw)
    # data_reshaped = numpy.reshape(raw.get_data(), (len(data[0]), len(data)))
    # # Visualisation with ICA shown in Figure 2.5 and similar to Figure 4.7
    # plt.plot(timestamps[:500], data_reshaped[:500])
    # plt.xlabel('Time(s)')
    # plt.ylabel('Voltage')
    # plt.title('Raw EEG Data')
    # plt.show()

    # # Tranform data into frequency domain with PSD
    # # Apply a bandpass filter to the data (e.g., between 8 and 12 Hz)
    # raw.filter(8, 12)
    # psds, freqs = mne.time_frequency.psd_array_multitaper(raw.get_data(), sfreq, fmin=0, fmax=40)

    # # Plot the PSD
    # fig, ax = plt.subplots()
    # ax.semilogy(freqs, psds.T)
    # ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('Power Spectral Density (dB/Hz)')
    # ax.set_title('PSD of the Raw data')
    # plt.show()

    data_reshaped = numpy.reshape(raw.get_data(), (len(data[0]), len(data)))
    plot(timestamps, data_reshaped)
    

def plot(time, data, start=0, end=0):
    if start == 0 and end == 0:
        plt.plot(time[:500], data[:500])
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.show()
    else:
        plt.plot(time[start:end], data[start:end])
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="EEG file: muse2.csv")
    parser.add_argument("function", help="Function name: visualise_default, visualise_filtered)")
    parser.add_argument("parameters", nargs='*', help="Array of string parameters (e.g. ['eeg_1', 'eeg_2', 'eeg_3'])", default=None)
    args = parser.parse_args()

    if args.function == "visualise_default":
        visualise_default(args.file)
    elif args.function == "visualise_filtered":
        if args.parameters is None or args.parameters == []:
            print("visualise_filtered requires parameters")
        else:
            visualise_filtered(args.file, args.parameters)
    else:
        print(f"Unknown function: {args.function}")