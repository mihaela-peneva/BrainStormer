import sys, csv
import pandas, numpy, mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

def activate():
    file = sys.argv[1]
    data = []
    object = pandas.read_csv(file)
    print(f'Length is {len(object)}')
    # drops everything because they all have nans
    print(object)
    # objectNoTimestamps = object.drop(columns=['timestamps', 'config', 'recorder_info', 'device'])
    # print(f'Object 2 is {object2}')
    object = object[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']]
    # object = object[['timestamps', 'acc_1', 'acc_2', 'acc_3']]
    # object = object[['timestamps', 'alpha_session_score_1', 'alpha_session_score_2', 'alpha_session_score_3', 'alpha_session_score_4']]
    # object = object[['timestamps', 'blink']]
    # object = object.dropna()
    # eeg_data = object[['eeg_1']]
    eeg_data = object[['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']]
    # eeg_data = object[['acc_1', 'acc_2', 'acc_3']]
    # eeg_data = object[['alpha_session_score_1', 'alpha_session_score_2', 'alpha_session_score_3', 'alpha_session_score_4']]
    # eeg_data = object[['blink']]
    print()
    timestamps = object[['timestamps']].to_numpy()
    # print(f'File is {eeg_data}')
    # print(f'File is {timestamps}')
    # timestamps = eeg_data[:1].to_numpy()

    # columns_list = list(objectNoTimestamps.columns)
    columns_list = list(eeg_data.columns)
    
    # timestamps = pandas.read_csv(file, usecols=lambda x: x.upper() in ["TIMESTAMPS"])
    
    data_types = []
    for i in range(0, len(columns_list)):
        data_types.append("eeg")
    # data = objectNoTimestamps.transpose().to_numpy()
    data = eeg_data.transpose().to_numpy()
    sfreq = 256  # Hz
    info = mne.create_info(columns_list, sfreq, ch_types=data_types)
    raw = mne.io.RawArray(data, info)
    # ica = ICA(n_components=len(columns_list), random_state=0)
    # ica.fit(raw)
    # ica.apply(raw)

    # remove entries with nan values

    time = timestamps - timestamps[0]
    data1 = numpy.reshape(data, (len(data[0]), len(data)))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(range(10))
    # fig.savefig('temp.png')
    # self.filterIntoFrequency(data1)
    value = mne.filter.filter_data(data1, 256, 8, 12)
    plt.plot(value)
    plt.show()
    # plt.plot(time[0:500], data1[0:500])
    # plt.plot(data1[0:500])
    # plt.xlabel('Time(s)')
    # plt.ylabel('Voltage')
    # plt.title('All channels without additional timestamps configuration')
    plt.show()

def main(start=0, end=0):
    object = sys.argv[1]
    data = []

    # select which columns to rad instead of specifying header as 2 columns
    # Wwhy does this work?
    eeg_data = pandas.read_csv(object, usecols=lambda x: x.upper() in ["EEG_1", "EEG_2", "EEG_3", "EEG_4", "EEG_5"])
    
    # eeg_data = pandas.read_csv(object, usecols=lambda x: x.upper() in ["ACC_1", "ACC_2", "ACC_3" "GYRO_1", "GYRO_2", "GYRO_3"])
    timestamps = pandas.read_csv(object, usecols=lambda x: x.upper() in ["TIMESTAMPS"]).to_numpy()
    columns_list = list(eeg_data.columns)
    data_types = ["misc"]
    for i in range(1, len(columns_list)):
        data_types.append("eeg")
    print(f'Length is {len(columns_list)}')
    # columns_list = ['eeg']
    # print(columns_list)

    # Read the CSV file as a NumPy array
    # data = numpy.genfromtxt(eeg_data, delimiter=',', skip_header=1, usecols=numpy.arange(1,7), filling_values=0)
    data = eeg_data.transpose().to_numpy()

    # print(data)
    # print(eeg_data)


    # # Some information about the channels
    # ch_names = []
    # for i in range(1 to 91):
    #     ch_names
            
        
    # # TODO: finish this list

    # Sampling rate of Muse
    sfreq = 256  # Hz

    # Create the info structure needed by MNE
    info = mne.create_info(columns_list, sfreq, ch_types=data_types)

    # # Finally, create the Raw object
    raw = mne.io.RawArray(data, info)
    print(f'Info is {info}')






    # # Plot it!
    # plot1 = raw.plot()
    # raw.plot(block=True, title="Raw EEG")
    time = timestamps - timestamps[0]
    # change the reshape to be dynamic
    data1 = numpy.reshape(data, (len(data[0]), len(data)))

    print(time)
    # print(time1)
    print(type(time))
    plot(time, data1)
    
    # print(type(data))
    # plt.plot(time[10:2000], data1[10:2000])
    # plt.xlabel('Time(s)')
    # plt.ylabel('Voltage')
    # plt.title("EEG With Timestamps Considered and Filter Applied")
    # plt.show()
    # raw.plot(block=True)


    # create a guard of the type of object passed trough the script
    # reader = csv.open(object, newline='')
    # print(reader.line_num)
    # run the program as a script and pass as argument the csv file
    # load and store csv file
    # transform file into edf
    transformer = []

def fun():
    object = sys.argv[1]
    data = []
    eeg_data = pandas.read_csv(object, usecols=lambda x: x.upper() in ["TIMESTAMPS", "EEG_1", "EEG_2", "EEG_3", "EEG_4", "EEG_5"])
    eeg_data1 = eeg_data.to_numpy()
    eeg_data2 = eeg_data1[0:, 0:1]
    timestamps = eeg_data1[0:, 1:]
    # with open('names.csv', newline='') as csvfile:
        
    #     
    # file1 = csv.writer(file)
    # print(file)
    # 
    
    # timestamps = pandas.read_csv(file, usecols=lambda x: x.upper() in ["TIMESTAMPS"]).to_numpy()
    columns_list = list(eeg_data.columns)
    data_types = []
    for i in range(1, len(columns_list)):
        data_types.append("eeg")
    print(f'eeg_data is {eeg_data}')
    print(f'eeg_data2 is {eeg_data2}')
    data = eeg_data2.transpose()
    print(f'Data length is {len(data)}')
    print(f'Data is {data[0]}')
    sfreq = 256  # Hz
    info = mne.create_info(columns_list, sfreq, ch_types=data_types)
    print(f'Info is {info}')
    raw = mne.io.RawArray(data, info)
    time = timestamps - timestamps[0]
    
    print(eeg_data1)
    print(f'Time is {timestamps}')
    # data1 = numpy.reshape(data, (len(data[0]), len(data)))
    # plot(time, data1)

def filterIntoFrequency(data):
    value = mne.filter.filter_data(data, 256, 8, 12)
    plt.plot(value)
    plt.show()
    

def plot(time, data, start=0, end=0):
    if start == 0 and end == 0:
        plt.plot(time[10:2000], data[10:2000])
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.title("EEG With Timestamps Considered and Filter Applied")
        plt.show()
    else:
        plt.plot(time[start:end], data[start:end])
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage')
        plt.title("EEG With Timestamps Considered and Filter Applied")
        plt.show()

if __name__ == "__main__":
    activate()