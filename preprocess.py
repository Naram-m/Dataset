import mne
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
from Seizure_times import seizure_times
import numpy as np
from sklearn.model_selection import train_test_split
import collections
import os
CPS_seizures = []
elec_seizures = []
normals = []
labels = []

for patient in [15, 14, 12, 11, 10]:
    if patient in (15, 14, 12, 11):
        files_list=["sz1.edf","sz2.edf","sz3.edf","sz4.edf","sz5.edf","sz6.edf"]
    elif patient == 10:
        files_list = ["sz21.edf","sz31.edf"]

    for file_id, file in enumerate(files_list):
        if patient == 15 and "sz3.edf" in file:
            continue                                                    # seizure duration not determined
        if patient == 14 and ("sz6.edf" in file or "sz3.edf" in file):  # sz3 is corrupted for this patient
            continue                                                    # seizure duration not determined

        file=os.path.join("./edfs", "p"+str(patient)+"_"+file)          # there should be a folder in the same directory called edfs that contains 
        data = mne.io.read_raw_edf(file, preload=True)                  # the files in the following format. eg.,: (p11_sz1.edf) for patient 11 file 1
        

        #print(data.info["ch_names"])
        #print(len(data.info["ch_names"]))

        # get record time
        record_time = data.info['meas_date']
        record_time = record_time.time()
        record_time = datetime.combine(date.today(), record_time)
        # get sizure time and duration
        s_time = time(seizure_times[patient][file_id][0], seizure_times[patient][file_id][1], seizure_times[patient][file_id][2])
        seizure_duration = seizure_times[patient][file_id][3]

        print("File ID: {}, Record time:{}, Seizure time: {}, Seizure duration: {}".format(file_id, record_time, s_time, seizure_duration))

        # get seizure index
        diff = datetime.combine(date.today(), s_time) - record_time # assigned date does not matter
        if diff.days < 0:   # neseccary for patient 12 to get the difference right since record time is one day before seizure time
            tom = record_time.date() + timedelta(days=1)
            diff = datetime.combine(tom, s_time) - record_time

        s_index = int( diff.total_seconds() * 500 ) # 500 sampling rate,
        s_index_end = s_index + int(seizure_duration * 500)
        point_duration = 1  # how many seconds is one point

        
        ###################### Ordering channels ######################

        if patient in (15, 14, 12, 11):
            data.drop_channels(['EEG Cz-Ref', 'EEG Pz-Ref', 'ECG EKG', 'Manual']) # see attached word file for details
        elif patient == 10:
            data.reorder_channels(['EEG Fp2-Ref', 'EEG Fp1-Ref', 'EEG F8-Ref', 'EEG F4-Ref', 'EEG Fz-Ref', 'EEG F3-Ref', 'EEG F7-Ref', 'EEG A2-Ref', 'EEG T4-Ref', 'EEG C4-Ref', 'EEG C3-Ref', 'EEG T3-Ref', 'EEG A1-Ref', 'EEG T6-Ref', 'EEG P4-Ref', 'EEG P3-Ref', 'EEG T5-Ref', 'EEG O2-Ref', 'EEG O1-Ref'])
        print(data.info["ch_names"])
        raw_data = data.get_data()      # ndarray 19 x ~5M

        ###################################################

        raw_data = np.array(raw_data)                   
        seizure_record = raw_data[:, s_index:s_index_end]
        normal_record = np.delete(raw_data, np.s_[s_index:s_index_end], axis=1) # remove all data corresponding to a seizure


        # below should be an if/elif statments to divide seizures types
        if patient == 10:
            for i in range(seizure_duration):
                data_point = seizure_record[:, i * 500:(i + 1) * 500]
                elec_seizures.append(data_point)
        else:
            for i in range(seizure_duration):           # i is the one second interval 
                data_point = seizure_record[:, i*500:(i+1)*500]
                CPS_seizures.append(data_point)         # CPS is set of points, each of which is 19x500

        # normals
        for i in range(seizure_duration): #take same number of normal points
            data_point = normal_record[:, i*500:(i+1)*500]
            normals.append(data_point)

        ########################################################################
        ## to be ~2000 points
        # for i in range(98): #take same number of normal points
        #     data_point = normal_record[:, i*500:(i+1)*500]
        #     normals.append(data_point)

        ########################################################################

        print('==================END FILE==================')
    print('==================END PATIENT==================')


# transpose second and third dim
CPS_seizures = np.array(CPS_seizures)
scaler = np.amax(abs(CPS_seizures))
CPS_seizures = CPS_seizures/scaler

elec_seizures = np.array(elec_seizures)
scaler = np.amax(abs(elec_seizures))
elec_seizures = elec_seizures/scaler

normals = np.array(normals)
scaler = np.amax(abs(normals))
normals = normals/scaler

# plt.figure('Normal')
# plt.plot(normals[100].T)
# plt.figure('Seizure')
# plt.plot(seizures[100].T)
# plt.show()
#

x = np.vstack((elec_seizures, CPS_seizures, normals)) #trying to construct the x #points x 19 x 500

# below order matters
labels.extend([2 for i in range(len(elec_seizures))])
labels.extend([1 for i in range(len(CPS_seizures))])
labels.extend([0 for i in range(len(normals))])

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.1, random_state=1)

np.save("./data_aligned/x_train", x_train)
np.save("./data_aligned/x_test", x_test)
np.save("./data_aligned/y_train", y_train)
np.save("./data_aligned/y_test", y_test)



print("Training: {}".format(collections.Counter(y_train)))
print("Testing: {}".format(collections.Counter(y_test)))

pass