import numpy as np
import math
import pickle
from scipy.signal import butter, lfilter

subjectList = [str('0')+str(i) if len(str(i))==1 else str(i) for i in range(33)][1:]
window_size = 128
step_size = 128
frequency = 128
band = [4,8,14,31,45] # bands
channels = [i for i in range(32)]
channels_num = len(channels)
per_DE_num = int((8064-384)/step_size)
bands_num = len(band)-1

#------------------------------------------------------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#------------------------------------------------------------------------------------------
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#------------------------------------------------------------------------------------------
def compute_DE(signal):
    variance = np.var(signal,ddof=1)
    return math.log(2*math.pi*math.e*variance)/2

#------------------------------------------------------------------------------------------
def save_de_data_label_base(sub):
    with open("D:\data_preprocessed_python/s/s" + sub + '.dat', 'rb') as file:
        subject = pickle.load(file, encoding='latin1')
        decomposed_de = np.empty([0,bands_num,per_DE_num])

        base_DE = np.empty([0,channels_num*bands_num])
        labels_de = np.empty([0])
        for trial in range(40):
            print(trial)
            data = subject["data"][trial]
            labels = subject["labels"][trial]

            temp_base_DE = np.empty([0])
            temp_base_delta_DE = np.empty([0])
            temp_base_theta_DE = np.empty([0])
            temp_base_alpha_DE = np.empty([0])
            temp_base_beta_DE = np.empty([0])
            # temp_base_gamma_DE = np.empty([0])

            temp_de = np.empty([0,per_DE_num])
            for channel in channels:
                trial_signal = data[channel,384:]
                base_signal = data[channel,:384]
                #****************compute base DE****************
                # base_delta （384，）
                base_delta = butter_bandpass_filter(base_signal, band[0], band[1], frequency, order=3)
                base_theta = butter_bandpass_filter(base_signal, band[1], band[2], frequency, order=3)
                base_alpha = butter_bandpass_filter(base_signal, band[2], band[3], frequency, order=3)
                base_beta = butter_bandpass_filter(base_signal, band[3], band[4], frequency, order=3)
                # base_gamma = butter_bandpass_filter(base_signal, band[4], band[5], frequency, order=3)
                # base_delta_DE （1，）

                base_delta_DE = (compute_DE(base_delta[:128])+compute_DE(base_delta[128:256])+compute_DE(base_delta[256:]))/3
                base_theta_DE = (compute_DE(base_theta[:128])+compute_DE(base_theta[128:256])+compute_DE(base_theta[256:]))/3
                base_alpha_DE =(compute_DE(base_alpha[:128])+compute_DE(base_alpha[128:256])+compute_DE(base_alpha[256:]))/3
                base_beta_DE =(compute_DE(base_beta[:128])+compute_DE(base_beta[128:256])+compute_DE(base_beta[256:]))/3
                # base_gamma_DE =(compute_DE(base_gamma[:128])+compute_DE(base_gamma[128:256])+compute_DE(base_gamma[256:]))/3

                temp_base_delta_DE = np.append(temp_base_delta_DE,base_delta_DE)
                temp_base_theta_DE = np.append(temp_base_theta_DE,base_theta_DE)
                temp_base_alpha_DE = np.append(temp_base_alpha_DE,base_alpha_DE)
                temp_base_beta_DE = np.append(temp_base_beta_DE,base_beta_DE)
                # temp_base_gamma_DE = np.append(temp_base_gamma_DE,base_gamma_DE)

                # delta （7680，）
                delta = butter_bandpass_filter(trial_signal, band[0], band[1], frequency, order=3)
                theta = butter_bandpass_filter(trial_signal, band[1], band[2], frequency, order=3)
                alpha = butter_bandpass_filter(trial_signal, band[2], band[3], frequency, order=3)
                beta = butter_bandpass_filter(trial_signal, band[3], band[4], frequency, order=3)
                # gamma = butter_bandpass_filter(trial_signal, band[4], band[5], frequency, order=3)

                DE_delta = np.zeros(shape=[0],dtype = float)
                DE_theta = np.zeros(shape=[0],dtype = float)
                DE_alpha = np.zeros(shape=[0],dtype = float)
                DE_beta =  np.zeros(shape=[0],dtype = float)
                # DE_gamma = np.zeros(shape=[0],dtype = float)

                start = 0
                while start + window_size <= data.shape[1]-384:

                    DE_delta = np.append(DE_delta,compute_DE(delta[start : start + window_size]))
                    DE_theta = np.append(DE_theta,compute_DE(theta[start : start + window_size]))
                    DE_alpha = np.append(DE_alpha,compute_DE(alpha[start : start + window_size]))
                    DE_beta = np.append(DE_beta,compute_DE(beta[start : start + window_size]))
                    # DE_gamma = np.append(DE_gamma,compute_DE(gamma[start : start + window_size]))
                    labels_de = np.append(labels_de,labels)
                    start = start + step_size

                temp_de = np.vstack([temp_de,DE_delta])
                temp_de = np.vstack([temp_de,DE_theta])
                temp_de = np.vstack([temp_de,DE_alpha])
                temp_de = np.vstack([temp_de,DE_beta])
                # temp_de = np.vstack([temp_de,DE_gamma])

            temp_trial_de = temp_de.reshape(-1,bands_num,per_DE_num)
            decomposed_de = np.vstack([decomposed_de,temp_trial_de])

            temp_base_DE = np.append(temp_base_DE,temp_base_delta_DE)
            temp_base_DE = np.append(temp_base_DE,temp_base_theta_DE)
            temp_base_DE = np.append(temp_base_DE,temp_base_alpha_DE)
            temp_base_DE = np.append(temp_base_DE,temp_base_beta_DE)
            # temp_base_DE = np.append(temp_base_DE,temp_base_gamma_DE)

            base_DE = np.vstack([base_DE,temp_base_DE])
        # print(decomposed_de.shape)  # 40*32,4,60
        # print(labels_de.shape)  # 40*32*4*60
        # print(base_DE.shape)  # 40,32*4
        decomposed_de = decomposed_de.reshape(-1,channels_num,bands_num,per_DE_num).transpose([0,3,2,1]).reshape(-1,bands_num,channels_num).reshape(-1,channels_num*bands_num)
        labels_de = labels_de.reshape(-1,channels_num,per_DE_num,4).transpose([0,2,1,3]).reshape(-1,channels_num,4)

        np.save('D:\GY\data_preprocessed_python\DE_nolap4/base_DE_' + sub, base_DE, allow_pickle=True, fix_imports=True)
        np.save('D:\GY\data_preprocessed_python\DE_nolap4/decomposed_DE_' + sub , decomposed_de, allow_pickle=True, fix_imports=True)
        np.save('D:\GY\data_preprocessed_python\DE_nolap4/labels_DE_' + sub, labels_de, allow_pickle=True, fix_imports=True)

#------------------------------------------------------------------------------------------
def main():
    for sub in subjectList:
        save_de_data_label_base(sub)

#------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
