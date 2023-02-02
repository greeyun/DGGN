import numpy as np
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import scipy.io
import os

scaler = StandardScaler()

def get_file_path(root_path, file_list, dir_list):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)
    return dir_list, file_list

def get_seq_data(data_temp, label_temp):
    window = config.window_size
    step_size = config.window_size // 2
    inshots = []
    outshots = []
    label = []

    start = 0
    while start + window < len(data_temp):
        inshots.append(data_temp[start:start + window])
        outshots.append(data_temp[start + window])
        label.append(label_temp[0])
        start = start + step_size
        continue
    inshot_np = np.array(inshots)
    outshot_np = np.array(outshots)
    label_np = np.array(label)

    inshots_data = torch.Tensor(inshot_np)
    outshot_data = torch.Tensor(outshot_np)
    shots_data = torch.cat((inshots_data,outshot_data.unsqueeze(1)),dim=1)
    target_label = torch.LongTensor(label_np)
    return shots_data, target_label

def get_train_test_clip(sub, clip_num):

    root_path = 'D:\GY\data_preprocessed_python\data\SEED\ExtractedFeatures'
    file_list = []
    dir_list = []
    file_list_seq = []
    get_file_path(root_path, file_list, dir_list)
    file_list_seq.extend(file_list[6 * 3:-2])
    file_list_seq.extend(file_list[:6 * 3])
    file_list_seq.extend([file_list[-2]])
    first_trial_file_list = [i for n, i in enumerate(file_list_seq) if n % 3 == 0]

    x_train = np.zeros((0, config.window_size+1, 62, 5))
    y_train = np.zeros(0)

    x_test = np.zeros((0, config.window_size+1, 62, 5))
    y_test = np.zeros(0)

    label = scipy.io.loadmat(first_trial_file_list[-1])

    file = first_trial_file_list[int(sub)-1]
    data = scipy.io.loadmat(file)
    for i in config.clip_num_list:

        data_temp = data['de_movingAve' + i]
        data_temp = scaler.fit_transform(data_temp.transpose(1, 2, 0).reshape(-1, 62)).reshape(-1, 5, 62).transpose(0, 2, 1)
        label_temp = np.array([1 if label['label'][:, int(i)-1] > 0 else 0]).repeat(data_temp.shape[0])
        data_seq, label_seq = get_seq_data(data_temp, label_temp)

        if clip_num != i:
            x_train = np.vstack((x_train, data_seq))
            y_train = np.append(y_train, label_seq)
        else:
            x_test = np.vstack((x_test, data_seq))
            y_test = np.append(y_test, label_seq)

    x_train = torch.Tensor(x_train).permute(0, 1, 3, 2)[:, :, -4:]
    x_test = torch.Tensor(x_test).permute(0, 1, 3, 2)[:, :, -4:]

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    x_train = shuffle(x_train, random_state=2022)
    x_test = shuffle(x_test, random_state=2022)

    y_train = shuffle(y_train, random_state=2022)
    y_test = shuffle(y_test, random_state=2022)
    return x_train, y_train, x_test, y_test

def get_train_test_sub(sub, shots_data, target_label):

    x_train, x_test, y_train, y_test = torch.cat((shots_data[:(int(sub)-1)], shots_data[int(sub):])),  \
                                       shots_data[(int(sub)-1) : int(sub)], \
                                       torch.cat((target_label[:(int(sub)-1)], target_label[int(sub):])), \
                                       target_label[(int(sub)-1) : int(sub)]

    x_train = x_train.reshape(-1,config.window_size+1, 62, 4).permute(0, 1, 3, 2)
    x_test = x_test.reshape(-1,config.window_size+1, 62, 4).permute(0, 1, 3, 2)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return x_train, y_train, x_test, y_test

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, inshots_list,  outshot_list, label_list):
        self.inshots_list = inshots_list
        self.outshot_list = outshot_list
        self.label_list = label_list

    def __getitem__(self, index):
        inshots = self.inshots_list[index]
        outshot = self.outshot_list[index]
        label = self.label_list[index]
        return inshots, outshot, label

    def __len__(self):
        return len(self.inshots_list)

