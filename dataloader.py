import numpy as np
import torch
from sklearn.utils import shuffle
import config
import utils


def load_input():
    dealed_X_all = np.zeros((160))
    ar_Y_all = np.zeros((0))
    va_Y_all = np.zeros((0))
    do_Y_all = np.zeros((0))
    for sub in config.subjectList:
        with open(config.index + '/decomposed_de_' + sub + '.npy', 'rb') as fileTrain:
            X  = np.load(fileTrain)
        with open(config.index + '/base_DE_' + sub + '.npy', 'rb') as fileTrainZ:
            Z  = np.load(fileTrainZ)
        with open(config.index + '/labels_de_' + sub + '.npy', 'rb') as fileTrainY:
            Y  = np.load(fileTrainY)
        ar_Y_all = np.append(ar_Y_all, Y[:, 0, 0].reshape(40, -1))
        va_Y_all = np.append(va_Y_all, Y[:, 0, 1].reshape(40, -1))
        do_Y_all = np.append(va_Y_all, Y[:, 0, 2].reshape(40, -1))
        dealed_X = utils.del_base_de(X, Z).reshape(-1, 160)
        dealed_X_all = np.vstack((dealed_X_all,dealed_X))
    dealed_X_all = dealed_X_all[1:]
    return dealed_X_all, ar_Y_all, va_Y_all, do_Y_all

def norm_input(dealed_X_all):
    removed_base_X = dealed_X_all
    removed_base_normed_X = (removed_base_X - removed_base_X.min()) / (removed_base_X - removed_base_X.min()).max()
    return removed_base_normed_X

def sequence_input(removed_base_normed_X, ar_Y_all, va_Y_all, do_Y_all, arousal=True, dominance=False):

    data_x = removed_base_normed_X.reshape(-1, 19, 5, 32)[:, :, -4:, :]
    data_y = ar_Y_all.reshape(-1, 19) if arousal is True else va_Y_all.reshape(-1, 19)
    data_y = do_Y_all.reshape(-1, 19) if dominance else data_y
    window = config.window_size
    step_size = config.step_size
    inshots = []
    outshot = []
    label = []
    for k, (i, j) in enumerate(zip(data_x, data_y)):
        start = 0
        while start + window < len(i):
            inshots.append(i[start:start + window])
            outshot.append(i[start + window])
            label.append(j[0])
            start = start + step_size
            continue
    inshot_np = np.array(inshots)
    outshot_np = np.array(outshot)
    label_list = [1 if i > 5 else 0 for i in label]
    label_np = np.array(label_list)
    inshots_data = torch.Tensor(inshot_np)
    outshot_data = torch.Tensor(outshot_np)
    shots_data = torch.cat((inshots_data, outshot_data.unsqueeze(1)), dim=1)
    target_label = torch.LongTensor(label_np)
    shots_data = shuffle(shots_data, random_state=2022)
    target_label = shuffle(target_label, random_state=2022)
    return shots_data, target_label

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

def get_xtrain_ytrain( sub, va=True, do=False):

    with open(config.index + '/decomposed_de_' + sub + '.npy', 'rb') as fileTrain:
        X  = np.load(fileTrain)
    with open(config.index + '/labels_de_' + sub + '.npy', 'rb') as fileTrainL:
        Y  = np.load(fileTrainL)
    with open(config.index + '/base_DE_' + sub + '.npy', 'rb') as fileTrainZ:
        Z  = np.load(fileTrainZ)
    ar_Y_all = Y[:, 0, 0].reshape(40, -1)
    va_Y_all = Y[:, 0, 1].reshape(40, -1)
    do_Y_all = Y[:, 0, 2].reshape(40, -1)
    dealed_X = utils.del_base_de(X, Z).reshape(-1, 160)
    removed_base_X = dealed_X
    removed_base_normed_X = (removed_base_X - removed_base_X.min()) / (removed_base_X - removed_base_X.min()).max()
    data_x = removed_base_normed_X.reshape(-1, 19, 5, 32)[:, :, -4:, :]
    data_ar = ar_Y_all.reshape(-1, 19)
    data_va = va_Y_all.reshape(-1, 19)
    data_do = do_Y_all.reshape(-1, 19)
    data_y = data_va if va else data_ar

    data_y = data_do if do else data_y

    window = config.window_size
    step_size = config.step_size
    inshots = []
    outshot = []
    label = []
    for k, (i, j) in enumerate(zip(data_x, data_y)):
        start = 0
        while start + window < len(i):
            inshots.append(i[start:start + window])
            outshot.append(i[start + window])
            label.append(j[0])
            start = start + step_size
            continue
    inshot_np = np.array(inshots)
    outshot_np = np.array(outshot)
    label_list = [1 if i > 5 else 0 for i in label]
    label_np = np.array(label_list)

    inshots_data = torch.Tensor(inshot_np)
    outshot_data = torch.Tensor(outshot_np)

    shots_data = torch.cat((inshots_data, outshot_data.unsqueeze(1)), dim=1)
    target_label = torch.LongTensor(label_np)

    x_train = shuffle(shots_data, random_state=2022)
    y_train = shuffle(target_label, random_state=2022)

    return x_train, y_train