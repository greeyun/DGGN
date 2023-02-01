import torch

########### dir
index = 'D:\GY\data_preprocessed_python\DE'
save_index = 'D:\GY\eeg_recognition/result_5fold_indep'

############ seting
BATCHSIZE = 8

pretrain_epoch = 500
gan_epoch = 50
classification_epoch = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
channels_num = 32

step_size = 3
window_size = 6

in_features = 4
out_features = 64
lstm_features = 128

input_size = 32 * 4
hidden_size = 512

##########  data

subjectList = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
               '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
               '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']


#########  opt

lr_pretrain_optimizer = 0.0005
lr_generator_optimizer = 0.0001
lr_discriminator_optimizer = 0.0001
lr_classfication_optimizer = 0.0005
