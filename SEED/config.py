import torch

########### dir
index = r'D:\GY\eeg_recognition\seed\DE'
save_index = r'D:\GY\eeg_recognition\seed'

############ seting
BATCHSIZE = 8

pretrain_epoch = 200
gan_epoch = 20
classification_epoch = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
channels_num = 62

step_size = 3
window_size = 6

in_features = 4
out_features = 64
lstm_features = 64

input_size = 248
hidden_size = 248

##########  data
subjectList = [str(i) for i in range(1, 16)]
clip_num_list = ['1', '3', '4', '6', '7', '9', '10', '12', '14', '15']

#########  opt
lr_pretrain_optimizer = 0.0001
lr_generator_optimizer = 0.00001
lr_discriminator_optimizer = 0.00001
lr_classfication_optimizer = 0.0001
lr_classfication_optimizer_indep = 0.00001
