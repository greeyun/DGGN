import torch
import config
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data.dataloader import DataLoader
from model import Generator, Discriminator, Classfication
from dataloader import load_input, norm_input, sequence_input, Mydataset
from utils import EarlyStopping_gan
from torch.autograd import Variable

BATCHSIZE = 16
device = config.device
dealed_X_all, ar_Y_all, va_Y_all, do_Y_all = load_input()
removed_base_normed_X = norm_input(dealed_X_all)
shots_data, target_label = sequence_input(removed_base_normed_X, ar_Y_all, va_Y_all, do_Y_all, dominance=True)
x_train, x_test, y_train, y_test = train_test_split(shots_data, target_label, test_size=0.2, random_state=2022)
train_ds = Mydataset(x_train[:, :6], x_train[:, -1], y_train)
test_ds = Mydataset(x_test[:, :6], x_test[:, -1], y_test)

train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=BATCHSIZE,
                                           shuffle=True,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(test_ds,
                                          batch_size=BATCHSIZE,
                                          shuffle=False,
                                          drop_last=True)

sample_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=BATCHSIZE,
                                           shuffle=False,
                                           drop_last=True)

generator = Generator(
    window_size=config.window_size,
    node_num=config.channels_num,
    in_features=config.in_features,
    out_features=config.out_features,
    lstm_features=config.lstm_features
).to(config.device)

discriminator = Discriminator(
    input_size=config.input_size,
    hidden_size=config.hidden_size
).to(config.device)

classfication = Classfication(
    window_size=config.window_size,
    node_num=config.channels_num,
    in_features=config.in_features,
    out_features=config.out_features,
    lstm_features=config.lstm_features
).to(config.device)

mse = nn.MSELoss(reduction='sum')
criterion = nn.CrossEntropyLoss(reduction='mean')


pretrain_optimizer = torch.optim.RMSprop(generator.parameters(), lr=config.lr_pretrain_optimizer)
generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=config.lr_generator_optimizer)
discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=config.lr_discriminator_optimizer)
classfication_optimizer = torch.optim.RMSprop(classfication.parameters(), lr=config.lr_classfication_optimizer)

print('pretrain generator')
pretrain_loss = []
for epoch in range(config.pretrain_epoch):
    train_losses = []
    for i, data in enumerate(train_loader):
        pretrain_optimizer.zero_grad()
        in_shots, out_shot, label = data
        in_shots, out_shot, label = in_shots.to(device), out_shot.to(device), label.to(device)
        predicted_shot = generator(in_shots)
        out_shot = out_shot.view(BATCHSIZE, -1)
        loss = mse(predicted_shot, out_shot)
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), 5)
        pretrain_optimizer.step()
        train_losses.append(loss.item())
    train_loss = np.average(train_losses)
    print('[epoch %d] [loss %.4f]' % (epoch, train_loss.item()))


early_stopping_gan = EarlyStopping_gan(patience=7, verbose=True, dep=False)
print('train GAN')
pretrain_gan_loss = []
correct = 0
total_train = 0
for epoch in range(config.gan_epoch):
    discriminator_losses = []
    generator_losses = []
    mse_losses = []

    for i, (data, sample) in enumerate(zip(train_loader, sample_loader)):
        # update discriminator
        discriminator_optimizer.zero_grad()     # 判别器梯度清零
        generator_optimizer.zero_grad()         # 生成器梯度清零
        real_label = Variable(torch.ones(BATCHSIZE, dtype=torch.int64)).to(device)   # 定义真实的图片label为1
        fake_label = Variable(torch.zeros(BATCHSIZE, dtype=torch.int64)).to(device)  # 定义假的图片的label为0
        # 输入序列-batch*time_step*channel*feather,
        # 输出后一秒的信号-batch*channel*feather,
        # 标签-batch
        in_shots, out_shot, label = data
        in_shots, out_shot, label = in_shots.to(device), out_shot.to(device), label.to(device)
        predicted_shot = generator(in_shots)    # 获得预测next的信号
        _, sample, label_sample = sample
        sample = sample.to(device)
        sample = sample.view(BATCHSIZE, -1)     # 真实next的信号
        real_logit = discriminator(sample)      # 真实next的信号得到的label，越靠近1越好
        fake_logit = discriminator(predicted_shot)      # 预测信号得到的label，越靠近0越好
        d_loss_real = criterion(real_logit, real_label)
        d_loss_fake = criterion(fake_logit, fake_label)

        discriminator_loss = d_loss_real + d_loss_fake
        discriminator_loss.backward()
        discriminator_optimizer.step()
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)
        generator_loss = criterion(discriminator(generator(in_shots)), real_label)
        generator_loss.backward()
        generator_optimizer.step()
        mse_loss = mse(predicted_shot, out_shot.reshape(BATCHSIZE,-1))

        discriminator_losses.append(discriminator_loss.item())
        generator_losses.append(generator_loss.item())
        mse_losses.append(mse_loss.item())

    av_discriminator_loss = np.average(discriminator_losses)
    av_generator_loss = np.average(generator_losses)
    av_mse_loss = np.average(mse_losses)
    print('[epoch %d][d_loss %.4f] [g_loss %.4f] [mse_loss %.4f]' % (epoch,
            av_discriminator_loss.item(), av_generator_loss.item(), av_mse_loss.item()))
    pretrain_gan_loss.append(av_discriminator_loss.item())
    early_stopping_gan(av_mse_loss, generator)





