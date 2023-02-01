import torch
import SEED_config as config
import numpy as np
from torch import nn
from torch.utils.data.dataloader import DataLoader
from SEED_model import Generator, Discriminator, Classfication
from SEED_dataloader import Mydataset, get_train_test_sub
from SEED_utils import EarlyStopping_gan, plot_confusion_matrix, schedule
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

BATCHSIZE = 256
device = config.device

def traink(model, train_loader, test_loader, learning_rate, TOTAL_EPOCHS, sub):

    loss_fn = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    schedule_lr = schedule(optim)
    scheduler_warm = schedule_lr.get_warm_scheduler(milestones=[], warmup_iters=len(train_loader) * 5)
    scheduler_step = schedule_lr.get_milestone_scheduler(milestones=[5, 10], gamma=0.5)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    f_score = []
    RECALL_score = []
    PRE_score = []


    pretrained_params = torch.load(config.save_index + '/checkpoint/subject_independent/checkpoint_gan_indep_' + sub + '.pt')
    model.load_state_dict(pretrained_params, strict=False)

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        correct_train = 0
        total_train = 0
        loss_train = 0

        for i, data in enumerate(train_loader):
            optim.zero_grad()
            in_shots, _, label = data
            in_shots, label = in_shots.to(device), label.to(device)
            predicted_shot_two = model(in_shots)
            loss = loss_fn(predicted_shot_two, label)
            loss.backward()
            if epoch < 5:
                scheduler_warm.step()
            optim.step()

            with torch.no_grad():
                loss_train += loss.item()
                total_train += predicted_shot_two.size(0)
                train_correct1 = (predicted_shot_two.argmax(1) == label).sum().item()
                correct_train += train_correct1

        epoch_acc = correct_train / total_train
        epoch_loss = loss_train / total_train
        if epoch >= 5:
            scheduler_step.step()

        model.eval()
        total_test = 0
        correct_test = 0
        loss_test = 0

        with torch.no_grad():
            # label_test = torch.zeros((BATCHSIZE))
            # predicted_shot_two_test = torch.zeros((BATCHSIZE))
            TP = 0
            FN = 0
            FP = 0
            TN = 0

            for i, data in enumerate(test_loader):
                in_shots, _, label = data
                in_shots, label = in_shots.to(device), label.to(device)
                predicted_shot_two = model(in_shots)
                loss = loss_fn(predicted_shot_two, label)
                loss_test += loss.item()
                total_test += predicted_shot_two.size(0)
                test_correct1 = (predicted_shot_two.argmax(1) == label).sum().item()
                correct_test += test_correct1

                # label_test = torch.cat((label_test, label.detach().cpu()))
                # predicted_shot_two_test = torch.cat((predicted_shot_two_test, predicted_shot_two.argmax(1).detach().cpu()))

                tp = (predicted_shot_two.argmax(1) == label).sum()
                fn = (predicted_shot_two.size(0) - tp)
                TP += tp.item()
                FN += fn.item()

        epoch_test_loss = loss_test / total_test
        epoch_test_acc = correct_test / total_test
        # f = f1_score(label_test, predicted_shot_two_test, average='macro')
        # r = recall_score(label_test, predicted_shot_two_test, average='macro')
        # p = precision_score(label_test, predicted_shot_two_test, average='macro')
        r = TP / (TP+FN)

        if (TP+FP) == 0:
            p = 0
            f = 0.5
        else:
            p = TP / (TP+FP)
            f = (r*p*2) / (r+p)


        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)
        f_score.append(f)
        RECALL_score.append(r)
        PRE_score.append(p)

        print('epoch: ', epoch,
              'loss： ', round(epoch_loss, 5),
              'accuracy:', round(epoch_acc, 4),
              'test_loss： ', round(epoch_test_loss, 5),
              'test_accuracy:', round(epoch_test_acc, 4),
              'F1_score', round(f, 4),
              'recall:', round(r, 4),
              'precision:', round(p, 4),
             )
    print('_________________________________________________________________________________________________________')
    return train_loss, train_acc, test_loss, test_acc, f_score, RECALL_score, PRE_score


def LOSO(sub, learning_rate, batch_size, epoch):

    with open(config.save_index + '/result_LOSO_indep/result_' + sub + '_' + '.txt', 'w') as f:
        X1 = np.load('SEED_data_1STtrial_6Swindow_GAN_2class.npy')
        Y1 = np.load('SEED_label_1STtrial_6Swindow_GAN_2class.npy')
        Y1 = [1 if i > 0 else 0 for i in Y1.flatten()]
        Y1 = np.array(Y1).reshape(15, -1)
        shots_data = torch.Tensor(X1)[:, :, :, :, -4:]
        target_label = torch.LongTensor(Y1)
        x_train, y_train, x_test, y_test = get_train_test_sub(sub, shots_data, target_label)
        print('subject: ', sub)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        train_ds = Mydataset(x_train[:, :config.window_size], x_train[:, -1], y_train)
        test_ds = Mydataset(x_test[:, :config.window_size], x_test[:, -1], y_test)

        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=BATCHSIZE,
                                                   shuffle=True,
                                                   drop_last=True)

        test_loader = torch.utils.data.DataLoader(test_ds,
                                                  batch_size=BATCHSIZE,
                                                  shuffle=False,
                                                  drop_last=True)
        net = Classfication(
            window_size=config.window_size,
            node_num=config.channels_num,
            in_features=config.in_features,
            out_features=config.out_features,
            lstm_features=config.lstm_features
        ).to(device)

        train_loss, train_acc, test_loss, test_acc, \
        f_score, RECALL_score, PRE_score = traink(net, train_loader, test_loader, learning_rate, batch_size, epoch, sub)

        print('train_loss:%.6f' % train_loss[test_acc.index(max(test_acc))],
              'train_acc:%.4f' % train_acc[test_acc.index(max(test_acc))], file=f)
        print('test_loss:%.6f' % test_loss[test_acc.index(max(test_acc))], 'test_acc:%.4f' % max(test_acc), file=f)
        print('f1_score:%.4f' % f_score[test_acc.index(max(test_acc))], file=f)
        print('RECALL_score:%.4f' % RECALL_score[test_acc.index(max(test_acc))], file=f)
        print('PRE_score:%.4f' % PRE_score[test_acc.index(max(test_acc))], file=f)

        torch.save(net.state_dict(),
                   config.save_index + '/result_LOSO_indep/trained_classfication_' + sub + '.pt')

    return train_acc, test_acc, f_score



for sub in config.subjectList:
    ##分类
    with open(config.save_index + '/result_LOSO_indep/result_' + sub + '_' + '.txt', 'w') as f:
        print('The classfication result', file=f)
        train_acc, test_acc, test_f1_score = LOSO(sub, learning_rate=config.lr_classfication_optimizer_indep, batch_size=BATCHSIZE, epoch=config.classification_epoch)
        f.close()

