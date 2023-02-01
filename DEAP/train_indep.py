import torch
from torch.optim import lr_scheduler
import config
import numpy as np
from torch import nn
from torch.utils.data.dataloader import DataLoader
from model import Classfication
from dataloader import load_input, norm_input, sequence_input, Mydataset
from utils import get_kfold_data
from sklearn.metrics import f1_score, recall_score, precision_score


BATCHSIZE = 8
device = config.device
save_index = r'D:\GY\eeg_recognition'


def traink(model, X_train, y_train, X_test, y_test, BATCH_SIZE, learning_rate, TOTAL_EPOCHS):

    train_loader = DataLoader(Mydataset(X_train[:, :6], X_train[:, -1], y_train), BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(Mydataset(X_test[:, :6], X_test[:, -1], y_test), BATCH_SIZE, shuffle=False, drop_last=True)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=TOTAL_EPOCHS)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    f_score = []
    RECALL_score = []
    PRE_score = []

    pretrained_params = torch.load(save_index + '/generator_sameloss4.pth')
    model.load_state_dict(pretrained_params, strict=False)

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        correct_train = 0
        total_train = 0
        loss_train = 0
        for i, data in enumerate(train_loader):
            optim.zero_grad()
            in_shots, out_shot, label = data
            in_shots, out_shot, label = in_shots.to(device), out_shot.to(device), label.to(device)
            predicted_shot_two = model(in_shots)
            loss = loss_fn(predicted_shot_two, label)
            loss.backward()
            optim.step()

            with torch.no_grad():
                loss_train += loss.item()
                total_train += predicted_shot_two.size(0)
                train_correct1 = (predicted_shot_two.argmax(1) == label).sum().item()
                correct_train += train_correct1

        epoch_acc = correct_train / total_train
        epoch_loss = loss_train / total_train

        exp_lr_scheduler.step()
        model.eval()
        total_test = 0
        correct_test = 0
        loss_test = 0

        with torch.no_grad():
            label_test = torch.zeros((BATCHSIZE))
            predicted_shot_two_test = torch.zeros((BATCHSIZE))
            for i, data in enumerate(test_loader):
                in_shots, out_shot, label = data
                in_shots, out_shot, label = in_shots.cuda(), out_shot.cuda(), label.cuda()
                predicted_shot_two = model(in_shots)
                loss = loss_fn(predicted_shot_two, label)
                loss_test += loss.item()
                total_test += predicted_shot_two.size(0)
                test_correct1 = (predicted_shot_two.argmax(1) == label).sum().item()
                correct_test += test_correct1
                label_test = torch.cat((label_test, label.detach().cpu()))
                predicted_shot_two_test = torch.cat(
                    (predicted_shot_two_test, predicted_shot_two.argmax(1).detach().cpu()))

        epoch_test_loss = loss_test / total_test
        epoch_test_acc = correct_test / total_test
        f = f1_score(label_test, predicted_shot_two_test, average='macro')
        r = recall_score(label_test, predicted_shot_two_test, average='macro')
        p = precision_score(label_test, predicted_shot_two_test, average='macro')

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


def k_fold_va(k, X_train, y_train, num_epochs=100, learning_rate=0.0001, batch_size=BATCHSIZE):
    train_loss_sum, test_loss_sum = 0, 0
    train_acc_sum, test_acc_sum = 0, 0
    f1_score_sum, RECALL_score_sum, PRE_score_sum = 0, 0, 0
    acc = []
    with open(save_index + '/result_5fold_indep/result_va.txt', 'w') as f:
        for i in range(k):
            print('*' * 25, '第', i + 1, '折', '*' * 25, file=f)
            data = get_kfold_data(k, i, X_train, y_train)
            # 获取k折交叉验证的训练和验证数据
            net = Classfication(
                                window_size=6,
                                node_num=32,
                                in_features=4,
                                out_features=64,
                                lstm_features=128
                            ).to(device)
            # 每份数据进行训练
            train_loss, train_acc, test_loss, test_acc, f_score, \
            RECALL_score, PRE_score = traink(net, *data, batch_size, learning_rate, num_epochs)
            acc_test = max(test_acc)
            acc.append(acc_test)

            print('train_loss:%.6f' % train_loss[test_acc.index(max(test_acc))],
                  'trian_acc:%.4f' % train_acc[test_acc.index(max(test_acc))], file=f)
            print('test_loss:%.6f' % test_loss[test_acc.index(max(test_acc))], 'test_acc:%.4f' % max(test_acc), file=f)
            print('f1_score:%.4f' % f_score[test_acc.index(max(test_acc))], file=f)
            print('RECALL_score:%.4f' % RECALL_score[test_acc.index(max(test_acc))], file=f)
            print('PRE_score:%.4f' % PRE_score[test_acc.index(max(test_acc))], file=f)

            train_loss_sum += train_loss[test_acc.index(max(test_acc))]
            test_loss_sum += test_loss[test_acc.index(max(test_acc))]
            train_acc_sum += train_acc[test_acc.index(max(test_acc))]
            test_acc_sum += max(test_acc)
            f1_score_sum += f_score[test_acc.index(max(test_acc))]
            RECALL_score_sum += RECALL_score[test_acc.index(max(test_acc))]
            PRE_score_sum += PRE_score[test_acc.index(max(test_acc))]
        print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10, file=f)

        print('average train loss:{:.4f}, average train accuracy:{:.4f}%'.format(train_loss_sum / k, train_acc_sum / k),
              file=f)
        print('average test loss:{:.4f}, average test accuracy:{:.4f}%'.format(test_loss_sum / k, test_acc_sum / k),
              file=f)
        print('average f1_score:{:.4f}'.format(f1_score_sum / k), file=f)
        print('average RECALL_score:{:.4f}'.format(RECALL_score_sum / k), file=f)
        print('average PRE_score:{:.4f}'.format(PRE_score_sum / k), file=f)
        print('std', np.std(acc, ddof=1), file=f)
    return

def k_fold_ar(k, X_train, y_train, num_epochs=1000, learning_rate=0.0001, batch_size=64):

    train_loss_sum, test_loss_sum = 0, 0
    train_acc_sum, test_acc_sum = 0, 0
    f1_score_sum, RECALL_score_sum, PRE_score_sum = 0, 0, 0
    acc = []
    with open(save_index + '/result_5fold_indep/result_ar.txt', 'w') as f:
        for i in range(k):
            print('*' * 25, '第', i + 1, '折', '*' * 25, file=f)
            data = get_kfold_data(k, i, X_train, y_train)
            # 获取k折交叉验证的训练和验证数据
            net = Classfication(
                window_size=6,
                node_num=32,
                in_features=4,
                out_features=64,
                lstm_features=128
            ).to(device)
            # 每份数据进行训练
            train_loss, train_acc, test_loss, test_acc, f_score, \
            RECALL_score, PRE_score = traink(net, *data, batch_size, learning_rate, num_epochs)
            acc_test = max(test_acc)
            acc.append(acc_test)

            print('train_loss:%.6f' % train_loss[test_acc.index(max(test_acc))],
                  'train_acc:%.4f' % train_acc[test_acc.index(max(test_acc))], file=f)
            print('test_loss:%.6f' % test_loss[test_acc.index(max(test_acc))], 'test_acc:%.4f' % max(test_acc), file=f)
            print('f1_score:%.4f' % f_score[test_acc.index(max(test_acc))], file=f)
            print('RECALL_score:%.4f' % RECALL_score[test_acc.index(max(test_acc))], file=f)
            print('PRE_score:%.4f' % PRE_score[test_acc.index(max(test_acc))], file=f)

            train_loss_sum += train_loss[test_acc.index(max(test_acc))]
            test_loss_sum += test_loss[test_acc.index(max(test_acc))]
            train_acc_sum += train_acc[test_acc.index(max(test_acc))]
            test_acc_sum += max(test_acc)
            f1_score_sum += f_score[test_acc.index(max(test_acc))]
            RECALL_score_sum += RECALL_score[test_acc.index(max(test_acc))]
            PRE_score_sum += PRE_score[test_acc.index(max(test_acc))]

        print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10, file=f)

        print('average train loss:{:.4f}, average train accuracy:{:.4f}%'.format(train_loss_sum / k,
                                                                                 train_acc_sum / k), file=f)
        print('average test loss:{:.4f}, average test accuracy:{:.4f}%'.format(test_loss_sum / k,
                                                                                 test_acc_sum / k), file=f)
        print('average f1_score:{:.4f}'.format(f1_score_sum / k), file=f)
        print('average RECALL_score:{:.4f}'.format(RECALL_score_sum / k), file=f)
        print('average PRE_score:{:.4f}'.format(PRE_score_sum / k), file=f)
        print('std', np.std(acc, ddof=1), file=f)
    return

def k_fold_do(k, X_train, y_train, num_epochs=1000, learning_rate=0.0001, batch_size=64):

    train_loss_sum, test_loss_sum = 0, 0
    train_acc_sum, test_acc_sum = 0, 0
    f1_score_sum, RECALL_score_sum, PRE_score_sum = 0, 0, 0
    acc = []
    with open(save_index + '/result_5fold_indep/result_do.txt', 'w') as f:
        for i in range(k):
            print('*' * 25, '第', i + 1, '折', '*' * 25, file=f)
            data = get_kfold_data(k, i, X_train, y_train)
            # 获取k折交叉验证的训练和验证数据
            net = Classfication(
                window_size=6,
                node_num=32,
                in_features=4,
                out_features=64,
                lstm_features=128
            ).to(device)
            # 每份数据进行训练
            train_loss, train_acc, test_loss, test_acc, f_score, \
            RECALL_score, PRE_score = traink(net, *data, batch_size, learning_rate, num_epochs)
            acc_test = max(test_acc)
            acc.append(acc_test)

            print('train_loss:%.6f' % train_loss[test_acc.index(max(test_acc))],
                  'train_acc:%.4f' % train_acc[test_acc.index(max(test_acc))], file=f)
            print('test_loss:%.6f' % test_loss[test_acc.index(max(test_acc))], 'test_acc:%.4f' % max(test_acc), file=f)
            print('f1_score:%.4f' % f_score[test_acc.index(max(test_acc))], file=f)
            print('RECALL_score:%.4f' % RECALL_score[test_acc.index(max(test_acc))], file=f)
            print('PRE_score:%.4f' % PRE_score[test_acc.index(max(test_acc))], file=f)

            train_loss_sum += train_loss[test_acc.index(max(test_acc))]
            test_loss_sum += test_loss[test_acc.index(max(test_acc))]
            train_acc_sum += train_acc[test_acc.index(max(test_acc))]
            test_acc_sum += max(test_acc)
            f1_score_sum += f_score[test_acc.index(max(test_acc))]
            RECALL_score_sum += RECALL_score[test_acc.index(max(test_acc))]
            PRE_score_sum += PRE_score[test_acc.index(max(test_acc))]

        print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10, file=f)

        print('average train loss:{:.4f}, average train accuracy:{:.4f}%'.format(train_loss_sum / k,
                                                                                 train_acc_sum / k), file=f)
        print('average test loss:{:.4f}, average test accuracy:{:.4f}%'.format(test_loss_sum / k,
                                                                                 test_acc_sum / k), file=f)
        print('average f1_score:{:.4f}'.format(f1_score_sum / k), file=f)
        print('average RECALL_score:{:.4f}'.format(RECALL_score_sum / k), file=f)
        print('average PRE_score:{:.4f}'.format(PRE_score_sum / k), file=f)
        print('std', np.std(acc, ddof=1), file=f)
    return



dealed_X_all, ar_Y_all, va_Y_all, do_Y_all = load_input()
removed_base_normed_X = norm_input(dealed_X_all)
shots_data, target_label = sequence_input(removed_base_normed_X, ar_Y_all, va_Y_all, do_Y_all, arousal=True)

#对效价分类
with open(save_index + '/result_5fold_indep/result_va.txt', 'w') as f:
    print('\n', 'The classfication of valence', file=f)
    x_train, y_train = sequence_input(removed_base_normed_X, ar_Y_all, va_Y_all, do_Y_all, arousal=False)
    k_fold_va(5, x_train, y_train, num_epochs=config.classification_epoch, learning_rate=config.lr_classfication_optimizer, batch_size=BATCHSIZE)


##对唤醒度分类
with open(save_index + '/result_5fold_indep/result_ar.txt', 'w') as f:
    print('\n', 'The classfication of arousal', file=f)
    x_train, y_train = sequence_input(removed_base_normed_X, ar_Y_all, va_Y_all, do_Y_all, arousal=True)
    k_fold_ar(5, x_train, y_train, num_epochs=config.classification_epoch, learning_rate=config.lr_classfication_optimizer, batch_size=BATCHSIZE)


##对主导度分类
with open(save_index + '/result_5fold_indep/result_do.txt', 'w') as f:
    print('\n', 'The classfication of dominance', file=f)
    x_train, y_train = sequence_input(removed_base_normed_X, ar_Y_all, va_Y_all, do_Y_all, dominance=True)
    k_fold_do(5, x_train, y_train, num_epochs=config.classification_epoch, learning_rate=config.lr_classfication_optimizer, batch_size=BATCHSIZE)



