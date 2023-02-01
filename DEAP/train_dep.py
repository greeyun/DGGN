import torch
from torch.optim import lr_scheduler
import config
import numpy as np
from torch import nn
from torch.utils.data.dataloader import DataLoader
from model import Classfication
from dataloader import Mydataset, get_xtrain_ytrain
from utils import get_kfold_data, plot_confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score


BATCHSIZE = 8
device = config.device
save_index = r'D:\GY\eeg_recognition'


def traink(model, X_train, y_train, X_test, y_test, BATCH_SIZE, learning_rate, TOTAL_EPOCHS, va=True):

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
    pretrained_params_ar = torch.load(save_index + '/checkpoint_gan_dep_ar' + sub + '.pt')
    pretrained_params_va = torch.load(save_index + '/checkpoint_gan_dep_va' + sub + '.pt')
    model.load_state_dict(pretrained_params_va if va else pretrained_params_ar, strict=False)



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
            # output_label = torch.zeros((0)).to(device)
            # output_predict = torch.zeros((0)).to(device)
            for i, data in enumerate(test_loader):
                in_shots, out_shot, label = data
                in_shots, out_shot, label = in_shots.cuda(), out_shot.cuda(), label.cuda()
                predicted_shot_two = model(in_shots)
                loss = loss_fn(predicted_shot_two, label)
                loss_test += loss.item()
                total_test += predicted_shot_two.size(0)
                test_correct1 = (predicted_shot_two.argmax(1) == label).sum().item()
                correct_test += test_correct1

                # if epoch == TOTAL_EPOCHS-1:
                #     output_label = torch.cat((output_label, label))
                #     output_predict = torch.cat((output_predict, predicted_shot_two.argmax(1)))
                #     # confusion matrix
                #     y_true = output_label.cpu()
                #     y_pred = output_predict.cpu()
                #     plot_confusion_matrix(y_true, y_pred, cmap=plt.cm.Blues, save_flg=True)

        epoch_test_loss = loss_test / total_test
        epoch_test_acc = correct_test / total_test
        f = f1_score(label.detach().cpu(), predicted_shot_two.argmax(1).detach().cpu(), average='macro')
        r = recall_score(label.detach().cpu(), predicted_shot_two.argmax(1).detach().cpu(), average='macro')
        p = precision_score(label.detach().cpu(), predicted_shot_two.argmax(1).detach().cpu(), average='macro')

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
    with open(save_index + '/result_5fold_dep/result' + sub + '_va.txt', 'w') as f:
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
            RECALL_score, PRE_score = traink(net, *data, batch_size, learning_rate, num_epochs, va=True)
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

            torch.save(net.state_dict(), save_index + '/result_5fold_dep/trained_classfication_va_' + sub + '.pth')
        print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10, file=f)

        print('average train loss:{:.4f}, average train accuracy:{:.4f}%'.format(train_loss_sum / k, train_acc_sum / k),
              file=f)
        print('average test loss:{:.4f}, average test accuracy:{:.4f}%'.format(test_loss_sum / k, test_acc_sum / k),
              file=f)
        print('average f1_score:{:.4f}'.format(f1_score_sum / k), file=f)
        print('average RECALL_score:{:.4f}'.format(RECALL_score_sum / k), file=f)
        print('average PRE_score:{:.4f}'.format(PRE_score_sum / k), file=f)
        print('std', np.std(acc, ddof=1), file=f)

    return train_acc_sum / k, test_acc_sum / k, f1_score_sum / k

def k_fold_ar(k, X_train, y_train, num_epochs=1000, learning_rate=0.001, batch_size=64):

    train_loss_sum, test_loss_sum = 0, 0
    train_acc_sum, test_acc_sum = 0, 0
    f1_score_sum, RECALL_score_sum, PRE_score_sum = 0, 0, 0
    acc = []
    with open(save_index + '/result_5fold_dep/result'+ sub +'_ar.txt', 'w') as f:
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
            RECALL_score, PRE_score = traink(net, *data, batch_size, learning_rate, num_epochs, va=False)
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
            torch.save(net.state_dict(),
                       save_index + '/result_5fold_dep/trained_classfication_ar_' + sub + '.pth')

        print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10, file=f)

        print('average train loss:{:.4f}, average train accuracy:{:.4f}%'.format(train_loss_sum / k,
                                                                                 train_acc_sum / k), file=f)
        print('average test loss:{:.4f}, average test accuracy:{:.4f}%'.format(test_loss_sum / k,
                                                                                 test_acc_sum / k), file=f)
        print('average f1_score:{:.4f}'.format(f1_score_sum / k), file=f)
        print('average RECALL_score:{:.4f}'.format(RECALL_score_sum / k), file=f)
        print('average PRE_score:{:.4f}'.format(PRE_score_sum / k), file=f)
        print('std', np.std(acc, ddof=1), file=f)

    return train_acc_sum / k, test_acc_sum / k, f1_score_sum / k



for sub in config.subjectList:

##对唤醒度分类
    with open(save_index + '/result_5fold_dep/result'+ sub +'_ar.txt', 'w') as f:
        print('\n', 'The classfication of arousal', file=f)
        x_train, y_train = get_xtrain_ytrain(sub, va=False)
        train_acc, test_acc, test_f1_score = k_fold_ar(5, x_train, y_train, num_epochs=config.classification_epoch, learning_rate=config.lr_classfication_optimizer, batch_size=BATCHSIZE)
        f.close()
    with open(save_index + '/result_5fold_dep/result_all.txt', 'a') as f:
        print('\n AR' + '*' * 25, '第', sub, '个', '*' * 25, file=f)
        print('average test accuracy:{:.4f}%'.format(train_acc), file=f)
        print('average test accuracy:{:.4f}%'.format(test_acc), file=f)
        print('average test accuracy:{:.4f}%'.format(test_f1_score), file=f)

##对效价分类
    with open(save_index + '/result_5fold_dep/result'+ sub +'_va.txt', 'w') as f:
        print('\n', 'The classfication of valence', file=f)
        x_train, y_train = get_xtrain_ytrain(sub, va=True)
        train_acc, test_acc, test_f1_score = k_fold_va(5, x_train, y_train, num_epochs=config.classification_epoch, learning_rate=config.lr_classfication_optimizer, batch_size=BATCHSIZE)
        f.close()
    with open(save_index + '/result_5fold_dep/result_all.txt', 'a') as f:
        print('\n VA' + '*' * 25, '第', sub, '个', '*' * 25, file=f)
        print('average test accuracy:{:.4f}%'.format(train_acc), file=f)
        print('average test accuracy:{:.4f}%'.format(test_acc), file=f)
        print('average test accuracy:{:.4f}%'.format(test_f1_score), file=f)
