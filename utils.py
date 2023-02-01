import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torchvision import transforms
import itertools
from sklearn.metrics import confusion_matrix
import config
from sklearn import manifold


def knn_value(x):
    inner = -2*torch.matmul(x.transpose(3, 2), x)
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(3, 2)
    value = abs(pairwise_distance)
    return value


def del_base_de(X, Z):
    undeal_X = X.reshape(40, -1, 5, 32).transpose([1, 0, 2, 3]).reshape(-1, 40, 160)
    dealed_X = undeal_X - Z
    return dealed_X.reshape(-1, 40, 5, 32).transpose([1, 0, 2, 3])


class EarlyStopping_gan:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path_indep=config.save_index + '/checkpoint_gan_indep.pt', trace_func=print, dep=True, sub='01',va=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dep = dep
        self.sub = sub
        self.va = va
        self.path_indep = path_indep
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        path_dep_va = 'checkpoint_gan_dep_va' + self.sub + '.pt'
        path_dep_ar = 'checkpoint_gan_dep_ar' + self.sub + '.pt'
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.va:
            torch.save(model.state_dict(), path_dep_va if self.dep else self.path_indep)
        else:
            torch.save(model.state_dict(), path_dep_ar if self.dep else self.path_indep)
        self.val_loss_min = val_loss


def get_kfold_data(k, i, X, y):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_test为测试数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 test
        X_part, y_part = X[idx, :], y[idx]
        if j == i:  ###第i折作test
            X_test, y_test = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)

    return X_train, y_train, X_test, y_test


def plot_confusion_matrix(y_true, y_pred, cmap=plt.cm.Blues,save_flg=False):
    classes=[ '1' , '0']
    labels = range(2)
    cm = confusion_matrix(y_true,y_pred,labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_norm = np.round(cm_norm, 2)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np. arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=7.5)
    plt.yticks(tick_marks, classes,fontsize=7.5)
    plt.rcParams['font.sans-serif']=["TimesNewRoman"]
    plt.rcParams['axes.unicode_minus'] =False
    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),range(cm. shape[1])):
        plt.text(j, i, cm_norm[i,j],
            horizontalalignment="center",
            color="white" if cm_norm[i,j]> thresh else "black")
    plt.ylabel("True label", fontsize=7.5, fontfamily="TimesNewRoman")
    plt.xlabel("Predicted label", fontsize=7.5, fontfamily="TimesNewRoman")
    if save_flg:
        plt.savefig("./ confusion_matrix_s01_va.png")
    plt.show()


def plot_tsne(x, label):
    tsne = manifold.TSNE(n_components=2,  random_state=501) # init='pca',
    X_tsne = tsne.fit_transform(x)
    print("Org data dimension is {}.Embedded data dimension is {}".format(x.shape[-1], X_tsne.shape[-1]))
    color = ['r' if i==0 else 'b' for i in label.cpu().detach().numpy()]
    marker = 'o'

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(dpi=300, figsize=(4, 4))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], s=3, c=color, marker=marker)
    plt.scatter(X_norm[0, 0], X_norm[0, 1], s=3, c=color[0], marker=marker, label='0' )
    plt.scatter(X_norm[-10, 0], X_norm[-10, 1], s=3, c=color[-10], marker=marker, label='1' )
    plt.legend(loc='lower right')
    plt.savefig(r'C:\Users\Administrator\Desktop/S01.png')
    plt.show()


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).cuda()  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1