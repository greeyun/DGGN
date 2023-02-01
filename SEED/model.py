import torch
import torch.nn as nn
from torch.nn import init
from SEED_utils import knn_value
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GraphConvolution(nn.Module):

    def __init__(self, window_size, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weights = nn.Parameter(
            torch.Tensor(window_size,in_features, out_features)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.weights)

    def forward(self, adjacency, nodes):
        batch_size = adjacency.size(0)
        window_size, in_features, out_features = self.weights.size()
        weights = self.weights.unsqueeze(0).expand(batch_size, window_size, in_features, out_features)
        output = adjacency.matmul(nodes).matmul(weights)
        return output

class Generator(nn.Module):

    def __init__(self, window_size, node_num, in_features, out_features, lstm_features):
        super(Generator, self).__init__()
        self.window_size = window_size
        self.node_num = node_num
        self.in_features = in_features
        self.out_features = out_features
        self.gcn = GraphConvolution(window_size, in_features, out_features)
        self.lstm = nn.LSTM(
            input_size=out_features * node_num,
            hidden_size=lstm_features,
            num_layers=1,
            batch_first=True
        )
        self.ffn = nn.Sequential(

            nn.Linear(lstm_features, node_num * in_features),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, nodes):
        adj = knn_value(nodes)
        batch_size, window_size, node_num = adj.size()[0: 3]
        eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
        adjs = adj + eye
        diag = adjs.sum(dim=-1, keepdim=True).pow(-0.5).expand(adjs.size()) * eye
        adjacency = diag.matmul(adjs).matmul(diag)
        nodes = nodes.permute([0,1,3,2])
        gcn_output = self.gcn(adjacency, nodes)
        gcn_output = gcn_output.view(batch_size, window_size, -1)
        rout, (_, _) = self.lstm(gcn_output)
        output = rout[:, -1, :]
        output = self.ffn(output)
        return output

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_size, 2),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.ffn2(self.ffn1(input)).squeeze(-1)
        return output

class Classfication(nn.Module):

    def __init__(self, window_size, node_num, in_features, out_features, lstm_features):
        super(Classfication, self).__init__()
        self.window_size = window_size
        self.node_num = node_num
        self.in_features = in_features
        self.out_features = out_features
        self.gcn = GraphConvolution(window_size, in_features, out_features)
        self.bn = nn.BatchNorm1d(self.window_size)
        self.lstm = nn.LSTM(
            input_size=out_features * node_num,
            hidden_size=lstm_features,
            num_layers=1,
            batch_first=True
        )
        self.ffn = nn.Sequential(

            nn.Linear(lstm_features, node_num * in_features),
            nn.Linear(node_num * in_features, 2),
            nn.Sigmoid()
        )

    def forward(self, nodes):
        adj = knn_value(nodes)
        batch_size, window_size, node_num = adj.size()[0: 3]
        eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
        adjs = adj + eye
        diag = adjs.sum(dim=-1, keepdim=True).pow(-0.5).expand(adjs.size()) * eye
        adjacency = diag.matmul(adjs).matmul(diag)
        nodes = nodes.permute([0,1,3,2])
        gcn_output = self.gcn(adjacency, nodes)
        gcn_output = gcn_output.view(batch_size, window_size, -1)
        gcn_output = self.bn(gcn_output)
        rout, (_, _) = self.lstm(gcn_output)
        output = rout[:, -1, :]
        output = self.ffn(output)
        return output


