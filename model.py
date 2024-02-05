# 作者: not4ya
# 时间: 2023/10/8 14:54
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn.dense.linear import Linear


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        # initial projection
        self.proj_het = Linear(self.args.herb_number + self.args.target_number, self.args.hidden_dim, weight_initializer='glorot', bias=True)
        self.proj_herb = Linear(self.args.herb_number, self.args.hidden_dim, weight_initializer='glorot', bias=True)
        self.proj_target = Linear(self.args.target_number, self.args.hidden_dim, weight_initializer='glorot', bias=True)

        # hidden layers: GCN + GAT + GCN
        self.gcn_het_1 = GCNConv(self.args.hidden_dim, self.args.hidden_dim)
        self.gat_het = GATConv(self.args.hidden_dim, self.args.hidden_dim, heads=self.args.heads, concat=False, edge_dim=1)
        self.gcn_het_2 = GCNConv(self.args.hidden_dim, self.args.hidden_dim)

        self.gcn_herb_1 = GCNConv(self.args.hidden_dim, self.args.hidden_dim)
        self.gat_herb = GATConv(self.args.hidden_dim, self.args.hidden_dim, heads=self.args.heads, concat=False, edge_dim=1)
        self.gcn_herb_2 = GCNConv(self.args.hidden_dim, self.args.hidden_dim)

        self.gcn_target_1 = GCNConv(self.args.hidden_dim, self.args.hidden_dim)
        self.gat_target = GATConv(self.args.hidden_dim, self.args.hidden_dim, heads=self.args.heads, concat=False, edge_dim=1)
        self.gcn_target_2 = GCNConv(self.args.hidden_dim, self.args.hidden_dim)

        # CNN combiner
        self.cnn_het = nn.Conv2d(in_channels=2,
                                 out_channels=self.args.hidden_dim,
                                 kernel_size=(self.args.hidden_dim, 1),
                                 stride=1,
                                 bias=True)
        self.cnn_herb = nn.Conv2d(in_channels=2,
                                  out_channels=self.args.hidden_dim,
                                  kernel_size=(self.args.hidden_dim, 1),
                                  stride=1,
                                  bias=True)
        self.cnn_target = nn.Conv2d(in_channels=2,
                                    out_channels=self.args.hidden_dim,
                                    kernel_size=(self.args.hidden_dim, 1),
                                    stride=1,
                                    bias=True)

        # Dropout
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, het_net, het_x, herb_net, herb_x, target_net, target_x):
        het_proj = self.proj_het(het_x)
        het_gcn1 = torch.relu(self.gcn_het_1(het_proj, het_net) + het_proj)
        het_gat = torch.relu(self.gat_het(het_gcn1, het_net) + het_gcn1)
        het_gcn2 = torch.relu(self.gcn_het_2(het_gat, het_net) + het_gat)

        herb_proj = self.proj_herb(herb_x)
        herb_gcn1 = torch.relu(self.gcn_herb_1(herb_proj, herb_net) + herb_proj)
        herb_gat = torch.relu(self.gat_herb(herb_gcn1, herb_net) + herb_gcn1)
        herb_gcn2 = torch.relu(self.gcn_herb_2(herb_gat, herb_net) + herb_gat)

        target_proj = self.proj_target(target_x)
        target_gcn1 = torch.relu(self.gcn_target_1(target_proj, target_net) + target_proj)
        target_gat = torch.relu(self.gat_target(target_gcn1, target_net) + target_gcn1)
        target_gcn2 = torch.relu(self.gcn_target_2(target_gat, target_net) + target_gat)

        X_het = torch.cat((het_gcn1, het_gcn2), 1).t()
        X_het = X_het.view(1, 2, self.args.hidden_dim, -1)

        X_herb = torch.cat((herb_gcn1, herb_gcn2), 1).t()
        X_herb = X_herb.view(1, 2, self.args.hidden_dim, -1)

        X_target = torch.cat((target_gcn1, target_gcn2), 1).t()
        X_target = X_target.view(1, 2, self.args.hidden_dim, -1)

        het_embedding = self.cnn_het(X_het)
        het_embedding = het_embedding.view(self.args.hidden_dim, self.args.herb_number + self.args.target_number).t()

        herb_embedding = self.cnn_herb(X_herb)
        herb_embedding = herb_embedding.view(self.args.hidden_dim, self.args.herb_number).t()

        target_embedding = self.cnn_target(X_target)
        target_embedding = target_embedding.view(self.args.hidden_dim, self.args.target_number).t()

        het_embedding = self.dropout(het_embedding)
        herb_embedding = self.dropout(herb_embedding)
        target_embedding = self.dropout(target_embedding)

        C = het_embedding[0:self.args.herb_number, :]
        D = het_embedding[self.args.herb_number:, :]
        herb_embedding = torch.cat((C, herb_embedding), 1)
        target_embedding = torch.cat((D, target_embedding), 1)

        return herb_embedding.mm(target_embedding.t())
