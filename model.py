"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn

import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax, GATConv, GraphConv, SAGEConv


class GCN(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 hidden_dim,
                 num_classes,
                 dropout = 0
                 ):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.gcn_layers = []
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GraphConv(hidden_dim, hidden_dim))

        self.gcn_layers.append(GraphConv(hidden_dim, num_classes))

    def forward(self, g, inputs):
        h = inputs
        middle_feats = []
        for l in range(self.num_layers):
            if l != 0:
                h = self.dropout(h)
            h = self.gcn_layers[l](g, h)
            middle_feats.append(h)
            h = F.relu(h)
        # output projection
        logits = self.gcn_layers[-1](g, h)

        return logits, middle_feats


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 hidden_dim,
                 num_classes,
                 heads,
                 activation=F.elu,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, hidden_dim, heads[0],
            feat_drop, attn_drop, negative_slope, False, activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                hidden_dim * heads[l - 1], hidden_dim, heads[l],
                feat_drop, attn_drop, negative_slope, residual, activation))
        # output projection
        self.gat_layers.append(GATConv(
            hidden_dim * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        middle_feats = []
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
            middle_feats.append(h)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)

        return logits, middle_feats


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation=F.relu,
                 dropout=0,
                 aggregator_type='mean'):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))  # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        mid_feats = []
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
                mid_feats.append(h)
        return h, mid_feats


class GCN_ens(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 hidden_dim,
                 num_classes,
                 num_branches,
                 dropout=0
                 ):
        super(GCN_ens, self).__init__()
        self.num_branches = num_branches
        for i in range(self.num_branches):
            setattr(self, 'gcn_base_' + str(i),
                    GCN(num_layers, in_dim, hidden_dim, num_classes, dropout=dropout))

    def forward(self, g, input):
        output = []
        featss = []
        for i in range(self.num_branches):
            branch = getattr(self, 'gcn_base_' + str(i))
            y, feats = branch(g, input)
            y = torch.unsqueeze(y, dim=0)
            output.append(y)
            featss.append(feats)

        output = torch.cat(output, dim=0)
        return output, featss


class GAT_ens(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 hidden_dim,
                 num_classes,
                 num_branches,
                 heads,
                 activation=F.relu,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=False):
        super(GAT_ens, self).__init__()
        self.num_branches = num_branches
        for i in range(self.num_branches):
            setattr(self, 'sage_base_' + str(i),
                    GAT(num_layers, in_dim, hidden_dim, num_classes, heads, activation,
                        feat_drop, attn_drop, negative_slope, residual))

    def forward(self, g, input):
        output = []
        featss = []
        for i in range(self.num_branches):
            branch = getattr(self, 'sage_base_' + str(i))
            y, feats = branch(g, input)
            y = torch.unsqueeze(y,dim=0)
            output.append(y)
            featss.append(feats)

        output = torch.cat(output,dim=0)
        return output, featss


class SAGE_ens(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 num_branches,
                 activation=F.relu,
                 dropout=0,
                 aggregator_type='mean'):
        super(SAGE_ens, self).__init__()
        self.num_branches = num_branches
        for i in range(self.num_branches):
            setattr(self, 'sage_base_' + str(i),
                    GraphSAGE(in_feats,
                              n_hidden,
                              n_classes,
                              n_layers,
                              activation=activation,
                              dropout=dropout,
                              aggregator_type=aggregator_type))

    def forward(self, g, input):
        output = []
        featss = []
        for i in range(self.num_branches):
            branch = getattr(self, 'sage_base_' + str(i))
            y, feats = branch(g, input)
            y = torch.unsqueeze(y,dim=0)
            output.append(y)
            featss.append(feats)

        output = torch.cat(output, dim=0)
        return output, featss

class SAGE_dif(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden1,
                 n_hidden2,
                 n_classes,
                 n_layers,
                 num_branches,
                 activation=F.relu,
                 dropout=0,
                 aggregator_type='mean'):
        super(SAGE_dif, self).__init__()
        self.num_branches = num_branches
        for i in range(self.num_branches):
            if i % 2 == 0:
                setattr(self, 'sage_base_' + str(i),
                        GraphSAGE(in_feats,
                                  n_hidden1,
                                  n_classes,
                                  n_layers,
                                  activation=activation,
                                  dropout=dropout,
                                  aggregator_type=aggregator_type))
            else:
                setattr(self, 'sage_base_' + str(i),
                        GraphSAGE(in_feats,
                                  n_hidden2,
                                  n_hidden1,
                                  n_layers-2,
                                  activation=activation,
                                  dropout=dropout,
                                  aggregator_type=aggregator_type))
                setattr(self, 'sage_out_' + str(i), SAGEConv(n_hidden1, n_classes, aggregator_type))

    def forward(self, g, input):
        output = []
        featss = []
        for i in range(self.num_branches):
            if i % 2 == 0:
                branch = getattr(self, 'sage_base_' + str(i))
                y, feats = branch(g, input)
                y = torch.unsqueeze(y,dim=0)
                output.append(y)
                featss.append(feats)
            else:
                branch_emb = getattr(self, 'sage_base_' + str(i))
                emb, feats =  branch_emb(g, input)
                feats.append(emb)
                branch_out = getattr(self, 'sage_out_' + str(i))
                y = branch_out(g, emb)
                y = torch.unsqueeze(y, dim=0)
                output.append(y)
                featss.append(feats)

        output = torch.cat(output, dim=0)

        return output, featss

class GAT_dif(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 hidden_dim1,
                 hidden_dim2,
                 num_classes,
                 num_branches,
                 heads,
                 activation=F.elu,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=False):
        super(GAT_dif, self).__init__()
        self.num_branches = num_branches
        for i in range(self.num_branches):
            if i % 2 == 0 :
                setattr(self, 'gat_base_' + str(i),
                        GAT(num_layers, in_dim, hidden_dim1, num_classes, heads, activation,
                            feat_drop, attn_drop, negative_slope, residual))
            else:
                heads.pop()
                setattr(self,'gat_enc_'+str(i),
                    GAT(num_layers-1, in_dim, hidden_dim2, hidden_dim1, heads, activation,
                        feat_drop, attn_drop, negative_slope, residual))
                setattr(self,'gat_out_'+str(i), GATConv(
                hidden_dim1 * heads[-1], num_classes, 1,
                feat_drop, attn_drop, negative_slope, residual, activation))


    def forward(self, g, input):
        output = []
        featss = []
        for i in range(self.num_branches):
            if i % 2 ==0 :
                branch = getattr(self, 'gat_base_' + str(i))
                y, feats = branch(g, input)
                y = torch.unsqueeze(y,dim=0)
                output.append(y)
                featss.append(feats)
            else:
                branch_emb = getattr(self, 'gat_base_' + str(i))
                emb, feats = branch_emb(g, input)
                feats.append(emb)
                branch_out = getattr(self, 'gat_out_' + str(i))
                y = branch_out(g, emb)
                y = torch.squeeze(y, dim=1)
                y = torch.unsqueeze(y, dim=0)
                output.append(y)
                featss.append(feats)

        for i in range(self.num_branches):
            for j in range(3):
                print(featss[i][j].shape)
            print('*********************************************')

        for i in range(self.num_branches):
            print(output[i].shape)


        output = torch.cat(output,dim=0)
        return output, featss




