import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import autograd

from dgl.nn.pytorch import edge_softmax, GATConv, GraphConv, SAGEConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Discriminator(nn.Module):
    def __init__(self, in_feats, activation, args):
        super(Discriminator, self).__init__()
        self.arch = args.arch
        if args.arch == 'GCN':
            # self.f1 = GraphConv(in_feats, 32, activation=activation)
            # self.f2 = GraphConv(32, 1, activation=activation)
            self.f = GraphConv(in_feats, 1, activation=activation)
        elif args.arch == 'GAT':
            # self.f1 = GraphConv(in_feats * args.heads[0], 64, activation=activation)
            # self.f2 = GraphConv(64, 1, activation=activation)
            self.f1 = GraphConv(in_feats * args.heads[0], args.d_dim, activation=activation)
            self.f2 = GraphConv(args.d_dim, 1, activation=activation)

        elif args.arch == 'SAGE':
            self.f1 = GraphConv(in_feats, args.d_dim, activation=activation)
            self.f2 = GraphConv(args.d_dim, 1, activation=activation)

            # self.f1 = SAGEConv(in_feats, 16, args.agg_type,activation=activation)
            # self.f2 = SAGEConv(16, 1, args.agg_type,activation=activation)

    def forward(self, g, input):
        if self.arch == 'GAT':
            output = self.f1(g, input)
            output = self.f2(g, output)

        elif self.arch == 'SAGE':
            output = self.f1(g, input)
            output = self.f2(g, output)
        else:
            # output = self.f1(g, input)
            # output = self.f2(g, output)
            output = self.f(g, input)

        return output


def train_gan_cycle(D_list, optimizer_D_list, feats_list, graph, args):
    gan_loss = 0.0
    criterion = nn.BCELoss()
    d_loss_total = 0.0


    for i in range(args.num_branches):

        j = args.num_hidden - 1
        real_labels = torch.ones(feats_list[0][j].shape[0], 1).to(device)
        fake_labels = torch.zeros(feats_list[0][j].shape[0], 1).to(device)

        real_feat_d = feats_list[i][j].detach()
        real_score = D_list[i](graph, real_feat_d)
        m = nn.Sigmoid()
        d_real_loss = criterion(m(real_score), real_labels)

        fake_score = D_list[i](graph, feats_list[(i + args.num_branches - 1) % args.num_branches][j])
        d_fake_loss = criterion(m(fake_score), fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss = d_loss * args.d_trade
        d_loss_total += d_loss

        optimizer_D_list[i].zero_grad()
        d_loss.backward(retain_graph=True)
        optimizer_D_list[i].step()

        output = D_list[i](graph, feats_list[(i + args.num_branches - 1) % args.num_branches][j])
        g_loss = criterion(m(output), real_labels)

        gan_loss += g_loss

    return gan_loss, d_loss_total


