import os
import copy
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
# from torch.utils.data import DataLoader
from torch.optim import SGD, Adam

from discriminator import Discriminator
from discriminator import train_gan_cycle
import dgl
from dgl.data.ppi import PPIDataset
from dgl.data.dgl_dataset import DGLBuiltinDataset
from dgl.data import CiteseerGraphDataset, PubmedGraphDataset

import utils
from utils import StepwiseLR, collate
from model import GAT, GCN, GCN_ens, GAT_ens, SAGE_ens
from loss import group_cls_loss, co_distill_loss, dml_loss, kdcl_naive_loss, kdcl_minlogit, mid_codistill_loss, edge_distill_loss

from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    start = time.time()
    writer = SummaryWriter()

    # Data loading code
    if args.data == 'cora':
        dataset, graph, feats, labels, train_mask, test_mask = utils.load_cora_data()
        num_feats = feats.shape[1]
        num_classes = dataset.num_classes
    else:
        dataset, graph, feats, labels, train_mask, val_mask, test_mask = utils.load_data(args)
        num_feats = feats.shape[1]
        num_classes = dataset.num_classes

    # create model
    if args.arch == 'GCN':
        model = GCN_ens(num_layers=args.num_hidden, in_dim=num_feats, hidden_dim=args.hidden_dim,
                        num_branches=args.num_branches,
                        num_classes=num_classes).to(device)
    elif args.arch == 'GAT':
        model = GAT_ens(num_layers=args.num_hidden, in_dim=num_feats, hidden_dim=args.hidden_dim,
                        num_classes=num_classes,
                        num_branches=args.num_branches, heads=args.heads, feat_drop=args.feat_drop,
                        attn_drop=args.attn_drop).to(device)
    elif args.arch == 'SAGE':
        model = SAGE_ens(in_feats=num_feats, n_hidden=args.hidden_dim, n_classes=num_classes, n_layers=args.num_hidden,
                         aggregator_type=args.agg_type, num_branches=args.num_branches, dropout=args.dropout).to(device)
    print(model)

    # Discriminators list
    D_list = []
    for i in range(args.num_branches):
        D_list.append(Discriminator(args.hidden_dim, F.relu, args).to(device))
    print("parameter of Discriminator:")
    print(utils.parameters(D_list[0]))

    # define optimizer
    optimizer_model = Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    optimizer_D_list = []
    for i in range(args.num_branches):
        op = Adam((D_list[i]).parameters(), lr=args.lr)
        optimizer_D_list.append(op)

    lr_scheduler = utils.StepwiseLR(optimizer_model, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    print("***********************************warmup training****************************************")
    best_acc1_warmup = 0.0
    acc_list = []
    for epoch in range(args.warmup_epoch):
        # train mode
        warmup_train(model, graph, lr_scheduler, optimizer_model, feats, labels, train_mask, epoch, args, writer)

        # evaluate mode
        if args.data == 'cora':
            acc1 = evaluate(model, graph, feats, labels, test_mask)
        else:
            acc1 = evaluate(model, graph, feats, labels, val_mask)

        acc_list.append(acc1)
        if acc1 > best_acc1_warmup:
            best_model_warmup = copy.deepcopy(model)
        best_acc1_warmup = max(acc1, best_acc1_warmup)

    best_acc1_okd = 0.0
    best_acc1_okd_epoch = 0
    best_train_acc = 0
    print("***************************online knowledge distillation training***************************")
    for epoch in range(args.okd_epoch):
        acc = okd_train(model, D_list, graph, lr_scheduler, optimizer_model, optimizer_D_list, feats, labels, train_mask,
                  epoch, args, writer)
        best_train_acc = max(acc,best_train_acc)

        # evaluate mode
        if args.data == 'cora':
            acc1 = evaluate(model, graph, feats, labels, test_mask)
        else:
            acc1 = evaluate(model, graph, feats, labels, val_mask)

        acc_list.append(acc1)
        if acc1 > best_acc1_okd:
            best_acc1_okd_epoch = epoch
            best_model_okd = copy.deepcopy(model)
        best_acc1_okd = max(acc1, best_acc1_okd)

    writer.close()

    torch.save(best_model_okd.state_dict(),"./save/{}_{}_okd200.pth".format(args.arch, args.data))

    print("**********************Testing**************************")
    print("train acc1:" + str(best_train_acc))
    print("time:" + str(time.time()-start))

    print("Test:")
    evaluate(best_model_okd, graph, feats, labels, test_mask)

    np.savetxt('acc_okd'+str(int(args.seed))+'.txt', acc_list, fmt="%.2f", delimiter=',')


def warmup_train(model, graph, lr_scheduler, optimizer, feats, labels, train_mask,
                 epoch, args, writer):
    train_acc1 = utils.RunningAverage()
    losses = utils.RunningAverage()


    # train mode
    model.train()
    end = time.time()

    # forward propagation by using all nodes
    graph = graph.to(device)
    feats = feats.to(device)
    labels = labels.to(device)

    logits_list, feats_list = model(graph, feats)
    labels = labels[train_mask]  # n

    logits_list = logits_list[:, train_mask, :]

    # compute loss
    loss = 0.0
    for i in range(args.num_branches):
        cls_loss = F.cross_entropy(logits_list[i], labels)
        loss += cls_loss

    writer.add_scalar('cls_loss', loss, epoch)

    # compute train acc
    acc_max = 0.0
    for i in range(args.num_branches):
        acc = utils.accuracy(logits_list[i], labels)
        acc_max = max(acc[0].item(), acc_max)

    train_acc1.update(acc_max)
    losses.update(loss.item())

    # backward propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_metrics = {
        'train_acc1': train_acc1.value(),
        'loss': losses.value(),
        'time': time.time() - end}

    end = time.time()

    metrics_string = "Epoch [{}/{}]: ".format(epoch, args.warmup_epoch)
    metrics_string += " | ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    print("- Train metrics: " + metrics_string)

    return train_acc1.value()


def okd_train(model, D_list, graph, lr_scheduler, optimizer_model, optimizer_D_list, feats, labels, train_mask,
              epoch, args, writer):
    train_acc1 = utils.RunningAverage()
    losses = utils.RunningAverage()
    d_losses = utils.RunningAverage()
    g_losses = utils.RunningAverage()

    # lr_scheduler.step()

    # train mode
    model.train()
    end = time.time()

    # forward propagation by using all nodes
    graph = graph.to(device)
    feats = feats.to(device)
    labels = labels.to(device)

    logits_list, feats_list = model(graph, feats)

    gan_loss, d_loss_total= train_gan_cycle(D_list, optimizer_D_list, feats_list, graph, args)

    d_losses.update(d_loss_total)
    g_losses.update(gan_loss)

    labels = labels[train_mask]  # n
    logits_list = logits_list[:, train_mask, :]

    for i in range(args.num_branches):
        for j in range(args.num_hidden):
            feats_list[i][j] = feats_list[i][j][train_mask][:]  # br * n_hidden * n * hidden_dim

    # compute loss

    cls_loss_total = group_cls_loss(logits_list, labels, args)
    logit_kd_loss_total = co_distill_loss(logits_list, args)

    node_kd_loss_total = 0.0


    loss = args.gan_trade * gan_loss + args.logitkd_trade * logit_kd_loss_total + cls_loss_total

    writer.add_scalar('gan_loss', gan_loss, epoch)
    writer.add_scalar('d_loss', d_loss_total, epoch)
    writer.add_scalar('logit_kd_loss', logit_kd_loss_total, epoch)
    writer.add_scalar('node_feats_kd_loss', node_kd_loss_total, epoch)
    writer.add_scalar('cls_loss', cls_loss_total, epoch + args.warmup_epoch)

    # compute train acc
    acc_max = 0.0
    for i in range(args.num_branches):
        acc = utils.accuracy(logits_list[i, :, :], labels)
        acc_max = max(acc[0].item(), acc_max)


    train_acc1.update(acc_max)
    losses.update(loss.item())

    # backward propagation
    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()

    train_metrics = {
        'train_acc1': train_acc1.value(),
        'loss': losses.value(),
        'time': time.time() - end}

    end = time.time()

    metrics_string = "Epoch [{}/{}]: ".format(epoch, args.okd_epoch)
    metrics_string += " | ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    print("- Train metrics: " + metrics_string)

    return train_acc1.value()


def evaluate(model, graph, feats, labels, val_mask):
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)
        feats = feats.to(device)
        labels = labels.to(device)

        logits_list, feats_list = model(graph, feats)

        logits_list = logits_list[:, val_mask, :]  # br * n * n_class
        labels = labels[val_mask]  # n

        max_acc1 = 0.0
        val_loss = 10000
        for i in range(args.num_branches):
            acc1 = utils.accuracy(logits_list[i], labels)
            val_loss_i = F.cross_entropy(logits_list[i], labels)
            if acc1[0].item() > max_acc1:
                max_acc1 = acc1[0].item()
                val_loss = val_loss_i

        val_metrics = {
            'val_acc1': max_acc1,
            'val_loss' : val_loss
        }

        metrics_string = " | ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        print("- Test metrics: " + metrics_string)
        return max_acc1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='GCN', help='network architecture')
    parser.add_argument('--data', default='cora', help='dataset')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-nh', '--num_hidden', default=1, type=int, help='num of hidden layers')
    parser.add_argument('--heads', default=[8, 8], type=list, help='attention heads')
    parser.add_argument('-lr', '--lr', default=5e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, help='momentum')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--warmup_epoch', default=0, type=int, help='num of warmup epoch')
    parser.add_argument('--okd_epoch', default=3, type=int, help='num of epochs in online knowledge distillation')
    parser.add_argument('--num_branches', default=4, type=int, help='num of branches in online knowledge distillation')
    parser.add_argument('--feat_drop', default=0.6, type=float, help='dropout rate of each hidden layer')
    parser.add_argument('--attn_drop', default=0.6, type=float, help='drop out rate of graph attention mechanism')
    parser.add_argument('--temp', default=3, type=float, help='temperature used in online knowledge distillation')
    parser.add_argument('--dropout', default=0.5, type=float, help=',dropout rate')
    parser.add_argument('--agg_type', default='mean', help='aggregate type of GraphSAGE')
    parser.add_argument('--hidden_dim', default=16, type=int, help='dim of hidden feats')
    parser.add_argument('--logitkd_trade', default=1, type=float,
                        help='coefficient of logit-based knowledge distillation loss')
    parser.add_argument('--gan_trade', default=1, type=float, help='coefficient of gan loss')
    parser.add_argument('--d_trade', default=1, type=float, help='coefficient of discriminator loss')
    parser.add_argument('--gan_type',default='gan_cycle',help='type of gan')
    parser.add_argument('--seed', default=1, type=int, help='seed of random parameter initialize')
    parser.add_argument('--d_dim',default=16,type=int,help='hidden dim of discriminator')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(device)
    print(args)
    main(args)
