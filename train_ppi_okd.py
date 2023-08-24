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

import dgl
from dgl.data.ppi import PPIDataset
from dgl.data.dgl_dataset import DGLBuiltinDataset
from dgl.dataloading import GraphDataLoader

from tensorboardX import SummaryWriter

import utils
from utils import StepwiseLR, collate
from model import GAT, GCN, GCN_ens, GAT_ens, SAGE_ens

from discriminator import Discriminator, train_gan_cycle
from loss import co_distill_loss, mid_codistill_loss, \
    edge_distill_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    writer = SummaryWriter()

    # Data loading code
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')
    train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size)
    valid_dataloader = GraphDataLoader(valid_dataset, batch_size=args.batch_size)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=args.batch_size)

    num_feats = train_dataset[0].ndata['feat'].shape[1]
    num_classes = train_dataset[0].ndata['label'].shape[1]

    # create model
    if args.arch == 'GCN':
        model = GCN_ens(num_layers=args.num_hidden, in_dim=num_feats, hidden_dim=args.hidden_dim,
                        num_classes=num_classes, num_branches=args.num_branches).to(device)
    elif args.arch == 'GAT':
        model = GAT_ens(num_layers=args.num_hidden, in_dim=num_feats, hidden_dim=args.hidden_dim,
                        num_classes=num_classes, activation=F.elu,
                        num_branches=args.num_branches, heads=args.heads, feat_drop=args.feat_drop,
                        attn_drop=args.attn_drop, residual=args.residual).to(device)
    elif args.arch == 'SAGE':
        model = SAGE_ens(in_feats=num_feats, n_hidden=args.hidden_dim, n_classes=num_classes, n_layers=args.num_hidden,
                         activation=F.relu, num_branches=args.num_branches).to(device)
    print(model)
    model_parameter = utils.parameters(model)

    # Discriminators list
    d_parameter = 0
    D_list = []
    for i in range(args.num_branches):
        D = Discriminator(args.hidden_dim, F.relu, args).to(device)
        D_list.append(D)

    print("Discriminator parameter:")
    print(utils.parameters(D_list[0]))

    print('total_parameters:')
    print(model_parameter + d_parameter)


    # define optimizer
    optimizer_warmup = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_okd=Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    optimizer_D_list = []
    for i in range(args.num_branches):
        op = Adam((D_list[i]).parameters(), lr=args.lr)
        optimizer_D_list.append(op)

    f1_list = []
    print('****************************Warmup training*******************************')
    end = time.time()
    for epoch in range(args.warmup_epoch):
        # train mode
        warmup_train(train_dataloader, model, optimizer_warmup, epoch, args, writer)

        # evaluate mode
        f1 = evaluate(valid_dataloader, model, args)
        f1_list.append(f1)

    print('**********************************OKD training**************************')
    best_f1_okd = 0.0
    best_okd_epoch = 0
    for epoch in range(args.okd_epoch):
        # train mode
        okd_train(train_dataloader, model, D_list, optimizer_okd, optimizer_D_list, epoch,
                  writer, args)

        # evaluate mode
        f1 = evaluate(valid_dataloader, model, args)
        f1_list.append(f1)

        if f1 > best_f1_okd:
            best_model_okd = copy.deepcopy(model)
            best_okd_epoch = epoch
        best_f1_okd = max(f1, best_f1_okd)

    print("*******************************Testing**************************")
    torch.save(best_model_okd, './save/' + args.arch + '_PPI_okd_alpha_0.pth')
    total_time = time.time() - end
    print("total time:" + str(total_time))

    print("best okd epoch:")
    print(best_okd_epoch)
    print("okd:")
    evaluate(test_dataloader, best_model_okd, args)


def warmup_train(train_dataloader, model, optimizer, epoch, args, writer):
    train_f1_stu = [utils.RunningAverage() for i in range(args.num_branches)]
    train_f1 = 0.0
    losses = utils.RunningAverage()

    # train mode
    model.train()
    end = time.time()
    for i, graph in enumerate(train_dataloader):
        graph = graph.to(device)
        x = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)
        # forward propagation by using all nodes
        logits_list, feats_list = model(graph, x)

        loss = 0
        for i in range(args.num_branches):
            loss += F.binary_cross_entropy_with_logits(logits_list[i], labels)

        # compute train f1 score
        for i in range(args.num_branches):
            f1 = utils.f1_evaluate(logits_list[i], labels)
            train_f1_stu[i].update(f1)

        losses.update(loss.item())

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_metrics = {
        'train_f1': train_f1,
        'loss': losses.value(),
        'time': time.time() - end}

    metrics_string = "Epoch [{}/{}]: ".format(epoch, args.warmup_epoch)
    metrics_string += " | ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    print("- Train metrics: " + metrics_string)


def okd_train(train_dataloader, model, D_list, optimizer, optimizer_D_list, epoch, writer, args):
    train_f1_stu = [utils.RunningAverage() for i in range(args.num_branches)]
    train_f1 = 0.0
    losses = utils.RunningAverage()
    cls_losses = utils.RunningAverage()
    logit_losses = utils.RunningAverage()


    # train mode
    model.train()
    end = time.time()
    for i, graph in enumerate(train_dataloader):
        x = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)
        graph = graph.to(device)
        # forward propagation by using all nodes
        logits_list, feats_list = model(graph, x)

        # Adversarial distillation
        if args.gan_trade != 0:
            gan_loss, d_loss_total = train_gan_cycle(D_list, optimizer_D_list, feats_list, graph, args)
        else:
            gan_loss = 0
            d_loss_total = 0

        # compute loss
        cls_loss_total = 0.0
        for i in range(args.num_branches):
            cls_loss_total += F.binary_cross_entropy_with_logits(logits_list[i], labels)

        logit_kd_loss_total = co_distill_loss(logits_list, args)


        if args.alpha == 0:
            loss = cls_loss_total + args.logit_kd_trade * logit_kd_loss_total + args.gan_trade * gan_loss
        else:
            wc = (args.okd_epoch - epoch) / args.okd_epoch
            wo = epoch / args.okd_epoch
            loss = cls_loss_total * wc + wo * (
                    args.logit_kd_trade * logit_kd_loss_total + args.gan_trade * gan_loss)

        # compute train f1
        for i in range(args.num_branches):
            f1 = utils.f1_evaluate(logits_list[i], labels)
            train_f1_stu[i].update(f1)
    
        losses.update(loss.item())
        cls_losses.update(cls_loss_total)
        logit_losses.update(logit_kd_loss_total)

        # backward propagation
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    for i in range(args.num_branches):
        train_f1 = max(train_f1, train_f1_stu[i].value())

    train_metrics = {
        'cls_loss': cls_losses.value(),
        'train_f1': train_f1,
        'loss': losses.value(),
        'time': time.time() - end}

    metrics_string = "Epoch [{}/{}]: ".format(epoch, args.okd_epoch)
    metrics_string += " | ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    print("- Train metrics: " + metrics_string)


def evaluate(valid_dataloader, model, args):
    val_f1_stu = [utils.RunningAverage() for i in range(args.num_branches)]
    val_f1 = 0.0
    model.eval()
    with torch.no_grad():
        for i, graph in enumerate(valid_dataloader):
            x = graph.ndata['feat'].to(device)
            labels = graph.ndata['label'].to(device)
            graph = graph.to(device)

            logits_list, _ = model(graph, x)
            for i in range(args.num_branches):
                f1 = utils.f1_evaluate(logits_list[i], labels)
                val_f1_stu[i].update(f1)

        if args.eval == 'best':
            for i in range(args.num_branches):
                val_f1 = max(val_f1, val_f1_stu[i].value())
        else:
            val_f1 = 0
            for i in range(args.num_branches):
                val_f1 += val_f1_stu[i].value()
            val_f1 /= args.num_branches

        val_metrics = {
            'val_f1': val_f1,
        }
        metrics_string = " | ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        print("- Test metrics: " + metrics_string)
        return val_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='GAT', help='network architecture')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size per iteration')
    parser.add_argument('-nh', '--num_hidden', default=4, type=int, help='num of hidden layers')
    parser.add_argument('--heads', default=[2, 2, 2, 2, 2], type=list, help='attention heads')
    parser.add_argument('--hidden_dim', default=64, type=int, help='dim of hidden layers')
    parser.add_argument('-lr', '--lr', default=5e-3, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', default=0, help='weight decay')
    parser.add_argument('--warmup_epoch', default=1, type=int, help='warmup epoch')
    parser.add_argument('--okd_epoch', default=10, type=int, help='online knowledge distillation epoch')
    parser.add_argument('--residual', default=True, help='whether using skip connection')
    parser.add_argument('--feat_drop', default=0, help='dropout rate of each hidden layer')
    parser.add_argument('--attn_drop', default=0, help='drop out rate of graph attention mechanism')
    parser.add_argument('--num_branches', default=4, type=int, help='num of branch networks')
    parser.add_argument('--temp', default=3.0, type=float, help='temperature used in kd')
    parser.add_argument('--gan_type', default='gan_cycle', help='type of gan')
    parser.add_argument('--logit_kd_trade', default=1, type=float, help='coefficient of logit-level distillation loss')
    parser.add_argument('--gan_trade', default=0, type=float, help='coefficient of generator loss')
    parser.add_argument('--d_trade', default=0, type=float, help='coefficient of discriminator loss')
    parser.add_argument('--d_dim', default=64, type=int, help='dimension of discriminator')
    parser.add_argument('--seed', default=1, type=int, help='random seed to initiate network parameters')
    parser.add_argument('--alpha', default=0, type=float, help='okd lr decay rate')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--eval', default='best', type=str, help='evaluation mode')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    print(args)


    main(args)
