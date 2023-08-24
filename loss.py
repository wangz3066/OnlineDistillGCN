import torch
import torch.nn.functional as F
import numpy as np

import utils
import dgl.function as fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def group_cls_loss(logit_list, labels, args):
    loss = 0.0
    for i in range(args.num_branches):
        loss += F.cross_entropy(logit_list[i], labels)
    return loss


def dml_loss(logit_list, args):
    loss = 0.0
    for i in range(args.num_branches):
        for j in range(args.num_branches):
            if i != j:
                loss += utils.kd_loss(logit_list[i], logit_list[j], args.temp)

    loss /= args.num_branches - 1
    return loss


def co_distill_loss(logit_list, args):
    loss = 0.0
    for i in range(args.num_branches):
        logit_t = torch.zeros(logit_list[i].shape).to(device)
        for j in range(args.num_branches):
            if i != j:
                logit_t += logit_list[j]
        logit_t /= args.num_branches - 1
        logit_t = logit_t.detach()

        loss += utils.kd_loss(logit_list[i], logit_t, args.temp)

    return loss


def kdcl_naive_loss(logit_list, labels, args):
    loss_min = float('inf')
    for i in range(args.num_branches):
        loss = F.cross_entropy(logit_list[i], labels)
        if loss < loss_min:
            logit_t = logit_list[i]
            idx_t = i
            loss_min = loss

    logit_t = logit_t.detach().to(device)
    loss = 0.0
    for i in range(args.num_branches):
        if i != idx_t:
            loss += utils.kd_loss(logit_list[i], logit_t, args.temp)

    return loss


def kdcl_minlogit(logit_list, labels, args):
    # logit_list : n_branches * n_sample * n_class
    # labels : n_sample * n_class
    z_c = torch.zeros(logit_list.shape)
    for i in range(args.num_branches):
        for j in range(logit_list.shape[1]):
            z_c[i, j, :] = logit_list[i, j, :] - logit_list[i, j, labels[j]]
    z_t, _ = torch.min(z_c, dim=0)
    z_t = z_t.detach()

    loss = 0.0
    for i in range(args.num_branches):
        loss += utils.kd_loss(logit_list[i], z_t, args.temp)

    return loss


def mid_codistill_loss(feats_list, args):
    # feats_list: num_branches * num_hidden * n_sample * hidden_dim
    loss = 0.0
    for i in range(args.num_branches):
        j = args.num_hidden - 1
        feats_t = torch.zeros(feats_list[0][j].shape).to(device)
        for k in range(args.num_branches):
            if i != k:
                feats_t += feats_list[k][j]
        feats_t /= args.num_branches - 1

        if args.emb_type == 'mse':
            loss += F.mse_loss(feats_list[i][j],feats_t)
        elif args.emb_type == 'kl':
            loss += utils.kd_loss(feats_list[i][j], feats_t, args.temp)

    return loss

def gen_edge_feat(feats, graph, args):
    graph = graph.local_var()
    graph.ndata.update({'ftl': feats, 'ftr': feats})

    if args.edge_type == 'norm2':
        graph.apply_edges(fn.u_sub_v('ftl', 'ftr', 'diff'))
        e = graph.edata.pop('diff')
        e = torch.norm(e, dim=1)
    elif args.edge_type == 'cos':
        graph.apply_edges(fn.u_dot_v('ftl','ftr','diff'))
        # print(graph.edata['diff'].shape)
        # graph.update_all(fn.copy_u('ftl','le'))
        # print(graph.edata['le'].shape)
        # # graph.update_all(copy_d)
        graph.apply_edges(cos_udf)
        e = graph.edata.pop('diff')
        e = torch.sum(e, dim=1)
    elif args.edge_type == 'sigmoid':
        graph.apply_edges(fn.u_sub_v('ftl', 'ftr', 'diff'))
        e = graph.edata.pop('diff')
        e = torch.norm(e, dim=1)
        e = torch.sigmoid(e)
    elif args.edge_type == 'norm1':
        graph.apply_edges(fn.u_sub_v('ftl', 'ftr', 'diff'))
        e = graph.edata.pop('diff')
        e = torch.norm(e, dim=1, p=1)
    return e

def copy_d(edges):
    return {'re': edges.dst['ftr']}

def cos_udf(edges):
    x = torch.norm(edges.src['ftl'])
    y = torch.norm(edges.src['ftr'])
    xy = x.mul(y)
    print(xy.shape)
    sim = torch.mul(edges.data['diff'],xy)
    return {'diff':1- sim}

def edge_dist(e_s, e_t):
    e_s = F.log_softmax(e_s)
    e_t = F.softmax(e_t)
    dist = F.kl_div(e_s, e_t)

    return dist

def edge_distill_mutual(feats_s, feats_t, graph, args):
    e_s = gen_edge_feat(feats_s, graph, args)
    e_t = gen_edge_feat(feats_t, graph, args)
    loss = edge_dist(e_s,e_t)

    return loss

def edge_distill_loss(feats_list, graph, args):
    loss = 0.0
    for i in range(args.num_branches):
        h = args.num_hidden-1
        feats_t = torch.zeros(feats_list[i][h].shape).to(device)
        for j in range(args.num_branches):
            if i != j:
                feats_t += feats_list[j][h]
        feats_t /= args.num_branches - 1
        feats_t = feats_t.detach()

        loss += edge_distill_mutual(feats_list[i][h], feats_t, graph, args)

    return loss


