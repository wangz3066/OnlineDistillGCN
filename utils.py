import torch
import dgl
import numpy as np

from typing import Optional
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score

from dgl.data import citation_graph as citegrh
import networkx as nx
from dgl import DGLGraph
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax

from dgl.data import CiteseerGraphDataset, PubmedGraphDataset
from dgl.data.ppi import PPIDataset


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:
    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},
    where `i` is the iteration steps.
    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """

    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


def f1_evaluate(output, target):
    predict = np.where(output.data.cpu().numpy() >= 0, 1, 0)
    score = f1_score(target.data.cpu().numpy(),
                         predict, average='micro')

    return score


def kd_loss(logits_s, logits_t, temp=3.0):
    """
    logits_s: pre-softmax or sigmoid activation output of student
    logits_t: pre-softmax or sigmoid activation output of teacher
    """
    soft_logits_s = F.log_softmax(logits_s / temp, dim=1)
    soft_logits_t = F.softmax(logits_t / temp, dim=1)
    loss = temp * temp * F.kl_div(soft_logits_s, soft_logits_t) / logits_s.shape[0]
    return loss

def generate_local_structure_preserving(feats, graph):
    graph = graph.local_var()
    feats = feats.view(-1, 1, feats.shape[1])
    graph.ndata.update({'ftl': feats, 'ftr': feats})
    # compute edge distance

    # gaussion
    graph.apply_edges(fn.u_sub_v('ftl', 'ftr', 'diff'))
    e = graph.edata.pop('diff')
    e = torch.exp((-1.0 / 100) * torch.sum(torch.abs(e), dim=-1))

    # compute softmax
    e = edge_softmax(graph, e)
    return e

def graph_KLDiv(graph, edgex, edgey, reduce='mean'):
    '''
    compute the KL loss for each edges set, used after edge_softmax
    '''
    with graph.local_scope():
        nnode = graph.number_of_nodes()
        graph.ndata.update({'kldiv': torch.ones(nnode,1).to(edgex.device)})
        diff = edgey*(torch.log(edgey)-torch.log(edgex))
        graph.edata.update({'diff':diff})
        graph.update_all(fn.u_mul_e('kldiv', 'diff', 'm'),
                            fn.sum('m', 'kldiv'))
        if reduce == "mean":
            return torch.mean(torch.flatten(graph.ndata['kldiv']))

def local_structure_preserving_distill_loss(feats_s, feats_t, graph):
    '''
        Generate the local structure preserving structures and distill.
    :param feats_s: mid feature of student
    :param feats_t: mid feature of teacher
    '''
    lsp_s = generate_local_structure_preserving(feats_s,graph)
    lsp_t = generate_local_structure_preserving(feats_t,graph)
    graph_kl = graph_KLDiv(graph, lsp_t, lsp_s)
    return graph_kl

def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)


    return data, g, features, labels, train_mask, test_mask

def load_cora_data_with_remove(args):
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)

    g = DGLGraph(data.graph)
    remove_edges(g, p=args.remove_rate)
    g = dgl.add_self_loop(g)

    return data, g, features, labels, train_mask, test_mask

def remove_edges(graph, p=0.0):
    num_edges = graph.num_edges()
    r = np.random.rand(num_edges)
    edge_id = np.array([i for i in range(num_edges)])
    remove = np.where(r<=p, True, False)
    edge_id_remove = edge_id[remove]
    graph.remove_edges(torch.tensor(edge_id_remove))


def load_data(args):
    if args.data == 'pubmed':
        dataset = PubmedGraphDataset()
    elif args.data == 'citeceer':
        dataset = CiteseerGraphDataset()

    graph = dataset[0]

    # get split masks
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    # get node features
    feats = graph.ndata['feat']
    # get labels
    labels = graph.ndata['label']

    return dataset, graph, feats, labels, train_mask, val_mask, test_mask

def load_data_with_remove(args):
    if args.data == 'pubmed':
        dataset = PubmedGraphDataset()
    elif args.data == 'citeceer':
        dataset = CiteseerGraphDataset()

    graph = dataset[0]

    # get split masks
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    # get node features
    feats = graph.ndata['feat']
    # get labels
    labels = graph.ndata['label']

    remove_edges(graph, p=args.remove_rate)
    graph = dgl.add_self_loop(graph)

    return dataset, graph, feats, labels, train_mask, val_mask, test_mask

def gen_edge_feat(feats,graph):
    graph = graph.local_var()
    graph.ndata.update({'ftl': feats, 'ftr': feats})

    graph.apply_edges(fn.u_sub_v('ftl', 'ftr', 'diff'))
    e = graph.edata.pop('diff')
    e = torch.norm(e, dim=1)

    e = F.softmax(e, dim=0)

    return e

def edge_dist(e_s, e_t, type):
    if type == 'square':
        dist = 1/2*torch.square(e_s-e_t).sum()
    elif type == 'norm1':
        dist = torch.sum(torch.abs(e_s-e_t))
    elif type == 'norm2':
        dist = torch.norm(e_s-e_t)
    elif type == 'kl':
        e_s * (torch.log(e_s)-torch.log(e_t))

    return dist

def edge_distill_loss(feats_s, feats_t, graph):
    e_s = gen_edge_feat(feats_s, graph)
    e_t = gen_edge_feat(feats_t, graph)
    loss = edge_dist(e_s,e_t,'kl')

    return loss

def load_ppi():
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')

    return train_dataset, valid_dataset, test_dataset

def parameters(model):
    num_params = 0
    for params in model.parameters():
        curn = 1
        for size in params.data.shape:
            curn *= size
        num_params += curn
    return num_params

