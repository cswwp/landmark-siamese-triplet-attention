from backbone.shufflenetv2 import ShuffleNetV2
from data.datasets import BalancedBatchSampler
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from trainer import fit
import numpy as np
import os
import math
import torchvision.models as models
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
from losses import OnlineContrastiveEucLoss, OnlineContrastiveCosLoss, OnlineTripletLoss
from utils import AllPositivePairSelector, HardNegativePairSelector  # Strategies for selecting pairs within a minibatch
from network import init_network
from config import config
from data.datasets import init_data_loader, init_transform, TEST_DATA_LOADER
from utils import HardestNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric
from test import inference_test


def main():
    # print('fsafsdaf:', args.training_dataset, args.arch)
    print(">> Creating directory if it does not exist:\n>> '{}'".format(args.directory))
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    log_dir = os.path.join(args.directory, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    params = {'architecture': args.arch, 'pooling': args.pool}
    n_classes = args.n_classes
    n_samples = args.n_samples
    cuda = args.cuda
    input_size = args.image_size
    transform, transform_te, transform_label = init_transform(input_size)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    online_train_loader, online_test_loader = init_data_loader(args.root, n_classes, n_samples, transform,transform_te, transform_label, kwargs)

    # Set up the network and training parameters
    model = init_network(params)
    parameters = []
    # add feature parameters
    parameters.append({'params': model.features.parameters()})
    if cuda:
        # print('model cuda:', cuda)
        model.cuda()
    pos_margin = 1.0
    neg_margin = 0.3
    # define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

    metrics = [AverageNonzeroTripletsMetric()]
    if args.loss.startswith('OnlineContrastiveEucLoss'):
        loss_fn = OnlineContrastiveEucLoss(pos_margin, neg_margin, HardNegativePairSelector())
    elif args.loss.startswith('OnlineContrastiveCosLoss'):
        loss_fn = OnlineContrastiveCosLoss(args.loss_margin)
    elif args.loss.startswith('OnlineTriplet'):
        loss_fn = OnlineTripletLoss(args.loss_margin, HardestNegativeTripletSelector(args.loss_margin))

    exp_decay = math.exp(-0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model.features, torch.ones([1, 3, 224, 224]).cuda())
    fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler,
        writer, metrics=metrics, args=args)


if __name__ == '__main__':
    args = config()
    directory = "{}".format(args.training_dataset)
    directory += "_{}".format(args.arch)
    directory += "_{}".format(args.pool)
    # directory += "_attention_{}".format(int(args.attention))
    directory += "_{}_m{:.2f}".format(args.loss, args.loss_margin)
    directory += "_{}_lr{:.1e}_wd{:.1e}".format(args.optimizer, args.lr, args.weight_decay)
    directory += "_imsize{}_nclasses{}_nsamples{}".format(args.image_size, args.n_classes, args.n_samples)
    args.directory = os.path.join(args.directory, directory)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
    if args.mode == 'train':
        main()
    else:
        inference_test(args)


