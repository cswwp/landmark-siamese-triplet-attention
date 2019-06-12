import torch
import numpy as np
import os
import math
import shutil
import sys
from torchvision.utils import make_grid
from data.datasets import inv_normalize

device_ids = [0, 1, 2]

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, writer,
        metrics=[], exp_decay=math.exp(-0.01), args=None):
    """
        Loaders, model, loss function and metrics should work together for a given task,
        i.e. The model should be able to process data output of loaders,
        loss function should process target output of loaders and outputs from the model

        Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
        Siamese network: Siamese loader, siamese model, contrastive loss
        Online triplet learning: batch loader, embedding model, online triplet loss
    """
    min_loss = float('inf')
    start_epoch = 0
    parallel_model = model
    if args.cuda:
        parallel_model = torch.nn.DataParallel(model, device_ids=device_ids, dim=0).cuda()  # Encapsulate the model


    if args.resume:
        args.resume = os.path.join(args.directory, args.resume)
        if os.path.isfile(args.resume):
            # load checkpoint weights and update model and optimizer
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            min_loss = checkpoint['min_loss']
            print('min_loss:', min_loss)
            parallel_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # important not to forget scheduler updating
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay, last_epoch=checkpoint['epoch']-1)
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))

    epochs = args.epochs
    for epoch in range(start_epoch, epochs):

        val_loss, metrics = test_epoch(val_loader, parallel_model, loss_fn, metrics, writer, args)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, epochs,
                                                                                 val_loss)

        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        print('minloss:', min_loss, 'current test mean loss:', val_loss)
        print('metric message:', message)

        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': parallel_model.state_dict(),
            'min_loss': min_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.directory)

        scheduler.step()
        print("curr learning rate:", scheduler.get_lr())
        # Train stage
        train_loss, metrics = train_epoch(train_loader, parallel_model, loss_fn, optimizer,
                                                                      metrics, writer, args, epoch)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())



def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    print("epoch model saved")
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)
        print("global best model updated")

def train_epoch(train_loader, model, loss_fn, optimizer, metrics, writer, args, epoch):
    for metric in metrics:
        metric.reset()
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if args.cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
        optimizer.zero_grad()
        outputs = model(*data)
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        [loss_outputs, distance_pos, distance_neg, pairs, len_pairs] = loss_fn(*loss_inputs)
        loss = loss_outputs
        # print("loss: ", loss)
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        for metric in metrics:
            metric(outputs, target, len_pairs)


        if batch_idx % args.log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print('metric message:', message)
            losses = []
            if args.loss.startswith('OnlineContrastive'):
                indice_show = np.random.choice(range(pair[0].shape[0]), args.train_show_pairs)
                writer.add_image('position image',
                    make_grid(inv_normalize(data[0][pair[0][indice_show]], args.train_show_pairs*2)))
                writer.add_image('negative image',
                    make_grid(inv_normalize(data[0][pair[1][indice_show]], args.train_show_pairs*2)))
                if 'cos' in args.loss:
                    writer.add_scalars('data/TRAIN_SIA_COS_DISTANCE', {
                                                    'TRAIN_COS_POS': distance_pos,
                                                    'TRAIN_COS_NEG': distance_neg
                    }, global_step=epoch*len(train_loader) + batch_idx)
                elif 'Euc' in args.loss:
                    writer.add_scalars('data/TRAIN_SIA_EUC_DISTANCE', {
                        'TRAIN_EUC_POS': distance_pos,
                        'TRAIN_EUC_POS': distance_neg
                    }, global_step=epoch*len(train_loader) + batch_idx)

            elif args.loss == 'OnlineTriplet':
                indice_show = np.random.choice(range(pairs.shape[0]), args.train_show_pairs)
                # print('inv_normalize(data[0][pairs[indice_show, 0]]:', data[0][pairs[indice_show, 0]].shape, inv_normalize(data[0][pairs[indice_show, 0]]))
                # x =
                # print("fdsafs:", x.size(), args.train_show_pairs)
                # y = make_grid(x, nrow=args.train_show_pairs, padding=0)
                # print(y.size())
                # a = input()
                writer.add_image('anchors image', make_grid(inv_normalize(data[0][pairs[indice_show, 0]]), nrow=args.train_show_pairs))
                writer.add_image('position image', make_grid(inv_normalize(data[0][pairs[indice_show, 1]]), nrow=args.train_show_pairs))
                writer.add_image('negative image', make_grid(inv_normalize(data[0][pairs[indice_show, 2]]), nrow=args.train_show_pairs))
                writer.add_scalars('data/TRAIN_TRI_EUC_DISTANCE', {
                    'TRAIN_EUC_POS': distance_pos,
                    'TRAIN_EUC_NEG': distance_neg
                }, global_step=epoch*len(train_loader) + batch_idx)

    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, metrics, writer, args):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if args.cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            [loss_outputs, distance_pos, distance_neg, pairs, len_pairs] = loss_fn(*loss_inputs)
            loss = loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, len_pairs)

            if batch_idx % args.log_interval == 0:
                # message = 'Valid: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     batch_idx * len(data[0]), len(val_loader.dataset),
                #     100. * batch_idx / len(val_loader), np.mean(losses))
                # for metric in metrics:
                #     message += '\t{}: {}'.format(metric.name(), metric.value())
                #
                # print('metric message:', message)
                if args.loss.startswith('OnlineContrastive'):
                    # indice_show = np.random.choice(range(pair[0].shape[0]), args.train_show_pairs)
                    # writer.add_images('position image',
                    #                   make_grid(
                    #                       inv_normalize(data[0][pair[0][indice_show]], args.train_show_pairs * 2)))
                    # writer.add_images('negative image',
                    #                   make_grid(
                    #                       inv_normalize(data[0][pair[1][indice_show]], args.train_show_pairs * 2)))
                    if 'cos' in args.loss:
                        writer.add_scalars('data/TEST_SIA_COS_DISTANCE', {
                            'TEST_COS_POS': distance_pos,
                            'TEST_COS_NEG': distance_neg
                        }, batch_idx)
                    elif 'Euc' in args.loss:
                        writer.add_scalars('data/TEST_SIA_EUC_DISTANCE', {
                            'TEST_EUC_POS': distance_pos,
                            'TEST_EUC_NEG': distance_neg
                        }, batch_idx)

                elif args.loss == 'OnlineTriplet':
                    # indice_show = np.random.choice(range(pair.shape[0]), args.train_show_pairs)
                    # writer.add_images('anchors image',
                    #                   make_grid(inv_normalize(data[0][pairs[indice_show, 0]]), args.train_show_pairs))
                    # writer.add_images('position image',
                    #                   make_grid(inv_normalize(data[0][pairs[indice_show, 1]]), args.train_show_pairs))
                    # writer.add_images('negative image',
                    #                   make_grid(inv_normalize(data[0][pairs[indice_show, 2]]), args.train_show_pairs))
                    writer.add_scalars('data/TEST_TRI_EUC_DISTANCE', {
                        'TEST_EUC_POS': distance_pos,
                        'TEST_EUC_NEG': distance_neg
                    }, batch_idx)

    return val_loss, metrics
