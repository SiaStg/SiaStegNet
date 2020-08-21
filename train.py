from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import os.path as osp
import shutil
import random
import time

import torch
import torch.nn as nn
from torch.optim.adamax import Adamax
from torch.optim.adadelta import Adadelta

import src
from src import utils
from src.data import build_train_loader
from src.data import build_val_loader
from src.data import build_otf_train_loader
from src.matlab import matlab_speedy

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-cover-dir', dest='train_cover_dir', type=str, required=True,
    )
    parser.add_argument(
        '--val-cover-dir', dest='val_cover_dir', type=str, required=True,
    )
    parser.add_argument(
        '--train-stego-dir', dest='train_stego_dir', type=str, required=True,
    )
    parser.add_argument(
        '--val-stego-dir', dest='val_stego_dir', type=str, required=True,
    )

    parser.add_argument('--epoch', dest='epoch', type=int, default=500)  # default=1000
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--wd', dest='wd', type=float, default=1e-4)
    parser.add_argument('--eps', dest='eps', type=float, default=1e-8)
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.1)
    parser.add_argument('--margin', dest='margin', type=float, default=1.00)

    parser.add_argument('--random-crop', dest='random_crop', action='store_true')

    parser.add_argument('--random-crop-train', dest='random_crop_train', action='store_true',
                        help='Retrain strategy of SID')

    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=0)

    parser.add_argument('--model', dest='model', type=str, default='kenet')
    parser.add_argument('--finetune', dest='finetune', type=str, default=None)

    parser.add_argument('--gpu-id', dest='gpu_id', type=int, default=0)
    parser.add_argument('--seed', dest='seed', type=int, default=-1)
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=200)
    parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, required=True)
    parser.add_argument('--lr-strategy', dest='lr_str', type=int, default=2,
                        help='1: StepLR, 2:MultiStepLR, 3:ExponentialLR, 4:CosineAnnealingLR, 5:ReduceLROnPlateau')

    args = parser.parse_args()
    return args


def setup(args):
    os.makedirs(args.ckpt_dir, exist_ok=False)

    args.cuda = args.gpu_id >= 0
    if args.gpu_id >= 0:
        torch.cuda.set_device(args.gpu_id)

    log_file = osp.join(args.ckpt_dir, 'log.txt')
    utils.configure_logging(file=log_file, root_handler_type=0)

    utils.set_random_seed(None if args.seed < 0 else args.seed)

    logger.info('Command Line Arguments: {}'.format(str(args)))


args = parse_args()
setup(args)

logger.info('Building data loader')

random_crop_train = args.random_crop_train
if random_crop_train:
    train_loader, epoch_length = build_otf_train_loader(
        args.train_cover_dir, num_workers=args.num_workers
    )
else:
    train_loader, epoch_length = build_train_loader(
        args.train_cover_dir, args.train_stego_dir, batch_size=args.batch_size,
        num_workers=args.num_workers
    )
val_loader = build_val_loader(
    args.val_cover_dir, args.val_stego_dir, batch_size=args.batch_size,
    num_workers=args.num_workers
)
train_loader_iter = iter(train_loader)

logger.info('Building model')
if args.model == 'kenet':
    net = src.models.KeNet()
elif args.model == 'sid':
    net = src.models.SID()
else:
    raise NotImplementedError
if args.finetune is not None:
    net.load_state_dict(torch.load(args.finetune)['state_dict'], strict=False)

criterion_1 = nn.CrossEntropyLoss()
criterion_2 = src.models.ContrastiveLoss(margin=args.margin)

if args.cuda:
    net.cuda()
    criterion_1.cuda()
    criterion_2.cuda()

optimizer = Adamax(net.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.wd)
if args.model == 'sid':
    optimizer = Adadelta(net.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.wd)

lr_str = args.lr_str
if lr_str == 1:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
elif lr_str == 2:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400],
                                                     gamma=0.1)  # milestones=[900,975]
elif lr_str == 3:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
elif lr_str == 4:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
elif lr_str == 5:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.3,
                                                           patience=10, verbose=True, min_lr=0,
                                                           eps=1e-08)
else:
    raise NotImplementedError('Unsupported learning rate strategy')


def preprocess_data(images, labels, random_crop):
    # images of shape: NxCxHxW
    if images.ndim == 5:  # 1xNxCxHxW
        images = images.squeeze(0)
        labels = labels.squeeze(0)
    h, w = images.shape[-2:]

    if random_crop:
        ch = random.randint(h * 3 // 4, h)  # h // 2      #256
        cw = random.randint(w * 3 // 4, w)  # square ch   #256

        h0 = random.randint(0, h - ch)  # 128
        w0 = random.randint(0, w - cw)  # 128
    else:
        ch, cw, h0, w0 = h, w, 0, 0

    if args.model == 'kenet':  
        cw = cw & ~1
        inputs = [
            images[..., h0:h0 + ch, w0:w0 + cw // 2],
            images[..., h0:h0 + ch, w0 + cw // 2:w0 + cw]
        ]
    elif args.model == 'sid':
        inputs = [images[..., h0:h0 + ch, w0:w0 + cw]]

    if args.cuda:
        inputs = [x.cuda() for x in inputs]
        labels = labels.cuda()
    return inputs, labels


def train(epoch):
    net.train()
    running_loss, running_accuracy = 0., 0.

    for batch_idx in range(epoch_length):
        data = next(train_loader_iter)
        inputs, labels = preprocess_data(data['image'], data['label'], args.random_crop)

        optimizer.zero_grad()
        if args.model == 'kenet':  #
            outputs, feats_0, feats_1 = net(*inputs)

            # count parameters start
            # print('parameters_count: {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
            # count parameters end

            loss = criterion_1(outputs, labels) + \
                   args.alpha * criterion_2(feats_0, feats_1, labels)

        elif args.model == 'sid':
            outputs = net(*inputs)
            loss = criterion_1(outputs, labels)

        accuracy = src.models.accuracy(outputs, labels).item()
        running_accuracy += accuracy
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            running_accuracy /= args.log_interval
            running_loss /= args.log_interval
            
            logger.info(
                'Train epoch: {} [{}/{}]\tAccuracy: {:.2f}%\tLoss: {:.6f}'.format(
                    epoch, batch_idx + 1, epoch_length, 100 * running_accuracy,
                    running_loss))
                    
            ###############################log per log_interval start
            is_best=False
            save_checkpoint(
                {
                    'iteration': batch_idx + 1,
                    'state_dict': net.state_dict(),
                    'best_prec1': running_accuracy,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                filename=os.path.join(args.ckpt_dir, 'checkpoint.pth.tar'),
                best_name=os.path.join(args.ckpt_dir, 'model_best.pth.tar'))
            ###############################
            running_loss = 0.
            running_accuracy = 0.
            net.train()


def valid():
    net.eval()
    valid_loss = 0.
    valid_accuracy = 0.
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = preprocess_data(data['image'], data['label'], False)

            if args.model == 'kenet':  #
                outputs, feats_0, feats_1 = net(*inputs)
                valid_loss += criterion_1(outputs, labels).item() + \
                              args.alpha * criterion_2(feats_0, feats_1, labels)
            elif args.model == 'sid':
                outputs = net(*inputs)
                valid_loss += criterion_1(outputs, labels).item()
            valid_accuracy += src.models.accuracy(outputs, labels).item()
    valid_loss /= len(val_loader)
    valid_accuracy /= len(val_loader)
    logger.info('Test set: Loss: {:.4f}, Accuracy: {:.2f}%)'.format(
        valid_loss, 100 * valid_accuracy))
    return valid_loss, valid_accuracy


def save_checkpoint(state, is_best, filename, best_name):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)


_time = time.time()
best_accuracy = 0.
for e in range(1, args.epoch + 1):
    logger.info('Epoch: {}'.format(e))
    logger.info('Train')
    train(e)
    logger.info('Time: {}'.format(time.time() - _time))
    logger.info('Test')
    _, accuracy = valid()
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(accuracy)
    else:
        scheduler.step()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        is_best = True
    else:
        is_best = False
    logger.info('Best accuracy: {}'.format(best_accuracy))
    logger.info('Time: {}'.format(time.time() - _time))
    save_checkpoint(
        {
            'epoch': e,
            'state_dict': net.state_dict(),
            'best_prec1': accuracy,
            'optimizer': optimizer.state_dict(),
        },
        is_best,
        filename=os.path.join(args.ckpt_dir, 'checkpoint.pth.tar'),
        best_name=os.path.join(args.ckpt_dir, 'model_best.pth.tar'))
