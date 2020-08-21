from __future__ import print_function

import argparse
import random

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn import metrics
import numpy as np

import src
from src import utils
from src.data import build_train_loader
from src.data import build_val_loader

parser = argparse.ArgumentParser(description='Validation of KeNet')
parser.add_argument(
        '--val-cover-dir', dest='val_cover_dir', type=str, required=True,
)
parser.add_argument(
        '--val-stego-dir', dest='val_stego_dir', type=str, required=True,
)
parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)
parser.add_argument('--gpu-id', dest='gpu_id', type=int, default=0)
parser.add_argument('--model', dest='model', type=str, default='kenet')
parser.add_argument('--checkpoint', dest='ckpt', type=str, required=True)
parser.add_argument('--num-workers', dest='num_workers', type=int, default=0)
parser.add_argument('--save-as', dest='img_name', type=str, default='roc.png', required=True)
parser.add_argument('--random-crop', dest='random_crop', action='store_true')

args = parser.parse_args()

print("Generate loaders...")
valid_loader = build_val_loader(
    args.val_cover_dir, args.val_stego_dir, batch_size=args.batch_size,
    num_workers=args.num_workers
)
print("Generate model")

if args.model == 'kenet':
    net = src.models.KeNet()
elif args.model == 'sid':
    net = src.models.SID()
else:
    raise NotImplementedError
    
args.cuda = args.gpu_id >= 0
if args.gpu_id >= 0:
    torch.cuda.set_device(args.gpu_id)
if args.cuda:
    net.cuda()    

if args.ckpt is not None:
    net.load_state_dict(torch.load(args.ckpt)['state_dict'])
    
if args.cuda:
    net.cuda()
    
def preprocess_data(images, labels, random_crop):
    h, w = images.shape[-2:]
    if random_crop:
        ch = random.randint(h * 3 // 4, h)
        cw = random.randint(w * 3 // 4, w)

        h0 = random.randint(0, h - ch)
        w0 = random.randint(0, w - cw)
    else:
    	ch, cw, h0, w0 = h, w, 0, 0

    if args.model == 'kenet':
        cw = cw & ~1
        inputs = [
            images[..., h0:h0+ch, w0:w0+cw//2],
            images[..., h0:h0+ch, w0+cw//2:w0+cw]
        ]
    elif args.model == 'sid':
        inputs = [images[..., h0:h0+ch, w0:w0+cw]]

    if args.cuda:
        inputs = [x.cuda() for x in inputs]
        labels = labels.cuda()
    return inputs, labels

def validation():
    net.eval()

    y_true, y_score = [], []
    accuracy = 0.
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = preprocess_data(data['image'], data['label'], args.random_crop)
            if args.model == 'kenet':
                outputs, feats_0, feats_1 = net(*inputs)
            elif args.model == 'sid':
                outputs = net(*inputs)
        
            accuracy += src.models.accuracy(outputs, labels).item()

            y_true.append(labels.detach())

            probability = torch.nn.functional.softmax(outputs, dim=1)
            predict = probability[:, 1]
            y_score.append(predict.detach())

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_score = torch.cat(y_score, dim=0).cpu().numpy()
    print('accuracy: {}'.format(accuracy / len(valid_loader)))

    return y_true, y_score
    
def alaska_weighted_auc(fpr, tpr):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min <= tpr) & (tpr < y_max)
#        from ipdb import set_trace
#        set_trace()
        val = fpr[mask]
        if val.size == 0:
            st = 0
        else
            st = val[-1]

        x_padding = np.linspace(st, 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization


def plot(y_true, y_score):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)

    roc_auc = metrics.auc(fpr, tpr)*100.
    
    weighted_auc = alaska_weighted_auc(fpr, tpr)*100.
    
    print('AUC: {:.2f}'.format(roc_auc))
    print('Weighted AUC: {:.2f}'.format(weighted_auc))
    
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %2.2f)' % roc_auc)  ###x:FT y:TT
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(args.img_name)
    #plt.show()

y_true, y_score = validation()
plot(y_true, y_score)
