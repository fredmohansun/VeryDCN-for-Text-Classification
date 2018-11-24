import os
import argparse
import time
import io
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from TextDataset import TextDataset
from model import VDCNN

#################################chamring block breaker#############################################
# Setting Environment
## Dictionaries and Lists
concatenater = lambda a,b,c: a+b+c
Datasets = ['MR', 'SST-1', 'SST-2', 'ag_news', 'sogou_news', 'dbpedia','yelp_review_full','yelp_review_polarity','yahoo_answers','amazon_review_full','amazon_review_polarity']
train_x_path = dict(zip(Datasets, list(map(concatenater,['preprocessed_data/']*len(Datasets),Datasets,['_train.txt']*len(Datasets)))))
train_y_path = dict(zip(Datasets, list(map(concatenater,['preprocessed_data/']*len(Datasets),Datasets,['_train_labels.txt']*len(Datasets)))))
test_x_path = dict(zip(Datasets, list(map(concatenater,['preprocessed_data/']*len(Datasets),Datasets,['_test.txt']*len(Datasets)))))
test_y_path = dict(zip(Datasets, list(map(concatenater,['preprocessed_data/']*len(Datasets),Datasets,['_test_labels.txt']*len(Datasets)))))
lr_dict = {'SGD': 0.001, 'ADAM':0.01}
batch_size_dict = {9: 200, 17: 150, 29:120, 49:75}
save_frequency = {9: 5, 17: 5, 29:2, 49:1}

## Parser
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str)
parser.add_argument('--dataset', type=str, choices=Datasets)
parser.add_argument('-n','--batch_size', type=int)
parser.add_argument('-t','--max_epoch', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--SGD', action='store_true')
parser.add_argument('--shortcut', action='store_true')
parser.add_argument('--depth', type=int, choices=[9,17,29,49])
parser.add_argument('--embed_size', type=int)
parser.add_argument('--downsample', type=int, choices=[0,1,2,3], help='0: No, 1(default): maxpool, 2: Resnet, 3: Kmax')
parser.add_argument('--kmaxpool', type=int)
args = parser.parse_args()

## Hyperpara
is_cuda = torch.cuda.is_available()
dataset = 'amazon_review_full' if args.dataset is None else args.dataset
vocab_size = 69
depth = 29 if args.depth is None else args.depth
batch_size = batch_size_dict.get(depth) if args.batch_size is None else args.batch_size
max_epoch = 20 if args.max_epoch is None else args.max_epoch
optimi = 'SGD' if args.SGD else 'ADAM'
lr = lr_dict.get(optim,0.001) if args.lr is None else args.lr
embed_size = 16 if args.embed_size is None else args.embed_size
downsample = 1 if args.downsample is None else args.downsample
kmaxpool = 8 if args.kmaxpool is None else args.kmaxpool
savefile = '_'.join([dataset, str(depth), str(batch_size), str(max_epoch), optimi, ['No downsample', 'MaxPooling', 'ResNet', '{0}-max'.format(kmaxpool)][downsample]])
print(savefile)

## Datasets
train_dataset = TextDataset(train_x_path.get(dataset), train_y_path.get(dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

test_dataset = TextDataset(test_x_path.get(dataset), test_y_path.get(dataset))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
print('Dataset loaded...')

## Model initialization
model = VDCNN(vocab_size, embed_size, depth, downsample, args.shortcut, kmaxpool)
if is_cuda:
    model.cuda()
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
if args.load is not None:
    model.load_state_dict(torch.load(args.load))

## Opitmizer
if optimi == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr = lr)

criterion = nn.CrossEntropyLoss

#################################chamring block breaker#############################################
# Train
basic_message = "Epoch {0}: {1} accuracy: {2:.4f}%, loss: {3:.6f}, time: {4:.2f}seconds"
train_loss = []
train_acc = []
val_loss = []
val_acc = []
print('start training...')
for epoch in range(max_epoch):
    model.train()
    running_acc = 0.0
    running_loss = 0.0
    counter = 0
    timer1 = time.time()

    for inputs, label in train_loader:
        if is_cuda:
            inputs, label = Variable(inputs).cuda(), Variable(label).cuda()
        else:
            inputs, label = Variable(inputs), Variable(label)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, label)
        prediction = outputs.data.max(1)[1]
        running_acc += prediction.eq(label.data).sum().cpu().data.numpy()
        running_loss += loss.data.item()
        counter += batch_size
        loss.backward()
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step'] >=1024):
                    state['step'] = 1000
        optimizer.step()
    train_acc.append(running_acc/counter)
    train_loss.append(running_loss/(counter/batch_size))
    timer2 = time.time()
    print(basic_message.format(epoch, 'training', train_acc[-1]*100.0, train_loss[-1], float(timer2-timer1)))

## "Validation"
    model.eval()
    running_acc = 0.0
    running_loss = 0.0
    counter = 0
    for inputs, label in test_loader:
        if is_cuda:
            inputs, label = Variable(inputs).cuda(), Variable(label).cuda()
        else:
            inputs, label = Variable(inputs), Variable(label)

        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, label)
        prediction = outputs.data.max(1)[1]
        running_acc += prediction.eq(label.data).sum().cpu().data.numpy()
        running_loss += loss.data.item()
        counter += batch_size
    val_acc.append(running_acc/counter)
    val_loss.append(running_loss/(counter/batch_size))
    print(basic_message.format(epoch, '\"validation\"', val_acc[-1]*100.0, val_loss[-1], float(time.time()-timer2)))

## Saving model
    if epoch+1 % save_frequency.get(depth,1) == 0:
        torch.save(model.state_dict(), 'model/'+savefile+'_Epoch_{0}.model'.format(epoch))
#################################chamring block breaker#############################################
# test
model.eval()
running_acc = 0.0
running_loss = 0.0
counter = 0
timer1 = time.time()
for inputs, label in test_loader:
    if is_cuda:
        inputs, label = Variable(inputs).cuda(), Variable(label).cuda()
    else:
        inputs, label = Variable(inputs), Variable(label)

    with torch.no_grad():
        outputs = model(inputs)
    loss = criterion(outputs, label)
    prediction = outputs.data.max(1)[1]
    running_acc += prediction.eq(label.data).sum().cpu().data.numpy()
    running_loss += loss.data.item()
    counter += batch_size
print(basic_message.format('Test', '\"validation\"', running_acc/counter*100.0, running_loss/(counter/batch_size),float(time.time()-timer1)))
torch.save(model.state_dict(), 'model/'+savefile+'_final.model')
np.save('state/' + savefile + 'training_acc.npy', np.array(train_acc))
np.save('state/' + savefile + 'training_loss.npy', np.array(train_loss))
np.save('state/' + savefile + 'val_acc.npy', np.array(val_acc))
np.save('state/' + savefile + 'val_loss.npy', np.array(val_loss))
