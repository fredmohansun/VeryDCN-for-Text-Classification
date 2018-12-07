import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import os
import argparse
import time
import io
import sys

from LSTMModel import RNN_model
concatenater = lambda a,b,c: a+b+c
Datasets = ['amazon_review_full','amazon','ag_news']
train_x_path = dict(zip(Datasets, list(map(concatenater,['preprocessed_lstm_data/']*len(Datasets),Datasets,['_train.txt']*len(Datasets)))))
train_y_path = dict(zip(Datasets, list(map(concatenater,['preprocessed_lstm_data/']*len(Datasets),Datasets,['_train_labels.txt']*len(Datasets)))))
test_x_path = dict(zip(Datasets, list(map(concatenater,['preprocessed_lstm_data/']*len(Datasets),Datasets,['_test.txt']*len(Datasets)))))
test_y_path = dict(zip(Datasets, list(map(concatenater,['preprocessed_lstm_data/']*len(Datasets),Datasets,['_test_labels.txt']*len(Datasets)))))
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=Datasets)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--hidden_units', type=int)
parser.add_argument('--vocab_size', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--sequence_length', type=int)
parser.add_argument('--save_frequency', type=int)
args = parser.parse_args()

vocab_size = 8000 if args.vocab_size is None else args.vocab_size
no_of_hidden_units = 500 if args.hidden_units is None else args.hidden_units
dataset = 'amazon_lstm' if args.dataset is None else args.dataset
num_classes = 5 if args.num_classes is None else args.num_classes
batch_size = 128 if args.batch_size is None else args.batch_size
no_of_epochs = 20 if args.num_epochs is None else args.num_epochs
seq_length = 100 if args.sequence_length is None else args.sequence_length
save_frequency = 3 if args.save_frequency is None else args.save_frequency
savefile = '_'.join([dataset, str(batch_size), str(no_of_epochs)])


x_train = []
with io.open(train_x_path.get(dataset),'r',encoding='utf-8') as f:
    lines = f.readlines()   
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)

y_train = np.zeros((len(x_train),))
with io.open(train_y_path.get(dataset),'r',encoding='utf-8') as f:
    lines = f.readlines()
i = 0 
for line in lines:
    line = line.strip()
    line = int(line)
    y_train[i] = line
    i=i+1

x_test = []
with io.open(test_x_path.get(dataset),'r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)

y_test = np.zeros((len(x_test),))
with io.open(test_y_path.get(dataset),'r',encoding='utf-8') as f:
    lines = f.readlines()
i = 0
for line in lines:
    line = line.strip()
    line = int(line)
    y_test[i] = line
    i = i+1

vocab_size += 1

print ("Dataset loaded...")

model = RNN_model(vocab_size,no_of_hidden_units,num_classes)
model.cuda()
# opt = 'sgd'
# LR = 0.01
opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

L_Y_train = len(y_train)
L_Y_test = len(y_test)
total_step = int(L_Y_train/batch_size)
train_loss = []
train_acc = []
val_loss = []
val_acc = []
print ("start training....")
for epoch in range(no_of_epochs):

    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):
        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = seq_length
        x_input = np.zeros((len(x_input2),sequence_length),dtype=np.int)
        for j in range(len(x_input2)):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        y_input = y_train[I_permutation[i:i+batch_size]]

        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.LongTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data,target,train=True)  
        loss.backward()
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step'] >=1024):
                    state['step'] = 1000
        optimizer.step() 
        
        prediction = pred.data.max(1)[1]
        acc = prediction.eq(target.data).sum()   #changed for pytorch 0.3.0
        epoch_acc += acc
        epoch_loss += loss.data[0]      #changed for pytorch 0.3.0
        epoch_counter += batch_size
        if (epoch_counter) % (batch_size*100) == 0:
           print(epoch_counter, epoch_acc/epoch_counter, epoch_loss/(epoch_counter/batch_size), "%.4f" % float(time.time()-time1), "Step [{}/{}]".format(epoch_counter/batch_size, total_step))
    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)


    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

    if ((epoch+1)%1==0):
        # ## test
        model.eval()
        for param in model.parameters():    #changed for pytorch 0.3.0
            param.requires_grad = False

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()
        
        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):

            x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
            sequence_length = seq_length
            x_input = np.zeros((len(x_input2),sequence_length),dtype=np.int)
            for j in range(len(x_input2)):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl < sequence_length):
                    x_input[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j,:] = x[start_index:(start_index+sequence_length)]
            y_input = y_test[I_permutation[i:i+batch_size]]

            data = Variable(torch.LongTensor(x_input)).cuda()
            target = Variable(torch.LongTensor(y_input)).cuda()

            loss, pred = model(data,target,train=False)     #changed for pytorch 0.3.0
            
            prediction = pred.data.max(1)[1]
            acc = prediction.eq(target.data).sum()   #changed for pytorch 0.3.0epoch_acc += acc
            epoch_acc += acc
            epoch_loss += loss.data[0]          #changed for pytorch 0.3.0
            epoch_counter += batch_size

        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)

        val_acc.append(epoch_acc)
        val_loss.append(epoch_loss/(epoch_counter/batch_size))
        time2 = time.time()
        time_elapsed = time2 - time1

        print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)
        for param in model.parameters():    #changed for pytorch 0.3.0
            param.requires_grad = True
    if ((epoch+1)%save_frequency == 0):
        print ("saving model...")
        torch.save(model.state_dict(), 'model/'+'LSTM_'+savefile+'_Epoch_{0}.model'.format(epoch))
        np.save('state/' + 'LSTM_'+ savefile + 'training_acc.npy', np.array(train_acc))
        np.save('state/' + 'LSTM_'+ savefile + 'training_loss.npy', np.array(train_loss))
        np.save('state/' + 'LSTM_'+ savefile + 'val_acc.npy', np.array(val_acc))
        np.save('state/' + 'LSTM_'+ savefile + 'val_loss.npy', np.array(val_loss))


torch.save(model,'model/'+'LSTM_'+savefile+'.model')
np.save('state/' + 'LSTM_'+ savefile + 'training_acc.npy', np.array(train_acc))
np.save('state/' + 'LSTM_'+ savefile + 'training_loss.npy', np.array(train_loss))
np.save('state/' + 'LSTM_'+ savefile + 'val_acc.npy', np.array(val_acc))
np.save('state/' + 'LSTM_'+ savefile + 'val_loss.npy', np.array(val_loss))
