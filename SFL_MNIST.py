rounds = 50
local_epochs = 1  # This has set to be 1 in the SFLG
users = 5  # number of clients 50,

lr = 0.001

group_client_ordered = True  # True means clients are grouped in order -- False means random selection
no_of_groups = users  # no. of groups/models in the process --- each group has several clients train sequent

import os
import h5py

import struct
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
from tqdm import tqdm
import numpy as np

import pandas
from pandas import DataFrame

from collections import defaultdict
import copy

import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler

matplotlib.rcParams['figure.facecolor'] = '#ffffff'

root_path = '/content'
model_path = '/content'
acc_path = '/content'
server_model_path = '/content'
client_model_path = '/content'

dataset = MNIST(root=root_path, download=True, transform=ToTensor())

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print(len(dataset))
print(dataset)
image, label = dataset[0]
print(label)
image, label = dataset[2]
# plt.imshow(image.permute(1,2,0), cmap='gray')

plt.imshow(image.reshape(28, 28), cmap="gray")

print('Label:', label)


def split_indices(n, val_pct):
    # determining the size of the validation data
    n_val = int(n * val_pct)

    # creating the random permutaio of n numbers

    idxs = np.random.permutation(n)

    # we wil use the first n_val indices for our validation set

    return idxs[n_val:], idxs[: n_val]


train_indices, val_indices = split_indices(len(dataset), 0.2)

# alternative syntax
# train_indices , val_indices = split_indices(len(dataset) ,  val_pct = 0.2 )

print(len(train_indices), " ", len(val_indices))

print(train_indices)

indices_per_client = int(len(train_indices) / users)

print(indices_per_client)

batch_size = 100

train_loaders = []

train_datasets = []

for i in range(users):
    start = indices_per_client * i
    end = indices_per_client * (i + 1)

    # print((train_indices[start:end]))

    train_datasets.append(list(train_indices[start:end]))

    # print(len(train_dataset))

    train_sampler = SubsetRandomSampler(train_indices[start: end])
    # train_loaders= DataLoader(dataset, batch_size, sampler= train_sampler)
    train_loaders.append(DataLoader(dataset, batch_size=batch_size, sampler=train_sampler))

    # print(start, end)

# print(len(train_loaders))

# print(len(train_dataset[0]))
# print(train_dataset[1])


val_sampler = SubsetRandomSampler(val_indices)
# valid_dl in MNIST
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

group_client_collect_dict = defaultdict(list)
# ordered division of clients in the group: group1 -- client0 to clientx
if group_client_ordered:
    # initialize
    start = 0
    step = int(users / no_of_groups)

    diff = users - step * no_of_groups
    # print(diff)

    if diff != 0:
        end = step + 1
        diff -= 1
    else:
        end = step

    for n in range(no_of_groups):

        clients_list = [i for i in range(start, end)]
        group_client_collect_dict[n] = clients_list

        # print("Start, end:",start, end)

        if end + step <= users:
            start = copy.deepcopy(end)
            if diff != 0:
                end += step + 1
                diff -= 1
            else:
                end += step

        else:
            start = copy.deepcopy(end)
            end = users

else:
    print("program is currently in group_client_ordered mode!")

print(group_client_collect_dict)

# checking the batch size and dimension

x_train, y_train = next(iter(train_loaders[0]))
print(x_train.size())
print(y_train.size())

# checking sum of all batch

# 24000 datasize per client, when client=2, batchsize=100, total batches = 24000/100=240
# testing datast = 12000, batchsize=100 , total testing batches =12000/100 =120


train_total_batch = len(train_loaders[0])
print(train_total_batch)
test_batch = len(test_loader)
print(test_batch)

# Dataset size for the serverside -- group wise averaging

datasetsize_server = []
len_dataset = 0
for g in range(no_of_groups):
    for idx in group_client_collect_dict[g]:
        len_dataset += len(train_datasets[idx])
    datasetsize_server.append(len_dataset)
    len_dataset = 0

# Dataset size for the client-size -- client wise averaging
datasetsize_client = []
for i in range(users):
    datasetsize_client.append(len(train_datasets[i]))

print(datasetsize_client)


class MnistNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)

        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.linear4 = nn.Linear(hidden_size, hidden_size)

        # out layer
        self.linear2 = nn.Linear(hidden_size, out_size)

    def forward(self, xb):
        # flatten the tensors to size 100x784
        xb = xb.view(xb.size(0), -1)
        # get intermediate outputs from the hidden layer

        out = self.linear1(xb)

        # apply activation function ReLU

        out = F.relu(out)

        out = self.linear3(out)

        out = F.relu(out)

        out = self.linear4(out)
        out = F.relu(out)

        # get the prediction using the output layer

        out = self.linear2(out)

        return out


input_size = 784
num_classes = 10

# model in MNIST
mnist_net = MnistNet(input_size, hidden_size=32, out_size=num_classes)

mnist_net.to(device)


class MnistClient(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)

        self.linear3 = nn.Linear(hidden_size, hidden_size)

        # self.linear4 = nn.Linear(hidden_size, hidden_size)

        # #out layer
        # self.linear2 = nn.Linear(hidden_size, out_size)

    def forward(self, xb):
        # flatten the tensors to size 100x784
        xb = xb.view(xb.size(0), -1)
        # get intermediate outputs from the hidden layer

        out = self.linear1(xb)

        # apply activation function ReLU

        out = F.relu(out)

        out = self.linear3(out)

        out = F.relu(out)

        # out = self.linear4(out)
        # out = F.relu(out)

        # # get the prediction using the output layer

        # out = self.linear2(out)

        return out

input_size = 784
num_classes = 10


mnist_client = MnistClient(input_size, hidden_size = 32, out_size = num_classes )
mnist_client.to(device)


class MnistServer(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        # self.linear1 = nn.Linear(in_size, hidden_size)

        # self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.linear4 = nn.Linear(hidden_size, hidden_size)

        # out layer
        self.linear2 = nn.Linear(hidden_size, out_size)

    def forward(self, xb):
        # flatten the tensors to size 100x784
        # xb = xb.view(xb.size(0),-1)
        # # get intermediate outputs from the hidden layer

        # out = self.linear1(xb)

        # # apply activation function ReLU

        # out = F.relu(out)

        # out = self.linear3(out)

        # out = F.relu(out)

        out = self.linear4(xb)
        out = F.relu(out)

        # get the prediction using the output layer

        out = self.linear2(out)

        return out

input_size = 784
num_classes = 10



mnist_server= MnistServer(input_size, hidden_size = 32, out_size = num_classes)
mnist_server.to(device)

clientsoclist = []
train_total_batch = []

criterion = nn.CrossEntropyLoss()
train_acc = []
val_acc = []
train_lo = []
val_lo = []
train_acc_average=[]

optimizer = Adam(mnist_net.parameters(), lr=lr)
optimizer_client = Adam(mnist_client.parameters(), lr=lr)
optimizer_server = Adam(mnist_server.parameters(), lr=lr)

weights_server_list_group = []
weights_client_list_group = []

total_sendsize_list = []
total_receivesize_list = []

client_sendsize_list = [[] for i in range(users)]
client_receivesize_list = [[] for i in range(users)]

train_sendsize_list = []
train_receivesize_list = []

import copy


def average_weights(w, datasize):
    """
    Returns the average of the weights.
    """

    for i, data in enumerate(datasize):
        for key in w[i].keys():
            w[i][key] *= float(data)

    w_avg = copy.deepcopy(w[0])

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

    return w_avg

start_time = time.time()    # store start time
print("timmer start!")

group_client_weights_collect = []

# global weights at first
global_client_weights = copy.deepcopy(mnist_client.state_dict())
global_server_weights = copy.deepcopy(mnist_server.state_dict())

for r in range(rounds):
    # initialize the weights of the model for this round
    round_client_weights = copy.deepcopy(global_client_weights)

    round_server_weights = copy.deepcopy(global_server_weights)
    # we will run by groups ----
    for g in range(no_of_groups):
        # for each user in the group g
        # group_client_weights = copy.deepcopy(round_client_weights)
        group_server_weights = copy.deepcopy(round_server_weights)

        for u in group_client_collect_dict[g]:
            # model for user u
            # print("User:", u)

            # client has the same initial state for the group
            mnist_client.load_state_dict(round_client_weights)

            # server model state changes with clients
            mnist_server.load_state_dict(group_server_weights)

            for local_epoch in range(local_epochs):

                for data in tqdm(train_loaders[u], ncols=100,
                                 desc='Round' + str(r + 1) + ': User' + str(u) + '_' + str(local_epoch + 1)):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.clone().detach().long().to(device)

                    # ------------ client forward training -----------
                    optimizer_client.zero_grad()
                    output_client = mnist_client(inputs)
                    client_output = output_client.clone().detach().requires_grad_(True)

                    # ---------- server forward/backward training ------------
                    optimizer_server.zero_grad()
                    output = mnist_server(client_output)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer_server.step()
                    client_grad = client_output.grad.clone().detach()

                    # ----------- client backward training ------------
                    output_client.backward(client_grad)
                    optimizer_client.step()

            # going for next user in the same group
            client_weights = copy.deepcopy(mnist_client.state_dict())
            group_client_weights_collect.append(client_weights)

            group_server_weights = copy.deepcopy(mnist_server.state_dict())

        # Now going for the different group
        # save the weights of the current model -- current group

        weights_server_list_group.append(group_server_weights)

    # Completion for all groups
    # average
    global_client_weights = average_weights(group_client_weights_collect, datasetsize_client)

    global_server_weights = average_weights(weights_server_list_group, datasetsize_server)

    weights_server_list_group = []
    group_client_weights_collect = []

    # ==============================================================================================

    # ================================================================================================================
    # Accuracy check at each round (this is for the gloabl model --- not local model) -------------
    # This is s result for the global model at each round --------
    mnist_client.load_state_dict(global_client_weights)
    mnist_server.load_state_dict(global_server_weights)
    mnist_client.eval()
    mnist_server.eval()

    # train acc for each client's training dataset
    # train acc
    with torch.no_grad():
        for u in range(users):
            corr_num = 0
            total_num = 0
            train_loss = 0.0
            for j, trn in enumerate(train_loaders[u]):
                trn_x, trn_label = trn
                trn_x = trn_x.to(device)
                trn_label = trn_label.clone().detach().long().to(device)

                trn_output = mnist_client(trn_x)
                trn_output = mnist_server(trn_output)
                loss = criterion(trn_output, trn_label)
                train_loss += loss.item()
                model_label = trn_output.argmax(dim=1)
                corr = trn_label[trn_label == model_label].size(0)
                corr_num += corr
                total_num += trn_label.size(0)
            train_accuracy = corr_num / total_num * 100
            r_train_loss = train_loss / len(train_loaders[u])
            print("client {}: rounds {}'s train_acc: {:.2f}%, train_loss: {:.4f}".format(u, r + 1, train_accuracy,
                                                                                         r_train_loss))
            train_acc.append(train_accuracy)
        avg_acc = sum(train_acc) / len(train_acc)
        print("Train_average accuracy:", avg_acc)
        train_acc_average.append(avg_acc)
        train_acc = []
    # test acc
    with torch.no_grad():
        corr_num = 0
        total_num = 0
        val_loss = 0.0
        for j, val in enumerate(test_loader):
            val_x, val_label = val
            val_x = val_x.to(device)
            val_label = val_label.clone().detach().long().to(device)

            val_output = mnist_client(val_x)
            val_output = mnist_server(val_output)
            loss = criterion(val_output, val_label)
            val_loss += loss.item()
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)
        test_accuracy = corr_num / total_num * 100
        test_loss = val_loss / len(test_loader)
        print("rounds {}'s test_acc: {:.2f}%, test_loss: {:.4f}".format(r + 1, test_accuracy, test_loss))
        val_acc.append(test_accuracy)

end_time = time.time()  # store end time
print("TrainingTime: {} sec".format(end_time - start_time))

print("training accuracy :")
print(train_acc_average)
print("testing accuracy: ")
print(val_acc)

mnist_client.load_state_dict(global_client_weights)
mnist_server.load_state_dict(global_server_weights)
mnist_client.eval()
mnist_server.eval()

test_loss = 0.0
class_correct = [0] * 10
class_total = [0] * 10

# For generating confusion matrix
conf_matrix = np.zeros((10, 10))

with torch.no_grad():
    corr_num = 0
    total_num = 0
    val_loss = 0.0
    for j, val in enumerate(test_loader):
        val_x, val_label = val

        data = val_x
        target = val_label

        data = data.to(device)
        # target = target.clone().detach().long().to(device)
        target = target.to(device)

        output = mnist_client(data)
        output = mnist_server(output)

        # loss = criterion(val_output, val_label)
        loss = criterion(output, target)

        # val_loss += loss.item()

        test_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(target.size(0)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
            # Update confusion matrix
            conf_matrix[label][pred.data[i]] += 1

# average test loss
test_loss = test_loss / len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %3s: %2d%% (%2d/%2d)' % (
            i, 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %3s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

import matplotlib.pyplot as plt
import numpy as np


test = np.array(train_acc_average)


# ypoints = np.array([3, 8, 1, 10])


ypoints = test

plt.plot(ypoints, marker = 'o')
plt.show()