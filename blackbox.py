import torch
import gc
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import os
import sys
from evaluator2 import *
from datahandler2 import DataHandler
from neural_net_motifs import *
from thop import clever_format, profile

gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data set
dataset = str(sys.argv[1])
num_conv_layers = int(sys.argv[2])  # 0

num_full_layers = int(sys.argv[3])  # 20
list_blocks = 30 * [-1]

for i in range(20):
    list_blocks[i] = int(sys.argv[3 + i + 1])
indx = len(list_blocks)

batch_size_index = 2 + (2 + num_conv_layers*5 + num_full_layers)
batch_size = int(sys.argv[batch_size_index])

# lr = float(sys.argv[batch_size_index+2])
arg2 = float(sys.argv[batch_size_index+3])
arg3 = float(sys.argv[batch_size_index+4])
arg4 = float(sys.argv[batch_size_index+5])
dropout = float(sys.argv[batch_size_index + 6])
arg1 = float(sys.argv[batch_size_index + 2])

# Dataloaders
dataloader = DataHandler(dataset, batch_size)
image_size, number_classes = dataloader.get_info_data
trainloader, testloader = dataloader.get_loaders()

initial_image_size = 32
total_classes = 10
number_input_channels = 3

model = NeuralNet(list_blocks, initial_image_size, total_classes, number_input_channels, dropout)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=arg1, momentum=arg2, dampening=arg3, weight_decay=arg4)

print(model)

# The evaluator trains and tests the network
evaluator = Evaluator(device, model, trainloader, testloader, optimizer, batch_size, dataset)
print('> Training')

best_val_acc, best_epoch, nb_epochs = evaluator.train()
cnt = 1
dsize = (1, 3, 32, 32)
inputs = torch.randn(dsize).to(device)
macs, params = profile(model, (inputs,), verbose=False)

# # Output of the blackbox
print('> Final accuracy %.3f' % best_val_acc)
print('Count eval', cnt)
print('Number of epochs ', nb_epochs)
print('MACS and NB_PARAMS', macs, params)

