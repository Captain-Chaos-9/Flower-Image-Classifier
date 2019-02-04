import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from math import ceil
from PIL import Image
import argparse
from build import loader, build_classifier


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        default = 'flowers',
                        dest = 'data_dir',
                        type = str,
                        help = 'flower directory')

    parser.add_argument('--arch',
                        action = 'store',
                        dest = 'arch',
                        default = 'densenet121',
                        choices = 'architectures',
                        help = 'Architectures available to use')

    parser.add_argument('--save_dir',
                        action = 'store',
                        dest = 'save_dir',
                        default = './checkpoint.pth',
                        help = 'Sets checkpoint directory')

    parser.add_argument('--epochs',
                        action = 'store',
                        dest = 'epochs',
                        type = int,
                        default = 7,
                        help = 'Number of epochs to run')

    parser.add_argument('--learning_rate',
                        action = 'store',
                        dest = 'learning_rate',
                        type = float,
                        default = 0.001,
                        help = 'Model learning rate')

    parser.add_argument('--gpu',
                        action = 'store',
                        default = True,
                        dest = 'gpu',
                        help = 'enable GPU for training')
    
    parser.add_argument('--hidden_units',
                        action='store',
                        dest='hidden_units',
                        type= int,
                        default = 512,
                        help = 'Sets the number of hidden units for the hidden layer')

    return parser.parse_args()

get_arguments()

args = get_arguments()
print("Data Directory: {}".format(args.data_dir))
print("Pytorch Architecture: {}".format(args.arch))
print("File Save Directory: {}".format(args.save_dir))
print("Number of epochs: {}".format(args.epochs))
print("Learning Rate: {}".format(args.learning_rate))
print("Use GPU: {}".format(args.gpu))

data_dir = args.data_dir
arch = args.arch
learning_rate = args.learning_rate
epochs = args.epochs
device = args.gpu

dloader, class_to_idx = loader(data_dir)
trainloader = dloader['train']
validloader = dloader['valid']
def validation(model, validloader, criterion):
    model.to('cuda')
    valid_loss = 0
    accuracy = 0
    #validloader = dloader['valid']
    for data in validloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


model, criterion, optimizer, hidden_units, in_size, out_size = build_classifier(arch, class_to_idx)
def train_model(model, epochs, criterion, optimizer, trainloader, validloader, device):
    
    if device == True and torch.cuda.is_available():
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu") 
    
    model.to(device)
    
    model.to(device)
    print_every = 30
    steps = 0
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                          "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()
    return model

model = train_model(model, epochs, criterion, optimizer, trainloader, validloader, device)

def sav_ck(model, arch, learning_rate, hidden_units, in_size, out_size, epochs, optimizer):
    checkpoint_path = 'checkpoint.pth'

    state = {
        'arch': 'densenet121',
        'in_size': in_size,
        'hidden_units': hidden_units,
        'out_size': out_size,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'class_to_idx' : model.class_to_idx
    }

    torch.save(state, checkpoint_path)

sav_ck(model, arch, learning_rate, hidden_units, in_size, out_size, epochs, optimizer)
