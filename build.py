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
from PIL import Image
import argparse


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

    parser.add_argument('--learning_rate',
                        action = 'store',
                        dest = 'learning_rate',
                        type = float,
                        default = 0.001,
                        help = 'Model learning rate')

    parser.add_argument('--category_names',
                        action='store',
                        default ='cat_to_name.json',
                        type=str)
    
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
print("Learning Rate: {}".format(args.learning_rate))

data_dir = args.data_dir
arch = args.arch
learning_rate = args.learning_rate

def loader(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
    'train' : transforms.Compose([transforms.RandomRotation(30),
                                  transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])]),

    'valid' : transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])]),

    'test' : transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])}

    # Load the datasets with ImageFolder
    image_data = {
        'train' : datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train']),
        'valid' : datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['valid']),
        'test' : datasets.ImageFolder(data_dir + '/test', transform=data_transforms['test'])}
     # Using the image datasets and the trainforms, define the dataloaders
    dloader = {
        'train' : torch.utils.data.DataLoader(image_data['train'], batch_size=64, shuffle=True),
        'valid' : torch.utils.data.DataLoader(image_data['valid'], batch_size=64, shuffle=True),
        'test' : torch.utils.data.DataLoader(image_data['test'], batch_size=64)}

    class_to_idx = image_data['train'].class_to_idx
    return dloader, class_to_idx

dloader, class_to_idx = loader(data_dir)

arch = args.arch

def build_classifier(arch, class_to_idx):
    print(arch)
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_size = 1024
        hidden_units = 512
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_size = 25088
        hidden_units = 4096
    if arch == 'resnet18':
        model = models.resnet(pretrained=True)
        in_size = 1024
        hidden_units = 512
        
    out_size = 102
    
    print("Model Retrieved")

    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_size, hidden_units)),
                                 ('relu', nn.ReLU()),
                                 ('dout1', nn.Dropout(p=0.15)),
                                 ('fc2', nn.Linear(hidden_units, out_size)),
                                 ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    model.class_to_idx = class_to_idx
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    return model, criterion, optimizer, hidden_units, in_size, out_size

model, criterion, optimizer, hidden_units, in_size, out_size = build_classifier(arch, class_to_idx)
