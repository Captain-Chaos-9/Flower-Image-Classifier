import matplotlib.pyplot as plt
import torch
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
from build import loader, build_classifier
import argparse
from scipy.io import loadmat

print('hello world')
def get_arg():
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument('--image_path', type=str, dest='image_path', default='flowers/test/13/image_05769.jpg')
    parser.add_argument('--checkpoint', dest='checkpoint', default = 'checkpoint.pth', type = str, help = 'Checkpoint to load')
    parser.add_argument('--topk', default = 5, type = int, help = 'Top k most likely classes')
    parser.add_argument('--class_to_names', default = 'cat_to_name.json', type = str, help = 'Map of category to names')
    parser.add_argument('--device', default = 'cpu', type = str, help = 'Device to train model on')
    parser.add_argument('--hidden_units', default=512, type=int, dest='hidden_units', help='number of layers')
    parser.add_argument('--arch', action = 'store', dest = 'arch', default = 'densenet121', choices = 'architectures', help = 'Architectures available to use')
    parser.add_argument('--data_dir', default = 'flowers', dest = 'data_dir', type = str, help = 'flower directory')
    args = parser.parse_args()

    return args

args = get_arg()
    
image_path = args.image_path
checkpoint_path = args.checkpoint
class_to_name = args.class_to_names
topk = args.topk
device = args.device
hidden_units = args.hidden_units
arch = args.arch
data_dir = args.data_dir

print(topk)
print(checkpoint_path)
print(hidden_units)

dloader, class_to_idx = loader(data_dir)
model, criterion, optimizer, hidden_units, in_size, out_size = build_classifier(arch, class_to_idx)
def load_checkpoint(checkpoint_path):
    state = torch.load(checkpoint_path)
    class_to_idx = state['class_to_idx']
    hidden_units = state['hidden_units']
    arch = state['arch']
    in_size = state['in_size']
    out_size = state['out_size']
    epochs = state['epochs']
    model, criterion, optimizer, hidden_units, in_size, out_size = build_classifier(arch, class_to_idx)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print("Loaded '{}' (arch={}, hidden_units={}, epochs={})".format(
        checkpoint_path,
        'arch',
        'in_size',
        'out_size',
        'hidden_units',
        'epochs'))
    return model

model = load_checkpoint(checkpoint_path)


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open('flowers/test/13/image_05769.jpg')
    # TODO: Process a PIL image for use in a PyTorch model
    size = 224
    # TODO: Process a PIL image for use in a PyTorch model
    width, height = image.size

    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)

    resized_image = image.resize((width, height))

    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size
    cropped_image = image.crop((x0, y0, x1, y1))
    np_image = np.array(cropped_image) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image_array = (np_image - mean) / std
    np_image_array = np_image.transpose((2, 0, 1))

    return np_image_array

np_image_array = process_image(image_path)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()
    image = Image.open(image_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)
    if use_gpu:
        var_inputs = Variable(tensor.float().cuda(), volatile=True)
    else:
        var_inputs = Variable(tensor, volatile=True)
    var_inputs = var_inputs.unsqueeze(0)
    output = model.forward(var_inputs)
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu() if use_gpu else ps[0]
    classes = ps[1].cpu() if use_gpu else ps[1]
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])
    return probabilities.numpy()[0], mapped_classes

# image_path = test_dir + '/13/image_05769.jpg'
probabilities, classes = predict(image_path, model)

print(probabilities)
print(classes)

max_index = np.argmax(probabilities)
mx_ind = np.argmax(classes)
max_probability = probabilities[max_index] * 100
print(max_probability)
label = classes[max_index]

with open('cat_to_name.json', 'r') as f:
     cat_to_name = json.load(f)
     flower = cat_to_name[label]

print("flower: {}, likelyhood: {}%".format(flower, max_probability))
