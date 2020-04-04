# TRAIN FLOWER CLASSIFIER
# Example to run: python train.py --arch vgg16 --gpu --epochs 3

import os
import PIL
import json
import time
import torch
import logging
import argparse
import numpy as np
from torch import nn
import seaborn as sns
from tqdm import tqdm
from torch import optim
from torchvision import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict

# Setup logging class
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

# Parse arguments received by prompt command
def parseArgs():
    parser = argparse.ArgumentParser(description="NeuralNet Arguments")
    
    # Add architecture selection to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Specify pretrained architecture from torchvision.models',
                        default='vgg16')
    
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Path to save checkpoints',
                        default='checkpoints')
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Specify learning rate',
                        default=0.001)
    
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Specify hidden layer output size',
                        default=4096)
    
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Specify training epoches',
                        default=3)

    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Enable GPU')
    
    args = parser.parse_args()   
    return args


# Create classifier model
def createClassifier(preTrainedModel, hidden_units, output=102):
    logging.info('Creating classifier with {} hidden_units and {} outputs'.format(hidden_units, output))

    clfInputSize = preTrainedModel.classifier[0].in_features
    clfOutputSize = output
    
    clf = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(clfInputSize, hidden_units, bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.4)),
        ('fc2', nn.Linear(hidden_units, clfOutputSize, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
        ]))

    return clf

# Load VGG16 pretrained model
def loadPretrainedModel(arch):
    logging.info('Loading pretrained model {}'.format(arch))

    model          = None
    try:
        model = getattr(models, arch)(pretrained=True)
        model.name = arch
        
        # Freeze feature extrator layers to not update with new weights
        for param in model.parameters():
            param.required_grad=False
    
    except Exception as e:
        print(e)
    
    return model

# Perform validation
def validate(model, testloader, criterion, device):
    logging.info('Validating model')

    testLoss = 0
    acc = 0
    
    for (inputs, labels) in testloader:
        # transfer data to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # perform forward pass
        output = model.forward(inputs)
        
        # get loss
        testLoss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        acc += equality.type(torch.FloatTensor).mean()
    
    return testLoss, acc

# Train the model
def train(model, epochs, train_loader, validation_loader, criterion, optimizer, device, print_every=40):
    logging.info('Training model...')
    steps = 0
    running_loss = 0

    for e in range(epochs):
        running_loss = 0
        model.train() 

        for (inputs, labels) in tqdm(train_loader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validate(model, validation_loader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validation_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)))

                running_loss = 0
                model.train()
    return model


# Save model checkpoint
def saveCheckpoint(model, save_dir, train_data):
    logging.info('Saving checkpoint')
                 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    checkpointFilename = 'modelCheckpoint.pth'

    # record mapping in model obj
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'architecture': model.name,
                 'classifier': model.classifier,
                 'class_to_idx': model.class_to_idx,
                 'state_dict': model.state_dict()}

    # dump the checkpoint object - in real scenario, use datetime to generate different file names
    torch.save(checkpoint, os.path.join(save_dir,"checkpoint.pth"))
    

# Start dataloaders based on train, test and validation directories
def loadData(train_dir, valid_dir, test_dir):
    
    # Compose transforms and return dir data
    train_data = composeTransform(train_dir)
    valid_data = composeTransform(valid_dir, test=True)
    test_data  = composeTransform(test_dir,  test=True)
    
    # Start DataLoader
    train_loader      = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_loader       = torch.utils.data.DataLoader(test_data, batch_size=32)
    return train_loader, validation_loader, test_loader, train_data, valid_data, test_data
    
# Generate torch transform for training / test / validation, set test for test and validation compose
def composeTransform(imageDir, test=False):
    composedTransform = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    if test:
        composedTransform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
        
    return datasets.ImageFolder(imageDir, transform=composedTransform)  
        
                
if __name__ == '__main__':
    # parse arguments from prompt command
    args = parseArgs()
    
    #specify directories
    data_dir  = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir  = data_dir + '/test'
    
    # start data loaders
    train_loader, validation_loader, test_loader, train_data, valid_data, test_data = loadData(train_dir, valid_dir, test_dir)
    
    # load Model
    model = loadPretrainedModel(arch=args.arch)
    
    # build Classifier
    model.classifier = createClassifier(model, hidden_units=args.hidden_units)
    
    device = 'cpu'
    # check GPU
    if 'gpu' in args:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
         
    # Train the classifier layers using backpropogation
    trainedModel = train(model, args.epochs, train_loader,validation_loader, criterion, optimizer, device)
    print("Training Complete")
    
    # Quickly Validate the model
    validate(trainedModel, test_loader, criterion, device)
    
    # Save the model
    saveCheckpoint(trainedModel, args.save_dir, train_data)