
# PREDICT FLOWERS
# Example to run: python predict.py --image "flowers/test/28/image_05242.jpg" --checkpoint checkpoints/checkpoint.pth --category_names cat_to_name.json --top_k 10

import PIL
import json
import torch
import logging
import argparse
import numpy as np
from PIL import Image
from torchvision import models

# Setup logging class
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

# Parse arguments received by prompt command
def parseArgs():

    parser = argparse.ArgumentParser(description="NeuralNet Inference Arguments")

    parser.add_argument('--image', 
                        type=str, 
                        help='Path to image',
                        required=True)

    
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Path to model checkpoint',
                        required=True)
    
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Return top k results',
                        default=5)
    
    parser.add_argument('--category_names', 
                        type=str, 
                        help='Category mapping file')

    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Enable GPU')

    args = parser.parse_args()
    return args


# Load model checkpoint according to checkpoint path
def loadCheckpoint(checkpointPath):
    logging.info('Loading model checkpoint {}...'.format(checkpointPath))

    # Load model checkpoint
    checkpoint = torch.load(checkpointPath)
    
    # Retrieve pytorch architecture model based on the checkpoint
    modelArchStr = checkpoint['architecture']
    model = getattr(models, modelArchStr)(pretrained=True)
    model.name = modelArchStr
    
    # Freeze feature extrator layers to not update with new weights
    for param in model.parameters():
        param.required_grad=False
    
    # load mapping from model obj
    model.class_to_idx = checkpoint['class_to_idx']
    
    # load classifier
    model.classifier = checkpoint['classifier']
   
    # load state dict
    model.load_state_dict(checkpoint['state_dict'])
       
    return model

# Preprocess image before feeding to the network
# Resize to 256, Center Crop, Resize to 224, Normalize Channels and transpose to channels first
def processImage(imagePath):
    logging.info('Preprocessing image {}...'.format(imagePath))

    img = Image.open(imagePath)
    width, height = img.size
    
    # find the shortest dimension, ratio and the new dimension
    if width < height:
        ratio = (height/width)
        width = 256
        height = int(ratio*width)
    else:
        ratio = (width/height)
        height = 256
        width = int(height*ratio)
    
    # perform resisze
    img = img.resize((width, height))

    # define center
    centerX, centerY = width // 2, height // 2    
   
    # define borders
    left = centerX-112
    right = centerX+112
    top = centerY-112
    bottom = centerY+112
    newWidth = right - left
    newHeight = bottom - top
    
    # crop image around center
    croppedImg = img.crop((left, top, right, bottom))
    croppedImg = np.array(croppedImg)/255
        
    # Normalize each color channel
    normalizeMeans = [0.485, 0.456, 0.406]
    normalizeStd = [0.229, 0.224, 0.225]
    normalizedImg = (croppedImg-normalizeMeans)/normalizeStd

    # make channels first
    normalizedImg = normalizedImg.transpose(2, 0, 1)
    return normalizedImg

# Perform network inference based on imagePath and model
def predict(imagePath, model, topk, device, cat_to_name):
    logging.info('Predicting image {}...'.format(imagePath))

    ktopLabels = []
    ktopFlowers = []
    
    # Send model to the respective device
    model.to(device)
    
    # Model as inference (just forward pass)
    model.eval();
    
    # Preprocess image, transform to tensor and send to the device
    img = processImage(imagePath)
    img = torch.from_numpy(np.expand_dims(img, axis=0))
    img = img.type(torch.FloatTensor).to(device)
    
    # perform forward pass
    logSoftmax = model.forward(img)
    
    # transform log to linear
    probs = torch.exp(logSoftmax)

    # Find the top 5 results
    top_probs, top_labels = probs.topk(topk)  
    
    # Relase tensor to CPU and transform to numpy
    top_probs, top_labels = np.squeeze(top_probs.cpu().detach().numpy()), np.squeeze(top_labels.cpu().detach().numpy())
    
    # Parse lbls to dict
    classes = {val: key for key, val in model.class_to_idx.items()}
    
    # Map classes to indexes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lbl] for lbl in top_labels]
    top_flowers = [cat_to_name[lbl] for lbl in top_labels]
        
    return top_probs, top_flowers

# Pretty display probs and flowers
def displayProbs(probs, flowers):
    for idx, (prob, flower) in enumerate(zip(probs,flowers)):
        print("[{}] Flower {} - Prob {}%".format(idx+1, flower, round(prob*100,2)))
    
    
if __name__ == '__main__':
    
    # parse arguments from prompt command
    args = parseArgs()

    # Load model checkpoint
    model = loadCheckpoint(args.checkpoint)
    
    # Check if GPU is available, otherwise use CPU as default
    device = 'cpu'
    # check GPU
    if 'gpu' in args:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load categories
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)
            
    top_probs, top_flowers = predict(args.image, model, args.top_k, device, cat_to_name)
    displayProbs(top_probs, top_flowers)