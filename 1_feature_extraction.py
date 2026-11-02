import os
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torchvision import datasets
import torchvision

"""
Creates an array with the feature vectors and a txt with the image labels:

    - loads pretrained VGG-16 model, removes final classification layers, freezes weights
    - load images and normalize with imagenet statistics, the resizing step was removed from the original
    - passes images through model and saves them as pickle file (4096 dim vector)
    - extract labels from image names (naming convention: <category>_<description>__<source_dataset>)
    - saving labels under image_labels.txt (for compatibility with vong and lake scene generation)
"""

### Configure directories dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, "data")
#RAW_DIR = os.path.join(BASE_DIR, "data", "raw_images") 
RESIZED_DIR = os.path.join(BASE_DIR, "data", "resized_images") 


class VGGFeatureExtractor:
    def __init__(self):
        # load VGG-16
        self.model = torchvision.models.vgg16(pretrained=True)
        
        #remove classification head
        self.model.classifier = torch.nn.Sequential(
            *list(self.model.classifier.children())[:2]
        )
        
        # freeze model
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def _extract_noun_features(self):
        """Extract 4096-dim feature vectors from resized images with batch method"""
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        # process the input data
        image_dataset = datasets.ImageFolder(root=RESIZED_DIR, transform=T)
        dataset_size = len(image_dataset)
        dataloader = torch.utils.data.DataLoader(image_dataset,
                                                 batch_size=dataset_size,
                                                 shuffle=False,
                                                 num_workers=0)
        inputs, _ = next(iter(dataloader))
        inputs = inputs.to(self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.model(inputs)
        
        outputs = outputs.cpu().detach()

        # convert to numpy array
        features_array = outputs.numpy()
        features_file = os.path.join(DATA_DIR, 'features.pickle')

        # save features as binary
        with open(features_file, 'wb') as f:
            pickle.dump(features_array, f)
            
        return features_array

def main():
    # setup extractor, it already received the target directory
    extractor = VGGFeatureExtractor()
    features = extractor._extract_noun_features()

    ### create a txt with the labels
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_filenames = []
    for root, dirs, files in os.walk(RESIZED_DIR):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_filenames.append(file)

    image_filenames.sort()
    
    image_labels = [fname.split('_')[0] for fname in image_filenames]  # extract the name from the images of the compiled dataset

    labels_file = os.path.join(DATA_DIR, 'image_labels.txt')    # save
    with open(labels_file, 'w') as f:
        for label in image_labels:
            f.write(label + '\n')
    
    # output to check shape and directory
    print(f"Features: {os.path.join(DATA_DIR, 'features.pickle')}") 
    print(f"Labels: {labels_file}")
    print(f"Feature shape: {features.shape}")

if __name__ == "__main__":
    main()