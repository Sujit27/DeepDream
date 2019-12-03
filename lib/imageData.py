import numpy as np
import pandas as pd
import os
from PIL import Image
import glob
from torch.utils.data import Dataset

# Created by Sujit Sahoo, 24.11.2019
# sujit.sahoo@fau.de

class ImageData(Dataset):
    '''
    Creates a map style dataset object to be used by dataloader
    To construct, takes input csv file with names of the dream images in the
    rows, similar for real images
    example:
    dataset = ImageData('/../images','dreamImageNames.csv','realImageNames.csv',preprocess)
    '''
    def __init__(self,data_path,dreamImagescsv,realImagescsv,transform=None):

        self.data_path = data_path
        
        # read file names from csv
        self.dreamImageIDs = pd.read_csv(dreamImagescsv,header=None)
        self.realImageIDs =  pd.read_csv(realImagescsv,header=None)

        self.transform = transform
        
        #dream_hot_encoded = np.array([1,0])
        #real_hot_encoded = np.array([0,1])
        # create labels, 0 for dream images and 1 for real images
        #labelDream = np.array([dream_hot_encoded]*len(self.dreamImageIDs))
        #labelReal = np.array([real_hot_encoded]*len(self.realImageIDs))

        labelDream = np.zeros(len(self.dreamImageIDs))
        labelReal = np.ones(len(self.realImageIDs))

        self.imageIDs = self.dreamImageIDs[0].tolist() + self.realImageIDs[0].tolist()
        self.labels = np.concatenate([labelDream,labelReal])


    def __len__(self):
        return len(self.imageIDs)
    
    # indexing the dataset will return a tuple of image tensor and the label
    def __getitem__(self,idx):
        fullFilePath = os.path.join(self.data_path,self.imageIDs[idx])
        im = Image.open(fullFilePath)
        imTensor = self.transform(im)
        label = self.labels[idx]
        im.close()
        return imTensor, label

