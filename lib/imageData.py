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
    def __init__(self,data_path,realImagescsv,dreamImagescsv,transform=None):

        self.data_path = data_path
        
        # read file names from csv
        self.realImageIDs =  pd.read_csv(realImagescsv,header=None)
        labelReal = np.ones(len(self.realImageIDs))
        self.transform = transform
        
        if dreamImagescsv is not None:
            self.dreamImageIDs = pd.read_csv(dreamImagescsv,header=None)
            labelDream = np.zeros(len(self.dreamImageIDs))

            self.imageIDs = self.dreamImageIDs[0].tolist() + self.realImageIDs[0].tolist()
            self.labels = np.concatenate([labelDream,labelReal])

        else:
            self.imageIDs = self.realImageIDs[0].tolist()
            self.labels = labelReal



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

