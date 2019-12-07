import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops
from torch.nn import functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import scipy
import scipy.misc
from scipy import ndimage
import math
import os
import random

# Created by Sujit Sahoo, 26 Sept 2019
# sujit.sahoo@fau.de

vgg19_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'

class VGG19(nn.Module):
  '''
    Creates VGG19 acrh model
  '''

  def __init__(self,num_classes=1000,init_weights=True):
    super().__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3,64,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64,64,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64,128,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128,128,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128,256,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256,256,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256,256,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256,256,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(256,512,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.avgpool = nn.AdaptiveAvgPool2d((7,7))
    self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
    )
    if init_weights:
      self._initialize_weights()

  def forward(self,x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x,1)
    x = self.classifier(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight,0,0.01)
            nn.init.constant_(m.bias, 0)


class DeepDream():
    '''
    Given a label (number between 0 and 1000) of the ImageNet and
    an input image(zero image by default),label specific 'deep dream'
    images can be created

    '''

    def __init__(self,net=None):
        self.device = None
        self.net = net
        self.gaussian_filter = None
        self.ouputImage = None
        # list variables used in randomDream method
        self.nItrs = [300,400,500,600]
        self.lrs = [0.08,0.1,0.12,0.14]
        self.sigmas = [0.4,0.42,0.44,0.46,0.48,0.5]
        self.labels = [i for i in range(1000)]
        # set methods
        self.setDevice()
        self.setNetwork()
        self.setGaussianFilter()

    def setDevice(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used to run this program: ",self.device)


    def setNetwork(self):
        print("Loading the network...")
        
        if self.net is None:
            self.net = VGG19()
            state_dict = load_state_dict_from_url(vgg19_url)
            self.net.load_state_dict(state_dict)

        self.net.eval() # inference mode

        self.net.to(self.device)
        print("Network Loaded")

    def __call__(self,im=None,label=0,nItr=500,lr=0.1):
        """Does activation maximization on a specific label for specified iterations,
           acts like a functor, and returns an image tensor
        """

        if im is None:
            im = self.createInputImage()
            im = self.prepInputImage(im)
            im = im.to(self.device)

            im = Variable(im.unsqueeze(0),requires_grad=True)

            # offset by the min value to 0 (shift data to positive)
            min_val = torch.min(im.data)
            im.data = im.data - min_val

        print("Dreaming...")

        for i in range(nItr):

            optimizer = torch.optim.SGD([im],lr)
            out = self.net(im)
            loss = -out[0,label]

            loss.backward()
            optimizer.step()

            im.data = self.gaussian_filter(im.data)

            im.grad.data.zero_()

        return im

    def randomDream(self,im=None,randomSeed=0):
        """Does activation maximization on a random label for randomly chosen learning rate,number of iterations and gaussian filter size, and returns an image tensor
        """
        random.seed(randomSeed)
        nItr = random.choice(self.nItrs)
        lr = random.choice(self.lrs)
        sigma = random.choice(self.sigmas)
        label = random.choice(self.labels)
        self.setGaussianFilter(sigma=sigma)

        if im is None:
            im = self.createInputImage()
            im = self.prepInputImage(im)
            im = im.to(self.device)

            im = Variable(im.unsqueeze(0),requires_grad=True)

            # offset by the min value to 0 (shift data to positive)
            min_val = torch.min(im.data)
            im.data = im.data - min_val

        print("Dreaming...")

        for i in range(nItr):

            optimizer = torch.optim.SGD([im],lr)
            out = self.net(im)
            loss = -out[0,label]

            loss.backward()
            optimizer.step()

            im.data = self.gaussian_filter(im.data)

            im.grad.data.zero_()

        return im


    def createInputImage(self):
        zeroImage_np = np.zeros((224,224,3))
        zeroImage = Image.fromarray((zeroImage_np).astype('uint8'),'RGB')

        return zeroImage

    def prepInputImage(selfi,inputImage):
        #standard normalization for ImageNet data
        normalise = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )

        preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalise
        ])

        return preprocess(inputImage)

    def setGaussianFilter(self,kernelSize=3,sigma=0.5):

        # Create a x, y coordinate grid of shape (kernelSize, kernelSize, 2)
        x_cord = torch.arange(kernelSize)
        x_grid = x_cord.repeat(kernelSize).view(kernelSize, kernelSize)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        xy_grid = xy_grid.float()

        mean = (kernelSize - 1)/2.
        variance = sigma**2.


        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /(2*variance))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernelSize, kernelSize)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        pad = math.floor(kernelSize/2)

        gauss_filter = nn.Conv2d(in_channels=3, out_channels=3,padding=pad,
                            kernel_size=kernelSize, groups=3, bias=False)

        gauss_filter.weight.data = gaussian_kernel
        gauss_filter.weight.requires_grad = False
        self.gaussian_filter = gauss_filter.to(self.device)
        #print("gaussian_filter created")


    def postProcess(self,image):
        image = image.data.squeeze() # remove the batch dimension

        image.transpose_(0,1) # convert from CxHxW to HxWxC format
        image.transpose_(1,2)
  
        image = image.cpu() # back to host
  
        # TRUNCATE TO THROW OFF DATA OUTSIDE 5 SIGMA
        mean = torch.mean(image)
        std = torch.std(image)
        upper_limit = mean + 5 * std
        lower_limit = mean - 5 * std
        image.data = torch.clamp_(image.data,lower_limit,upper_limit)


        # normalize data to lie between 0 and 1
        image.data = (image.data - lower_limit) / (10*std)

        img = Image.fromarray((image.data.numpy()*255).astype('uint8'), 'RGB') #torch tensor to PIL image

        return img

    def show(self):
        plt.figure(num=1, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
        plt.imshow(np.asarray(self.outputImage))

    def save(self,image,fileName):
        output = image.resize((448,448), Image.ANTIALIAS)
        output.save(fileName,'PNG')
        print(f'{fileName} saved')



if __name__ == "__main__":
    dreamer = DeepDream()
    dreamer.setGaussianFilter(3,0.48) # optional step
    dreamtImage = dreamer(label=130)  # 130 is the label for flamingo, see https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    dreamtImage =  dreamer.postProcess(dreamtImage)
    dreamer.show() # shows image
    dreamer.save(dreamtImage,"myImage.png") # saves image

