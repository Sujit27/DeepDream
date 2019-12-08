from deepDream import *
from helperFunctions import *

# created by Sujit Sahoo, 08.12.2019
# sujit.sahoo@fau.de

class DeepDreamGAN(DeepDream):
    '''
    Given a network and a discriminator network, creates a 
    GAN like system where the network iteratively dreams an
    image and the discriminator adds its gradients on top of
    the input depending on its opinion of how 'realistic' it is
    '''

    def __init__(self,net=None,discrimNet=None,dataPath='/var/tmp/imageData/'):
        super().__init__(net)
        self.discrimNet = discrimNet
        self.dataPath = dataPath
        # list variables used in randomDream method
        self.nItrs = [5,10,15,20]
        # set methods
        self.setDiscrimNet()

    def setDiscrimNet(self):
        print("Loading the discriminator...")
        
        if self.discrimNet is None:
            self.discrimNet = createDiscriminator()

        self.discrimNet.to(self.device)
        print("Discriminator Loaded")

        print("Discriminator being trained")
        self.trainDiscrimNet()

    def trainDiscrimNet(self,steps=50):
        '''
        Trains the discriminator to differentiate between
        real and dream images for a given number of steps,
        and ideally not for a whole epoch. For our purpose
        We do not want a discriminator which is too confident
        '''
        realImagescsv = 'realImages.csv'
        dreamImagescsv = 'dreamImages.csv'

        #creates dataset from images in the local drive
        dataset = createDataset(self.dataPath,realImagescsv,dreamImagescsv)

        #split dataset into train and validate
        datasetTrain,datasetValidate = splitData(dataset,0.75)

        numEpochs = 1
        batch_size = 32
        
        # set parameters for dataloader
        params = {'batch_size': batch_size,'shuffle':True,'pin_memory':True}
        
        # create dataloader
        trainLoader = torch.utils.data.DataLoader(datasetTrain,**params)
        valLoader = torch.utils.data.DataLoader(datasetValidate,**params)
        # loss criterion and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(self.discrimNet.parameters(),lr=0.005,momentum=0.9)

        for epoch in range(numEpochs):
                print("Training the discriminator...")

                for step,data in enumerate(trainLoader):
                   
                    # extracts images and labels and moves them to gpu
                    inputs,labels = extractData(data,self.device)                        
                    
                    optimizer.zero_grad()

                    outputs = self.discrimNet(inputs)
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optimizer.step()
                   
                    if step % 10 == 0:
                        logStatus('Training',self.device,epoch,step,loss,labels,outputs) 
                    
                    if step == steps:
                        break

        print("Discriminator training ends")



    def __call__(self,im=None,label=0,nLoop=20,nItr1=20,nItr2=10,lr1=0.1,lr2=0.1):
        '''
        Functor that creates a tensor from an input tensor after combined effects
        from the network(parameters nItr1,lr1) and discriminator(parameters nItr2,lr2)
        ,iterating through it nLoop times
        '''
        pass

        return im

    def randomDream(self,im=None,randomSeed=0):
        '''
        Does the same thing as the functor but choses parameters randomly from a list
        '''
        nLoop = 20
        random.seed(randomSeed)
        nItr1 = random.choice(self.nItrs)
        nItr2 = random.choice(self.nItrs)
        lr1 = random.choice(self.lrs)
        lr2 = random.choice(self.lrs)
        sigma = random.choice(self.sigmas)
        label = random.choice(self.labels)
        self.setGaussianFilter(sigma=sigma)

        pass

        return im
