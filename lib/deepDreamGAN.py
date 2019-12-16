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

        batch_size = 32
        
        # set parameters for dataloader
        params = {'batch_size': batch_size,'shuffle':True,'pin_memory':True}
        
        # create dataloader
        trainLoader = torch.utils.data.DataLoader(datasetTrain,**params)
        valLoader = torch.utils.data.DataLoader(datasetValidate,**params)
        # loss criterion and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(self.discrimNet.parameters(),lr=0.005,momentum=0.9)

        print("Discriminator will be updated and initialized to weights that achieve 'decent' accuracy for classifying dream and real images")
        print("Initializing...")

        for step,data in enumerate(trainLoader):
           
            # extracts images and labels and moves them to gpu
            inputs,labels = extractData(data,self.device)                        
            
            optimizer.zero_grad()

            outputs = self.discrimNet(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
           
                                
            if step == steps:
                logStatus('Training',self.device,0,step,loss,labels,outputs)
                print('##########################')
                break

        print("Initialization ends")



    def __call__(self,im=None,label=0,num_adversarial_loops=20,gen_nItr=20,dis_nItr=10,gen_lr=0.1,dis_lr=0.1):
        '''
        Functor that creates a tensor from an input tensor after combined effects
        from the network(parameters gen_nItr,gen_lr) and discriminator(parameters dis_nItr,dis_lr)
        ,iterating through it nLoop times
        '''
        if im is None:
            im = self.createInputImage()
            im = self.prepInputImage(im)
            im = im.to(self.device)

            im = Variable(im.unsqueeze(0),requires_grad=True)

            # offset by the min value to 0 (shift data to positive)
            min_val = torch.min(im.data)
            im.data = im.data - min_val

        print("Dreaming...")

        for _ in range(num_adversarial_loops):
            
            for _ in range(gen_nItr):
                optimizer = torch.optim.SGD([im],gen_lr)
                out = self.net(im)
                loss = -out[0,label]

                loss.backward()
                optimizer.step()

                im.data = self.gaussian_filter(im.data)

                im.grad.data.zero_()


            for _ in range(dis_nItr):

                optimizer = torch.optim.SGD([im],dis_lr)
                out = self.discrimNet(im)
                loss = -out  #criterion(out,desiredLabel)
                
                loss.backward()
                optimizer.step()
                
                im.grad.data.zero_()

        return im

    def randomDream(self,im=None,randomSeed=0):
        '''
        Does the same thing as the functor but choses parameters randomly from a list
        '''
        num_adversarial_loops = 20
        random.seed(randomSeed)
        rand_gen_nItr = np.asscalar(np.random.normal(20,1,1).astype(int))
        rand_dis_nItr = np.asscalar(np.random.normal(10,1,1).astype(int))
        rand_gen_lr = np.asscalar(np.random.normal(0.12,0.01,1))
        rand_dis_lr = np.asscalar(np.random.normal(0.10,0.01,1))
        rand_sigma = np.asscalar(np.random.normal(0.45,0.05,1))
        label = random.choice(self.labels)
        self.setGaussianFilter(sigma=rand_sigma)

        im = self.__call__(im,label=label,num_adversarial_loops=num_adversarial_loops,gen_nItr=rand_gen_nItr,dis_nItr=rand_dis_nItr,gen_lr=rand_gen_lr,dis_lr=rand_dis_lr)

        return im
