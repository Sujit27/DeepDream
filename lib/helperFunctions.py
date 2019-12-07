from deepDream import *
from imageData import *
import torch.optim as optim
import sklearn.metrics as skm
import csv
from statistics import mean

# Created by Sujit Sahoo, 24.11.2019
# sujit.sahoo@fau.de

# Description : Has a bunch of helper functions needed to train a disriminator. must import

# creates a csv file given a list
def csvFromList(listName,fileName):
    with open(fileName,'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(listName)

# creates the VGG network, loads weights from pretrained model, adds a linear layer
# at the end to create a binary classifier
def createDiscriminator():
    net = VGG19()
    #state_dict = load_state_dict_from_url(vgg19_url)
    #net.load_state_dict(state_dict)

    #net.classifier = nn.Sequential(net.classifier,nn.ReLU(inplace=True),nn.Linear(1000,2))
    net.classifier = nn.Sequential(net.classifier,nn.ReLU(inplace=True),nn.Linear(1000,1))
    nn.init.normal_(net.classifier[2].weight,0,0.01)
    nn.init.constant_(net.classifier[2].bias,0)

    return net

def createDataset(data_path,realImagescsv,dreamImagescsv=None):
    # set data path for images and csv files
    data_path = data_path
    
    realImagescsv = os.path.join(data_path,realImagescsv)
    if dreamImagescsv is not None:
        dreamImagescsv = os.path.join(data_path,dreamImagescsv)

    # create dataset object
    dataset = ImageData(data_path,realImagescsv,dreamImagescsv
            ,preprocessFunc()) # creates dataset object
    
    return dataset

# split dataset into training and validation
def splitData(dataset,trainFraction):
    train_size = int(trainFraction * len(dataset))
    validate_size = len(dataset) - train_size
    datasetTrain,datasetValidate = torch.utils.data.random_split(dataset,[train_size,validate_size])

    return datasetTrain,datasetValidate

# creates a weightd sampler for the dataset 
def sampleDataset(dataset,datasetTrain):
    label_sample_count = [len(dataset.dreamImageIDs),len(dataset.realImageIDs)]
    weights = 1. / torch.Tensor(label_sample_count)
   
    samples_weights = []
    for item in datasetTrain:
        if item[1] == 0:
            samples_weights.append(weights[0])
        else:
            samples_weights.append(weights[1])

    samples_weights = torch.FloatTensor(samples_weights)

    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=samples_weights,num_samples=batch_size,replacement=False)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=samples_weights,num_samples=len(datasetTain),replacement=True)
    
    return sampler

# normalize by imagenet mean and std deviation
def normalizeFunc():
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    
    return normalize

# create preprocess transform
def preprocessFunc():
    #preprocess = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    preprocess = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalizeFunc()])

    return preprocess

def extractData(data,device):
    inputs,labels = data
    labels = labels.unsqueeze(1) # making the labels shape (batch_size x 1)
    inputs,labels = inputs.to(device), labels.to(device)
    labels = labels.float()

    return inputs,labels


# create accuracy and f1 score metrics from label and predicted values
def createMetrics(y_true,y_pred):
    accuracy = skm.accuracy_score(y_true,y_pred)
    f1_score = skm.f1_score(y_true,y_pred,average="weighted")
    
    return accuracy,f1_score

# function for printing out status
def logStatus(mode,device,epoch,step,loss,labels,outputs):
    memory_allocated = torch.cuda.memory_allocated(device=device)
    accuracy,f1_score = createMetrics(labels.cpu().detach().numpy(),outputs.cpu().detach().numpy() > 0.5)
    if step==0:
        print("#######################################################")
    else:
        print("###################")
    print(f"Epoch: {epoch} {mode} step: {step} loss: {loss}")
    print(f"F1 score : {f1_score}")
    print(f"Accuracy : {accuracy}")
    print(f"Memory allocated in gpu: {memory_allocated}")





