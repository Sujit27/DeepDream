import sys
sys.path.append('../lib/')
from trainDiscriminator import *

# created by Sujit Sahoo, 30.11.2019
# sujit.sahoo@fau.de

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

numInitialDreamItrs = 100
numIntermediateLoops = 20
numIntermediateDreamItrs = 20
numIntermediateDiscrimItrs = 10
lrDiscriminator = 0.1
label = 538 # dome

def main():
    '''
    Uses DeepDream and Discriminator to create  more realistic dreams
    '''

    dreamer = DeepDream()
    dreamer.setGaussianFilter(sigma=0.48)
    
    discriminator = createDiscriminator()
    try:
        discriminator.load_state_dict(torch.load('discriminator.pth'))
    except FileNotFoundError:
        print("There is no discriminator.pth file available. Maybe run trainDiscriminator.py to create it")
        raise
    discriminator.to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    desiredLabel = np.array([1.0]) # For the discriminator, label for dream and real images is 0 and 1 respectively
    desiredLabel = torch.from_numpy(desiredLabel).float().to(device)
    desiredLabel = desiredLabel.unsqueeze(0)

    dreamTensor = dreamer(label=label,nItr=numInitialDreamItrs,lr=0.12)

    for i in range(numIntermediateLoops):
        print(f'Loop {i} out of {numIntermediateLoops}')

        for j in range(numIntermediateDiscrimItrs):

            optimizer = torch.optim.SGD([dreamTensor],lrDiscriminator)
            out = discriminator(dreamTensor)
            loss = criterion(out,desiredLabel)
            
            loss.backward()
            optimizer.step()
            
            ## POSSIBLY A GAUSSIAN FILTER STEP SHOULD COME HERE ???
            dreamTensor.grad.data.zero_()


        dreamTensor = dreamer(im=dreamTensor,label=label,nItr=numIntermediateDreamItrs,lr=0.12)

    outputImage = dreamer.postProcess(dreamTensor)
    dreamer.save(outputImage,"realDream.png")

if __name__ == "__main__":
    main()
