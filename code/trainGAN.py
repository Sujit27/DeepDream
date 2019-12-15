import sys
sys.path.append('../lib/')
from deepDreamGAN import *

#created by sujit Sahoo 09.12.2019
#sujit.sahoo@fau.de

#Description : To train a DeepDreamGAN object and save it

def main():
    
    # create DeepDreamGAN type object
    dreamer = DeepDreamGAN()

    #dreamer.trainDiscrimNet() # TRAINS THE DISCRIMINATOR A BIT AT START

    dataPath = '/proj/ciptmp/on63ilaw/imageData'
    realImagescsv = 'realImages.csv'
    batch_size = 32
    lr = 0.005
    
    # loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(dreamer.discrimNet.parameters(),lr=lr,momentum=0.9)

    numDreamImages = 16
    numRealImages = batch_size - numDreamImages
    
    realImageDataset = createDataset(dataPath,realImagescsv)
    params = {'batch_size': numRealImages,'shuffle':True,'pin_memory':True}
    realImageLoader = torch.utils.data.DataLoader(realImageDataset,**params)

    dreamLabels = np.zeros((numDreamImages,1)) # target label for dream is 0
    dreamLabels = torch.from_numpy(dreamLabels).float().to(dreamer.device)

    # tensor of dream images
    dreamImages = torch.zeros([numDreamImages,3,224,224], device=dreamer.device)
    
    print("GAN training begins")
    for step,data in enumerate(realImageLoader):
    
        print(f'Step : {step}')
        # generate the dream images for this batch
        for i in range(numDreamImages):
            dreamImages[i,:,:,:] = dreamer.randomDream()
        
        optimizer.zero_grad()
        # collect gradient from dream images
        #dreamOutputs = dreamer.discrimNet(dreamImages)
        #loss = criterion(dreamOutputs,dreamLabels)
        #loss.backward(retain_graph=True)

        # collect gradient from real images
        realImages,realLabels = extractData(data,dreamer.device)
        #realOutputs = dreamer.discrimNet(realImages)
        #loss = criterion(realOutputs,realLabels)
        #loss.backward()
        
        all_images = torch.cat((dreamImages,realImages),0)
        all_labels = torch.cat((dreamLabels,realLabels),0)

        all_outputs = dreamer.discrimNet(all_images)
        loss = criterion(all_outputs,all_labels)
        loss.backward()

        optimizer.step()

        if step % 30 == 29:
            savedFileName = 'discriminatorGAN_'+str(step)+'.pth'
            torch.save(dreamer.discrimNet.state_dict(),savedFileName)

if __name__ == "__main__":
    main()

