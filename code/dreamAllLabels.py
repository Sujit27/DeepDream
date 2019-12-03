import sys
sys.path.append('../lib/')
from deepDream import *
'''
dream all 1000 labels, multiple copies of a label by changing learning rate and  gaussian kernel
'''
def main():
    nItrs = [500]
    lrs = [0.08,0.1,0.12,0.14]
    sigmas = [0.4,0.42,0.44,0.46,0.48,0.5]

    image_dict = {}

    dreamer = DeepDream()
    for sigma in sigmas:
        dreamer.setGaussianFilter(sigma=sigma) # create gaussian filter
        for label in range(1000):
            for lr in lrs:
                for nItr in nItrs:
                    outputImage = dreamer(label=label,nItr=nItr,lr=lr) # dream 
                    fileName = "dream_"+str(label)+"_"+str(nItr)+"_"+str(lr)+"_"+str(sigma)+".png"
                    print (fileName)
                    image_dict[fileName]=outputImage
        
            if (label % 10 == 0): # clear out by saving the output images
                for name,image in image_dict.items():
                    dreamer.save(image,name) # save the images
        
                image_dict.clear() # clear the dictionary

if __name__ == "__main__":
    main()
