# GAN_DeepDream

**Deep dream with generative adversarial flavor**

Required softwares : Pytorch, cuda

***

1. lib/deepDream.py : Has class difnition for basic DeepDream() type object that lets you dream on any specific label. Default network
initialization for the object is VGG19.

``` dreamer = DeepDream()
imgTensor = dreamer(label=130)
img = dreamer.postProcess(imgTensor)
dreamer.save(img,"file_name.png")
```
OR
``` dreamer = DeepDream(net) 
imgTensor = dreamer(im=inputTensor,label=some_suitable_label) # inputTensor is some image tensor with shape appropriate for net
img = dreamer.postProcess(imgTensor)
dreamer.save(img,"file_name.png")
```


2. testing/ : unittests for DeepDream implementation. Run with following command

```$ python -m unittest```


3. code/dreamAllLabels.py : Script that creates multiple dream images on each label of ImageNet by varying learning rate
and gaussian filter.

Sample outputs:

![Label 538(Dome)](https://github.com/Sujit27/GAN_DeepDream/blob/master/results/dream_538_500_0.12_0.48.png)
![Label 71(Scorpion)](https://github.com/Sujit27/GAN_DeepDream/blob/master/results/dream_71_500_0.12_0.48.png)

***

__To train the discriminator, first import the dataset(~14GB) from the following remote system into your local disk 
 and unzip. This should create an imageNet directory at /var/tmp on the local system__
 
```$ scp user_name@faui04m.informatik.uni.erlangen.de:/var/temp/ /var/temp/```

This imageNet directory contains 37856 .png dream images created by lib/dreamAllLabels.py, 40100 Image Net validation set
images available for download[here](https://academictorrents.com/collection/imagenet-2012) and 2 .csv files listing the image names.

4. lib/imageData.py : Class definition of customozied dataset.

5. code/trainDiscriminator.py : A linear layer(1000,1) is added on top of VGG19 and is trained as a classifier on dream and
real imageNet data mentioned above. Saves the trained net as __'discriminator.pth'__.

6. code/createRealisticDreams.py : Has flavor of a static genrative adversarial system. DeepDream object and the discriminator
created from code/trainDiscriminator.py act in tandem to change an input image.

Sample outputs: plain dream on left, 'more realistic' dream on right

![Label 538(Dome)](https://github.com/Sujit27/GAN_DeepDream/blob/master/results/dome_0.48sigma.png)
