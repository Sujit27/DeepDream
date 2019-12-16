import sys
sys.path.append('../lib/')
import unittest
from deepDreamGAN import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def imageToTensor(img):
    '''
    :param img: PIL image RGB format
    :return: tensor, batch x channel x height x width
    '''
    normalise = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalise
    ])

    img_tensor = preprocess(img)
    img_tensor = img_tensor.to(device)

    inputTensor = img_tensor.unsqueeze(0)

    return  inputTensor

def createModels():
    '''
    Create two VGG19, one from implementation in deepDream and
    the other from torchvision models
    '''
    model = DeepDream().net
    state_dict = load_state_dict_from_url(vgg19_url)
    model.load_state_dict(state_dict)

    torch_model = models.vgg19(pretrained=True)
    torch_model.eval() # inference mode

    torch_model.to(device)

    return model, torch_model

class TestVgg(unittest.TestCase):
    '''
    To test the correctness of creation of vgg19 in the
    DeepDream class
    '''
    def test_equalWeights(self):
        '''Test if our implementation of vgg19 and one
        from torchvision have same weights and bias
        '''
        model, torch_model = createModels()

        # check if both the models have same weights and biases
        a = model.state_dict().items()
        b = torch_model.state_dict().items()
        self.assertEqual((a > b) - (a < b),0)

    def test_equalPredictions(self):
        '''
        Test if  our implementation of vgg19 and one
        from torchvision give same prediction on some sample images
        present in the directory test_dataset
        '''
        model, torch_model = createModels()

        directory = 'test_dataset'
        ourLabelPreds = []
        torchModelLabelPreds = []
        ourLabelPredVals = []
        torchModelLabelPredVals =[]

        for file in os.listdir(directory):
            fullFile = os.path.join(directory,file)
            im = Image.open(fullFile)
            input = imageToTensor(im)

            ourLabelPreds.append(torch.argmax(model(input)))
            torchModelLabelPreds.append(torch.argmax(torch_model(input)))
            ourLabelPredVals.append(torch.max(model(input)))
            torchModelLabelPredVals.append(torch.max(torch_model(input)))

        # check if the label predictions are same
        self.assertEqual(ourLabelPreds,torchModelLabelPreds)

        # check if the label prediction values are same
        np.testing.assert_almost_equal(ourLabelPredVals,torchModelLabelPredVals,4)

class TestDiscrimNet(unittest.TestCase):
    '''
    To test the discriminator in DeepDreamGAN type object
    '''
    def test_discrimNetTraining(self):
        '''
        Test whether after training, the discriminator network
        of the DeepDreamGAN type object becomes better at  detecting
        a dream image tensor 
        '''
        # create a basic dreamer and dreamerGAN whose disciminator is not trained
        dreamer_plain = DeepDream()
        dreamerGAN_not_trained = DeepDreamGAN()
        
        # create a dreamerGAN whose discrkminator is already trained
        discrimNet = createDiscriminator()
        discrimNet.load_state_dict(torch.load('../discriminatorGAN_539.pth'))
        dreamerGAN_trained = DeepDreamGAN(discrimNet=discrimNet)

        labels = [71,130,407,430] #random labels to check
        
        #lists for activations of the dream image tensors input into the discrimNets
        activations_not_trained = []
        activations_trained = []

        for label in labels:
            im = dreamer_plain(label=label)
            x = dreamerGAN_not_trained.discrimNet(im)
            y = dreamerGAN_trained.discrimNet(im)

            activations_not_trained.append(x.cpu())
            activations_trained.append(y.cpu())
        
        # assert that activations for trained should be less than for untrained discriminator
        np.testing.assert_array_less(activations_trained,activations_not_trained)

        #lists for activations of plain dream and adverserially updated dream
        activations_plain_dream = []
        activations_GAN_dream = []
        for label in labels:
            im = dreamer_plain(label=label)
            x = dreamerGAN_trained.discrimNet(im)

            # do one adversarial loop on the im input hence making the tensor more 'realistic'
            im2 = dreamerGAN_trained(im=im,label=label)
            y = dreamerGAN_trained.discrimNet(im2)

            activations_plain_dream.append(x.cpu())
            activations_GAN_dream.append(y.cpu())

        #assert that plain dream activations should be less than GAN updated dream
        np.testing.assert_array_less(activations_plain_dream,activations_GAN_dream)
