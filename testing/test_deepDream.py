import sys
sys.path.append('../lib/')
import unittest
from deepDream import *

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
