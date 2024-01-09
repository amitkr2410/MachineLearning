import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import src.model as model_script
from pathlib import Path
import os
import PIL

# Function for the validation pass
def Transform_Image(input_image):
    test_predict_transforms = transforms.Compose([transforms.Resize(32*4),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5],
                                                                 [0.3, 0.3, 0.3])])

    pil_input_image = PIL.Image.open(input_image, mode='r').convert('RGB')
    image = test_predict_transforms(pil_input_image)
    return image

def Make_prediction(model, input_image, device):
    model.eval() #De-activates dropout, batchnormalization layers
    #Since the model expects an input with 4 dimension 
    # .. which corresponds to BxCxHxW =(Batch x Channel X Height X Width) 
    # ..we need to add one more dimension. As we are testing with one image, we are missing
    # ..the Batch (B) dimension
    # To solve this, we can add this dimension by using unsqueeze
    valid_image = input_image.unsqueeze(0) 
    valid_image = valid_image.to(device)
    print('image:' , type(valid_image), valid_image)
    outputs = model.forward(valid_image)
    print(outputs)
    outputs_class = torch.argmax(outputs, dim=1)
    outputs_class = outputs_class.item() #Convert torch tensor into scalar number
    #outputs_class =0
    return outputs_class

if __name__ == '__main__':
    print('######### Start of prediction step ##### ')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device is ', device)
    cwd = os.getcwd()
    model_filename = cwd + '/final_model/cnn_with_attention_Run.pth'
    #input_image_name = cwd + '/data/test/yes/y0.jpg'
    #input_image_name = cwd + '/data/test/yes/y10.jpg'
    input_image_name = cwd + '/data/test/no/no30.jpg'
    #input_image_name = cwd + '/data/predict/822496.jpg'

    num_classes=2
    model = model_script.model_attention.get_cnn_with_attention(num_classes)
    model.eval()
    model.load_state_dict(torch.load(model_filename))
    print(model)
    
    input_image = Transform_Image(input_image_name)
    ans_class = Make_prediction(model, input_image, device)
    print('The input image corresponds to the following class:', ans_class)

    print('######### End of prediction step ##### ')
