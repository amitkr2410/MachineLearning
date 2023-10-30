import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import src.model as model_script

# Function for the validation pass
def custom_validation(model, data_loader, criterion, device):
    val_loss = 0
    accuracy = 0
    for images, labels in iter(data_loader):
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        val_loss += criterion(output, labels).item()
        probabilities=output
        #probabilities = torch.exp(output) #If nn.NLLLoss is used for loss function
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return val_loss, accuracy

def Main(dataclass_params, train_loader, test_loader):
    print('######### Start of training step ##### ')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device is ', device)
    
    learning_rate = dataclass_params.learning_rate
    weight_decay = dataclass_params.weight_decay
    batch_size = dataclass_params.batch_size
    num_epochs = dataclass_params.num_epochs
    model_name = dataclass_params.model_name

    model = model_script.get_model(dataclass_params)
    print(model)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_step = len(train_loader)
    '''
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    '''
    print('######### End of training step ##### ')
