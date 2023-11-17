import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import src.model as model_script
from pathlib import Path

# Function for the validation pass
def custom_validation(model, data_loader, criterion, device):
    model.eval() #De-activates dropout, batchnormalization layers
    # torch.no_grad #Disables the gradient calculation
    torch.set_grad_enabled(False) #Turn off the gradient calculation

    val_loss = 0
    accuracy = 0
    for i, (images, labels) in enumerate(data_loader):  

        images, labels = images.to(device), labels.to(device)
        outputs = model.forward(images)
        outputs_class = torch.argmax(outputs, dim=1)

        val_loss += criterion(outputs, labels).item()/len(labels)
        accuracy += (outputs_class == labels).float().sum().item()/len(labels)

        #probabilities = torch.exp(output) #If nn.NLLLoss is used for loss function
        #print('i:',i , '/', len(data_loader))
        if i==30:
                break
    torch.set_grad_enabled(True) #Turn ON the gradient calculation
    return val_loss/(i+1), accuracy/(i+1)
    #return val_loss/len(data_loader), accuracy/len(data_loader)
def save_model(model, full_filename):
    torch.save( model.state_dict(), full_filename)  
    print('Saving the model into the file:', full_filename )

def Main(dataclass_params, train_loader, test_loader):
    print('######### Start of training step ##### ')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device is ', device)
    
    learning_rate = dataclass_params.learning_rate
    weight_decay = dataclass_params.weight_decay
    batch_size = dataclass_params.batch_size
    num_epochs = dataclass_params.num_epochs
    model_name = dataclass_params.model_name
    save_model_flag = dataclass_params.save_model_flag
    save_model_filename = dataclass_params.save_model_filename
    save_model_filename_path = Path(save_model_filename)
    save_model_filename_path.parent.mkdir(exist_ok=True, parents=True)

    model = model_script.get_model(dataclass_params)
    print(model)
    model.to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_step = len(train_loader)
    print('Size of train_loader=', total_step)
    train_accuracy=0

    for epoch in range(num_epochs):
        train_accuracy=0    
        model.train() #Put the model in the training phase, activates dropout, batch-normalization layers
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            #print('i=', i, type(images), len(images), len(labels) ,labels )
            
            outputs = model(images)
            outputs_class = torch.argmax(outputs, dim=1)
            #print(type(outputs), len(outputs), ' pred output=', outputs, ' \n pred class=', torch.argmax(outputs, dim=1) )
            #print('i=', i, type(labels), len(labels) ,' ground_truth=' , labels)

            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #print('Batch: ',i,'/',total_step)
            train_accuracy += (outputs_class == labels).float().sum().item()/len(labels)
            #if i==30:
            #    break
            
        train_accuracy = train_accuracy/(i+1)
        print('End of epoch: ', epoch,'/',num_epochs, ', training accuracy=', train_accuracy)    
        
    val_loss, accuracy = custom_validation(model, test_loader, criterion, device)
    dataclass_params.train_accuracy = train_accuracy
    dataclass_params.test_val_loss = val_loss
    dataclass_params.test_accuracy = accuracy
    print(": The validation loss = ",val_loss,  ', accuracy=', accuracy)
    if save_model_flag == 'yes':
        save_model(model, save_model_filename)

    print('######### End of training step ##### ')
