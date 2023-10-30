import torch
from torchvision import datasets, transforms, models

def Main(dataclass_params):
    print('######### Start of data loader creation step ##### ')

    train_dir = dataclass_params.train_folder
    test_dir  = dataclass_params.test_folder
    predict_dir = dataclass_params.predict_folder
    batch_size = dataclass_params.batch_size
    
    # Define transforms for the training, testing and predict sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.Resize(32*4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.3, 0.3, 0.3])])

    test_predict_transforms = transforms.Compose([transforms.Resize(32*4),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5],
                                                                 [0.3, 0.3, 0.3])])


    # Create datasets using datasets.ImageFolder,
    # Input is root directory of data, transforms takes RGB PIL Images and returns transformed version
    # By default the imageFolder loads images with 3 channel
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=test_predict_transforms)
    #predict_dataset = datasets.ImageFolder(predict_dir, transform=test_predict_transforms)
    predict_dataset = test_dataset
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)
    #predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size)
    predict_loader = test_loader

    print('######### End of data loader creation step ##### ')

    return train_loader, test_loader, predict_loader
