Brain Tumor detection project using PyTorch

Goal: The goal of this project is to build CT scan tumor detection app using Self Attention module and compare the performance with traditional VGG16 architecture. We shall deploy the predictive model on AWS platform, google cloud run and google Kubernet engine.


1. How to run the code on the terminal 
    git clone https://github.com/amitkr2410/MachineLearning.git
    cd MachineLearning/PyTorch_BrainTumor
    python master.py 

2. Run on Google Cloud (Collab): Google Compute Engine T4 GPU
    a. Upload the code "PyTorch_BrainTumor" on google drive
    b. Open google collab jupyter notebook and Mount the code "PyTorch_BrainTumor"
    c. Connect to gpu machine 
    d. execute the following command on python cell:
       !cd  /content/drive/MyDrive/PyTorch_BrainTumor ; python master.py
    In above, I assume that we uploaded  "PyTorch_BrainTumor" directory in "MyDrive" of google drive
3.  File Description
    master.py:
    params.yaml:
    TerminalOutput: It is directory where we can store the terminal output. Useful for checks and verification
    src/data_download.py: 
    src/initialize.py:
    src/parameters.py:
    src/preprocess.py:
    src/model_attention.py:
    src/model.py:
    src/train.py:
    src/save_metric.py:

    plots/EpochVsTrainAccuracy.py:
    plots/LearningRateVsAccuracy.py:

4. Results: Attach the two plots
5. Model architecture:
   (a) Self attention with positional encoding:
   (b) VGG16:
   (c) 4 layers CNN + self attention module with positional encoding:
   (d) 4 layers CNN
6. Stages involved in buidling end-to-end application
    (1) Define data pipelines and input parameter file
    (2) Data download
    (3) Preprocessing and Data Augmentation
    (4) Define model
    (5) Train the model
    (6) Model validation
    (7) Save the model and run parameters
    (8) Host the model on the Cloud    

Data Source: Kaggle CT Scane image data set
                      https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

Data Augmentation: We used transform and datasets objects from torchvision library. We created         
                                 dataloaders with batches of images using torch library

Model training: We explored hyper-parameters: 
                            Learning rate = {0.01, 0.001, 0.0001, 0.00001} 
                            Weight decay = {0.002}
                            The VGG16 model performed with the highest accuracy of 97%

The full model exploration is shown below: