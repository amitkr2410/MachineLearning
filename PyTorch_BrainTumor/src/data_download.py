import os
import yaml
from pathlib import Path
import pandas as pd
#import opendatasets as od
import numpy as np
import shutil
def Download_Data(path):
    # Download the data from Kaggle
    # Read this page to learn how to download data from Kaggle
    # https://www.geeksforgeeks.org/how-to-download-kaggle-datasets-into-jupyter-notebook
    #Alternnatively, you can use the code below to directly download the data
    #using your kaggle username and kaggle api key
    #import opendatasets as od
    #od.download("https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection")           
    #os.listdir('.')
    #os.listdir('brain-tumor-detection')    
    #command = 'mv brain-tumor-detection ' + path
    #os.system(command)

    command = 'cd ' + path + ' ; ' + ' gdown --id 1FafUyXmrTMUbLyKTMeKZ6BYBkpDi4OjO '
    command = command + ' ; ' + ' unzip brain-tumor-detection.zip '
    os.system(command)

#Create directory for train, test and validation data set
#NameType_of_data = "train", "test", "validation"
def generate_data_dir(input_dir, output_dir, test_size ):
    #ROOT_DIR='brain-tumor-detection'
    input_dir_yes = input_dir + '/yes'
    input_dir_no = input_dir + '/no'
    tumor_neg = os.listdir(input_dir_no)
    tumor_pos = os.listdir(input_dir_yes)
    #ROOT_DIR2 = os.path.join(ROOT_DIR, NameType_of_data)
    if not os.path.exists(Path(output_dir)):
        os.makedirs(os.path.join(output_dir,"yes"))
        os.makedirs(os.path.join(output_dir,"no"))
    else:
        print(output_dir," folder exists")
    
    print('All files (yes,no) are ', len(tumor_pos),len(tumor_neg))
    #print(tumor_neg)
    #a=[4,10,3.3, 4,6]
    tumor_neg_cut = np.random.choice(tumor_neg,size=int(len(tumor_neg)*test_size), replace=False)
    tumor_pos_cut = np.random.choice(tumor_pos,size=int(len(tumor_pos)*test_size), replace=False)

    print('New size of (yes,no) is ',len(tumor_pos_cut),len(tumor_neg_cut))
    for file in tumor_pos_cut:
        File1=input_dir_yes + '/' +file
        File2=output_dir + "/yes/" + file
        shutil.copy(File1, File2)

    for file in tumor_neg_cut:
        File1=input_dir_no + "/" + file
        File2= output_dir + "/no/" + file
        shutil.copy(File1, File2)
  
    mylist = os.listdir(os.path.join(output_dir,"yes"))
    print('Few files inside output_dir with tumor status yes are ',mylist[0:5])
    mylist = os.listdir(os.path.join(output_dir,"no"))
    print('Few files inside output_dir with tumor status no are ', mylist[0:5])
        

    
def Main(dataclass_params):
    print('######### Start data download step ##### ')

    main_dir= dataclass_params.parent_dir
    test_size = dataclass_params.test_size

    # Download_Data(main_dir)
    
    #Check if the data directory is empty
    datafolder = Path(main_dir , 'data/')
    if not os.path.exists(datafolder):
      command = 'mkdir ' + str(datafolder)
      os.system(command)

    if not os.listdir(datafolder):
        Download_Data(str(datafolder))
          
    allfiles = Path(main_dir , 'data/', 'brain-tumor-detection/')
    
    train   = dataclass_params.train_folder
    test    = dataclass_params.test_folder
    predict = dataclass_params.predict_folder

    if not os.path.exists(train):
      generate_data_dir( str(allfiles), str(train), 1.0 - test_size)    
    if not os.path.exists(test):
      generate_data_dir(str(allfiles), str(test), test_size )
    if not os.path.exists(predict):
      generate_data_dir(str(allfiles), str(predict), test_size )
    
    print('######### End of the data download step ##### ')
