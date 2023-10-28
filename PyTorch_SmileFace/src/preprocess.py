import os
import yaml
from pathlib import Path
import pandas as pd

def Download_Data(parent_dir):
    path = parent_dir + '/data/'
    print(path)
    command = 'mkdir ' + path
    os.system(command)
    command = 'cd ' + path
    os.system(command)
    command = 'cd ' + path + ';' + ' gdown --id 1Q6OJ2PVzmUi0RSDqnC1C8rf8-wCcqO3o'
    #os.system(command)
    #command = 'mv data_participant.csv'  + path + "/"
    #command ='rm data_participant.csv'
    command = command + ';' + ' gdown --id 1cirJQi9ssDONxaxp9ZZvJcjrv3tnhk8Y'
    #os.system(command)
    command = command + ';' + ' unzip files.zip '
    os.system(command)

def create_panda_dataframe(filename):
    df = pd.read_csv(filename, delimiter=',')
    df_main = df.loc[ (df['label'].notna() ),: ].reset_index(drop=True)

    from sklearn.preprocessing import LabelEncoder    
    le = LabelEncoder()
    LabelEncodedY = le.fit_transform(df_main['label'])
    df_main['class'] = LabelEncodedY
    print(df_main.head())

    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df_main, test_size=0.2)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_predict = df.loc[ (df['label'].isnull() ),: ].reset_index(drop=True)
    
    return df_train, df_test, df_predict
    
def generate_data_dir(dir_input, dir_out, df_temp):
  #command = 'rm data'
  #os.system(command)
  class_label = []
  class_numeric=[]

  command = 'mkdir  ' + dir_out
  os.system(command)
  total_images = df_temp.shape[0]
  for i in range(total_images):
    filenameIn= dir_input + '/' + df_temp.loc[i, 'filename']
    if str(df_temp['label'].iloc[i]) == 'nan':
      filenameOut= dir_out + '/' + df_temp.loc[i, 'filename']
      command = 'cp ' +  filenameIn + '  ' + filenameOut
      print('Name of files: ',command)
      os.system(command)
    else:
      if df_temp.loc[i, 'label'] not in class_label:
        command = 'mkdir ' +  dir_out + '/' + str(df_temp.loc[i,'class']) + '/'
        os.system(command)
        class_label.append(df_temp.loc[i, 'label'])
        class_numeric.append(df_temp.loc[i, 'label'])
      filenameOut = dir_out + '/' + \
                    str(df_temp.loc[i,'class']) + '/' + df_temp.loc[i, 'filename']
      command = 'cp ' +  filenameIn + '  ' + filenameOut
      print(command)
      os.system(command) 

    
def Main(dataclass_params):
    main_dir= dataclass_params.parent_dir
    # Download_Data(main_dir)
    
    #Check if the data directory is empty
    datafolder = Path(main_dir , 'data/')
    
    if not os.listdir(datafolder):
        Download_Data(str(main_dir))

    csv_file_name = str(datafolder) + '/data_participant.csv'
    df_train, df_test, df_predict = create_panda_dataframe(csv_file_name)
    
    allfiles = Path(main_dir , 'data/', 'files/')
    
    train   = dataclass_params.train_folder
    test    = dataclass_params.test_folder
    predict = dataclass_params.predict_folder
        
    generate_data_dir(str(allfiles), str(train), df_train)
    generate_data_dir(str(allfiles), str(test), df_test)
    generate_data_dir(str(allfiles), str(predict), df_predict)
