import os
import yaml
from pathlib import Path
import src.initialize
import src.data_download
import src.preprocess
import src.train

yaml_input = "params.yaml"

#Run Initialization
yaml_input_fullname= Path.cwd() / yaml_input
print(yaml_input_fullname)
dataclass_params = src.initialize.Main(yaml_input_fullname)

# Run download_data_module
src.data_download.Main(dataclass_params)

#Run Preprocessing module
train_loader, test_loader, predict_loader = src.preprocess.Main(dataclass_params)

#Run the training part
src.train.Main(dataclass_params, train_loader, test_loader)
