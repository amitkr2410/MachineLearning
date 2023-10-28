import os
import yaml
from pathlib import Path
import src.initialize
import src.preprocess
#import src.train

yaml_input = "params.yaml"

#Run Initialization
yaml_input_fullname= Path.cwd() / yaml_input
print(yaml_input_fullname)
dataclass_params = src.initialize.Main(yaml_input_fullname)

#Run Preprocessing module
src.preprocess.Main(dataclass_params)

#Run the training part
src.train.Main(dataclass_params)
