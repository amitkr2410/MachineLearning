import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from src.parameters import AllParameters

def Main(filename : Path) -> AllParameters:
    print('######### Start initializing parameters ##### ')
    UpdatedParameters = AllParameters()
    print(UpdatedParameters.__dict__.keys())
    print(UpdatedParameters.__dict__.values())
    
    params = yaml.safe_load( open(filename) )["preprocess"]
    for p in params:
        print(p)
        setattr(UpdatedParameters, p, params[p])
        
    params = yaml.safe_load( open(filename) )["train"]
    for	p in params:
        print(p)
        setattr(UpdatedParameters, p, params[p])

    UpdatedParameters.parent_dir = Path(UpdatedParameters.parent_dir)
    UpdatedParameters.train_folder = Path(UpdatedParameters.parent_dir, UpdatedParameters.train_folder)
    UpdatedParameters.test_folder = Path(UpdatedParameters.parent_dir, UpdatedParameters.test_folder)
    UpdatedParameters.predict_folder = Path(UpdatedParameters.parent_dir, UpdatedParameters.predict_folder)

    #setattr(UpdatedParameters, '', )
    print(UpdatedParameters.__dict__.values())

    print('######### Done initializing parameters ##### ')

    return UpdatedParameters



