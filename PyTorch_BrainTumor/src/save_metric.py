import os
from pathlib import Path
#import txt2pdf 
#import textwrap 
#from fpdf import FPDF  
def Main(dataclass_params):
    keys = list(dataclass_params.__dict__.keys())
    values = list(dataclass_params.__dict__.values())

    dir  = os.getcwd() + '/' + dataclass_params.save_model_dir
    file = dir + '/' + dataclass_params.save_parameters_file
    filename_path = Path(file)
    filename_path.parent.mkdir(exist_ok=True, parents=True)
    text_file  =  file + '.txt'
    f = open(text_file,'w')
    print('Amit:', keys[0])
    for i in range(len(keys)):
        line =  str(keys[i]) +  ' : ' + str(values[i]) + '\n'
        print(line)
        f.write(line)
    f.close()    
