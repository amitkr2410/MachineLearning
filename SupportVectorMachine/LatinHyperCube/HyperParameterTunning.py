import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import preprocess as preproc
import LatinHyperCubeSampling as lhs
import train as tr
import evaluation as eval
import os
mpl.rcParams['savefig.format'] = "pdf"

column=['all', 'independent']
filename = "../cell_samples_for_svm.csv"

model_id=[]
kernel=[]
gamma=[]
C=[]
cv=[]
Feature=[]
train_score=[]
#test_score= []
df_tune = pd.DataFrame()
best_score=0
best_model_index=0
model_index=-1
DataTag=[]

for i in range(0,len(column)):
    data = preproc.ProcessDataFrame(filename, column[i])
    print("Size of Train and test is ", len(data) )
    DataTag.append(data)
    X_train, X_test, y_train, y_test = data
    Parameters = lhs.DesignPoint(column[i])
    Kernels = ['linear', 'rbf', 'sigmoid']
    for KernelName in Kernels:
        for j in range(0,len(Parameters[:,0])):
            mean_score = tr.Train(KernelName, Parameters[j,:], X_train, y_train)
            model_index = model_index + 1
            if best_score < mean_score:
                best_score = mean_score
                best_model_index = model_index

            model_id.append(model_index)    
            kernel.append(KernelName)
            gamma.append(Parameters[j,0])
            C.append(Parameters[j,1])
            cv.append(5)
            Feature.append(column[i])
            train_score.append(mean_score)
            
#Create a data frame with Hyperparameters tuning data
df_tune = pd.DataFrame()
df_tune['model_index'] = model_id
df_tune['kernel'] = kernel
df_tune['gamma'] = np.round(gamma, 2)
df_tune['C'] = np.round(C, 2)
df_tune['cv'] = cv
df_tune['feature'] = Feature
df_tune['train_score'] = np.round(train_score, 2)

#Evaluate Best parameters
BestParams = [df_tune.loc[best_model_index,'gamma'], df_tune.loc[best_model_index,'C'] ]
if df_tune.loc[best_model_index, 'feature'] == column[0]:
    X_train, X_test, y_train, y_test = DataTag[0]
if df_tune.loc[best_model_index, 'feature'] == column[1]:
    X_train, X_test, y_train, y_test = DataTag[1]
print("For best parameters, size of Train and test is ", len(DataTag[0]) )

test_score = eval.Eval(df_tune.loc[best_model_index,'kernel'],
                       BestParams, X_train, X_test, y_train, y_test)



#Show the Hyperparameter using pdf file 
fig, ax =plt.subplots(figsize=(12,8))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=df_tune.values, colLabels=df_tune.columns, loc='center')
plt.tight_layout()

OutputFilename='ExplorationTable.pdf'    
plt.savefig(OutputFilename)
Command="open " + " "+OutputFilename
os.system(Command)
