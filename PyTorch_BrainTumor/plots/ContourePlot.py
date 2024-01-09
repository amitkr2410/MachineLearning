import matplotlib as mpl
import matplotlib.pyplot as plt
import plotsettings
import scipy.interpolate as ml
import numpy as np
import seaborn as sns
import pandas as pd

def GetXY(RunNumbers):
    learning_rate = []
    weight_decay = []
    train_accuracy = []
    test_accuracy = []
    for Run in RunNumbers:
        filename = '../TerminalOutput/ScreenOutputRun' + Run + '.txt'
        f = open(filename,'r')
        lines = f.readlines()
        for line in lines:
            line_elements = line.split(' ')
            #print(line_elements)
            if(len(line_elements) <=1):
                continue
            #print(line_elements)
            
            if 'learning_rate' in line_elements:
                learning_rate.append(float(line_elements[2].split('\n')[0]))
            if 'weight_decay' in line_elements:
                weight_decay.append(float(line_elements[2].split('\n')[0]))
            if 'train_accuracy' in line_elements:
                train_accuracy.append(float(line_elements[2].split('\n')[0])*100)
            if 'test_accuracy' in line_elements:
                test_accuracy.append(float(line_elements[2].split('\n')[0])*100)
            
    print('Run number is ', RunNumbers)
    print('learning_rate is ', learning_rate)
    print('weight_decay is ', weight_decay)
    print('train_accuracy is ', train_accuracy)
    print('test_accuracy is ', test_accuracy)
    return learning_rate, weight_decay, train_accuracy, test_accuracy
        
if __name__ == '__main__':
    #SetDefault plot settings
    plotsettings.configure(mpl)
    
    #for VGG16 architecture
    RunNumbers=['48', '49', '50', '51']    
    X1, Y1, Z1, Z10 = GetXY(RunNumbers )
    XG1, YG1 = np.meshgrid(X1,Y1)
    #ZG1 = ml.griddata((X1, Y1), Z1, (XG1, YG1))
    #ZG10 =  ml.griddata(X1, Y1, Z10, XG1, YG1)
    
    #for 4-layers CNN
    RunNumbers=['31', '21', '12']
    X2, Y2, Z2, Z22 = GetXY(RunNumbers)

    
    fig, ax = plt.subplots(1,1, figsize= (10,5))
    style1 = dict(marker='o', markersize=10 )
    style2 = dict(marker='o', markersize=10, markerfacecolor='none' )
    #ax.contour(XG1, YG1, ZG1, colors='k')
    #plt.hist2d(X1, Y1, bins=4, weights=Z1, cmap='Greys')
    #plt.colorbar()
    ax.hist2d(X1, Y1, c=Z1, cmap='viridis')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.colorbar() 
    '''
    ax.plot(X1_train, Y1_train, color='red', linestyle='solid', **style1, label='VGG16 [Train set]')
    ax.plot(X1_test, Y1_test, color='blue', linestyle='solid', **style1, label='VGG16 [Test set]')
    ax.plot(X2_train, Y2_train, color='red', linestyle='dashed', **style2, label='4 layers CNN [Train set]')
    ax.plot(X2_test, Y2_test, color='blue', linestyle='dashed', **style2, label='4 layers CNN [Test set]')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xscale('log')
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
    '''
    #plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
    #plt.savefig('figure_.png')
