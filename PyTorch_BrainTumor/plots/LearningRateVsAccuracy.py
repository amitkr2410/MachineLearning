import matplotlib as mpl
import matplotlib.pyplot as plt
import plotsettings



def GetXY(RunNumbers):
    learning_rate =	[]
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
    return learning_rate, train_accuracy, test_accuracy
        
if __name__ == '__main__':
    #SetDefault plot settings
    plotsettings.configure(mpl)
    #For SelfAttention with positional encoding
    RunNumbers=['150', '151', '152', '153']
    X0_train, Y0_train, Y0_test = GetXY(RunNumbers )

    #for VGG16 architecture
    RunNumbers=['48', '49', '50', '51']    
    X1_train, Y1_train,  Y1_test = GetXY(RunNumbers )
    print(' ')

    #Run100: cnn_with_attention_Run100
    RunNumbers=['100', '101', '102', '103']
    X2_train, Y2_train, Y2_test = GetXY(RunNumbers)
    
    #for 4-layers CNN
    RunNumbers=['31', '21', '12', '0']
    X3_train, Y3_train, Y3_test = GetXY(RunNumbers)

    fig, ax = plt.subplots(1,1, figsize= (10,8))
    style1 = dict(marker='o', markersize=10 )
    style2 = dict(marker='o', markersize=10, markerfacecolor='none' )

    ax.plot(X0_train, Y0_train, color='red', linestyle='solid', **style1, label='SelfAttention + Position Enc [Train set]')
    ax.plot(X0_train, Y0_test, color='red', linestyle='dashed', **style2, label='SelfAttention + Position Enc [Test set]')
    ax.plot(X1_train, Y1_train, color='blue', linestyle='solid', **style1, label='VGG16 [Train set]')
    ax.plot(X1_train, Y1_test, color='blue', linestyle='dashed', **style2, label='VGG16 [Test set]')
    ax.plot(X2_train, Y2_train, color='green', linestyle='solid', **style1, label='4 layers CNN + SelfAttention with Position Enc [Train set]')
    ax.plot(X2_train, Y2_test, color='green', linestyle='dashed', **style2, label='4 layers CNN + SelfAttention with Position Enc [Test set]')
    ax.plot(X3_train, Y3_train, color='purple', linestyle='solid', **style1, label='4 layers CNN [Train set]', alpha=0.5)
    ax.plot(X3_train, Y3_test, color='purple', linestyle='dashed', **style2, label='4 layers CNN [Test set]', alpha=0.5)

    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xscale('log')
    ax.set_ylim(50, 100)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
    plt.legend(loc='lower left')
    plt.tight_layout()
    #plt.show()
    plt.savefig('figure_learning_rate_VGG16.png')
