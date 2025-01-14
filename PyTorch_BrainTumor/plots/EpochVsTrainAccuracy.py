import matplotlib as mpl
import matplotlib.pyplot as plt
import plotsettings



def GetXY(RunNumbers):
    epoch = []
    train_accuracy = []
    val_loss = []
    validation_accuracy = []
    
    for Run in RunNumbers:
        filename = '../TerminalOutput/ScreenOutputRun' + Run + '.txt'
        f = open(filename,'r')
        lines = f.readlines()
        for line in lines:
            line_elements = line.split(':')
            #print(line_elements)
            if(len(line_elements) <=1):
                continue
            #print(line_elements)
            
            if 'End of epoch' in line_elements:
                #print(line_elements)
                epoch_info = line_elements[1].split('/')
                epoch_string = epoch_info[0]
                accuracy_info = epoch_info[1].split('=')
                train_string    = accuracy_info[1].split(',')[0]
                val_loss_string = accuracy_info[2].split(',')[0]
                val_string      = accuracy_info[3].split(',')[0]
                
                print(epoch_string, train_string, val_loss_string, val_string)
                epoch.append(int(epoch_string))
                train_accuracy.append(float(train_string)*100)
                val_loss.append(float(val_loss_string))
                validation_accuracy.append(float(val_string)*100)
                
    print('Run number is ', RunNumbers)
    print('epoch is ', epoch)
    print('train_accuracy is ', train_accuracy)
    print('val_loss is ', val_loss)
    print('val accuracy is ', validation_accuracy)
    return epoch, train_accuracy, validation_accuracy
        
if __name__ == '__main__':
    #SetDefault plot settings
    plotsettings.configure(mpl)

    #Run151:  only_attention_Run151
    RunNumbers=['151']
    X0_train, Y0_train, Y0_valid = GetXY(RunNumbers )

    #Run49: vgg16_pretrained_false_Run49 (for VGG16 architecture)
    RunNumbers=[ '49']    
    X1_train, Y1_train, Y1_valid = GetXY(RunNumbers )
    print(' ')

    #Run101: cnn_with_attention_Run101
    RunNumbers=[ '101']
    X2_train, Y2_train, Y2_valid = GetXY(RunNumbers)
    
    #Run21: cnn_4layers_results_Run21 (for 4-layers CNN)
    RunNumbers=[ '21']
    X3_train, Y3_train, Y3_valid = GetXY(RunNumbers)
    
    fig, ax = plt.subplots(1,1, figsize= (10,8))
    style1 = dict(marker='o', markersize=10 )
    style2 = dict(marker='o', markersize=10, markerfacecolor='none' )
    
    ax.plot(X0_train, Y0_train, color='red', linestyle='solid', **style1, label='SelfAttention + Position Enc [Train set]')
    ax.plot(X0_train, Y0_valid, color='red', linestyle='dashed', **style2, label='SelfAttention + Position Enc [Valid set]')
    ax.plot(X1_train, Y1_train, color='blue', linestyle='solid', **style1, label='VGG16 [Train set]')
    ax.plot(X1_train, Y1_valid, color='blue', linestyle='dashed', **style2, label='VGG16 [Valid set]')
    ax.plot(X2_train, Y2_train, color='green', linestyle='solid', **style1, label='4 layers CNN + SelfAttention with Position Enc [Train set]')
    ax.plot(X2_train, Y2_valid, color='green', linestyle='dashed', **style2, label='4 layers CNN + SelfAttention with Position Enc [Valid set]')
    ax.plot(X3_train, Y3_train, color='purple', linestyle='solid', **style1, label='4 layers CNN [Train set]', alpha=0.5)
    ax.plot(X3_train, Y3_valid, color='purple', linestyle='dashed', **style2, label='4 layers CNN [Valid set]', alpha=0.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Model Accuracy (%)')
    ax.set_ylim(60, 100)
    ax.set_xlim(-0.5, 20)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
    
    #ax.set_xscale('log')
    plt.legend(loc='lower right')
    plt.tight_layout()
    #plt.show()
    plt.savefig('figure_epochVsAccuracy_VGG16.png')
