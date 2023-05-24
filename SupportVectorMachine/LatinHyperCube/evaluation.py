from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
mpl.rcParams['savefig.format'] = "png"

def Eval(KernelName, Parameters, X_train, X_test, y_train, y_test):
    Gamma, c0 = Parameters
    if KernelName =="linear":
        classifier = svm.SVC(kernel=KernelName, gamma=Gamma, C=c0, probability=True)
    if KernelName == "poly":
        classifier = svm.SVC(kernel=KernelName, degree=poly_deg, probability=True)
    if KernelName == "rbf":
        classifier = svm.SVC(kernel=KernelName, gamma=Gamma, C=c0, probability=True)
    if KernelName == "sigmoid":
        classifier = svm.SVC(kernel=KernelName, gamma=Gamma, C=c0, probability=True)

    print("Eval: Shape of Xtrain Xtest is ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)    
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    accuracy = EvaluateModel(y_test, y_predict)
    print("The final model accuracy is ", accuracy*100,'\%')
    return accuracy

def EvaluateModel(y_test, y_predict):
    print( classification_report(y_test, y_predict) )
    cnf_mat = confusion_matrix(y_test, y_predict, labels=[0,1])
    print(cnf_mat)
    plot_confusion_matrix(cnf_mat, LabelS=['Benign(0)','Malignant(1)'])
    accuracy = accuracy_score(y_test, y_predict)
    return accuracy

#define a function to plot 2x2 confusion matrix
def plot_confusion_matrix(cnf_mat, LabelS, cmap=plt.cm.Blues):
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    im = plt.imshow(cnf_mat, interpolation='nearest', cmap=cmap)
    #plt.colorbar()
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(LabelS))
    plt.xticks(tick_marks, LabelS, rotation=45)
    plt.yticks(tick_marks, LabelS)
    for i  in range(0,2):
        for j in range(0,2):
            plt.text(j,i,cnf_mat[i][j])

    plt.tight_layout()
    OutputFilename='ConfusionMatrix.png'
    plt.savefig(OutputFilename)
    Command="open " + " "+OutputFilename
    os.system(Command)
