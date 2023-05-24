from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

def Train(KernelName, Parameters, X,y):
    #X_train =  np.asarray(df_train.iloc[:,:-1])
    #y_train = np.asarray(df_train.iloc[:,-1])
    #print(y_train)
    Gamma, c0 = Parameters
    if KernelName =="linear":
        classifier = svm.SVC(kernel=KernelName, gamma=Gamma, C=c0, probability=True)
    if KernelName == "poly":
        classifier = svm.SVC(kernel=KernelName, degree=poly_deg, probability=True)
    if KernelName == "rbf":
        classifier = svm.SVC(kernel=KernelName, gamma=Gamma, C=c0, probability=True)
    if KernelName == "sigmoid":
        classifier = svm.SVC(kernel=KernelName, gamma=Gamma, C=c0, probability=True)
        # penality parameter=C and gamma are tunning parameters                                                     
    #classifier.fit(X_train, y_train)
    scores = cross_val_score(classifier, X, y, cv=5)
    print('Score=', scores)
    mean_score = np.sum(scores)/len(scores)
    print('Mean score=', mean_score)

    return mean_score
