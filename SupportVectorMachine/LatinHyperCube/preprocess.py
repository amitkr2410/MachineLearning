#Import library
import pandas as pd    #to manipulate data in csv in row and column format
import numpy as np     #
import matplotlib as mpl
import matplotlib.pyplot as plt #
import pickle, os
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
mpl.rcParams['savefig.format'] = "pdf"

def ProcessDataFrame(dfname, column='all'):
    #df = pd.read_csv('cell_samples_for_svm.csv')
    df = pd.read_csv(dfname)
    print("The data is for case:", column, " column")
    print(df.head())
    print("df shape is ", df.shape)
    print("df Columns count=\n", df.count())
    print("df mode = \n", df['Class'].value_counts())
    print("df data types= \n", df.dtypes)

    #Check for null values
    print( "df null value summary: \n", df.isnull().sum() )
    
    #Make plots
    #From above we find there are no-null entry in the data set
    #Also, we note that column='BareNuc' is non-numerical !!!
    #Visualize Size and shape of benign and malignant cells
    benign_df = df[df['Class']==2]
    malignant_df = df[df['Class']==4]
    axes_0 = benign_df.plot( x='Clump',y='UnifSize', 
                             color='blue', marker='o', markersize=10,linewidth=0,
                             label='Benign')
    malignant_df.plot( x='Clump',y='UnifSize', linewidth=0,
                       color='red', marker='v',  markersize=15, markerfacecolor='none', label='Malignant', ax=axes_0)
    axes_0.set_ylim(0,15)
    axes_0.set_xlim(0,12)
    
    #Columns 'BareNuc' is non-numerical data
    df_new = df[ pd.to_numeric(df['BareNuc'], errors='coerce').notnull() ].copy()
    df_new['BareNuc'] = df_new['BareNuc'].astype('int')
    df_new = df_new.iloc[:,1:]
    print(" Converted BareNucl column to numerical type, \
     and dropped ID column, so new df is  : \n", df_new.dtypes)

    #See range of variables
    print( "Summary of df: \n", df_new.describe() )
    
    ColumnLabels = df_new.columns
    ColumnLabelsList =list(ColumnLabels)
    # Pearson correlation between variables including Y
    if column=='all':
        fig0, axs0 = plt.subplots(1, 1, figsize=(6,5),gridspec_kw={'hspace': 0, 'wspace': 0} )
        Pearson_matrix = df_new[ColumnLabelsList].corr()
        sb.heatmap( Pearson_matrix, cmap="PiYG",vmin=-1, vmax=1, annot=True, fmt=".1f")
        plt.tight_layout()

        OutputFilename='PearsonMatrix_' + column +'.pdf'
        plt.savefig(OutputFilename)
        Command="open " + " "+OutputFilename
        os.system(Command)
    #list(ColumnLabels)

    

    Nmax = len(ColumnLabelsList)
    if column=='all':
        fig, axs = plt.subplots(Nmax,Nmax, figsize=(14,10))
        for i in range(Nmax):
            for j in range(0,Nmax):
                if j <=i:
                    sb.scatterplot( x=df_new[ColumnLabelsList[j]], y=df_new[ColumnLabelsList[i]], ax=axs[i,j] )
                    axs[i,j].spines[['right', 'top']].set_visible(False)
                else: 
                    axs[i,j].spines[['right', 'top', 'left', 'bottom']].set_visible(False)
                    axs[i,j].set(ylabel=None)
                    axs[i,j].set(xlabel=None)
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])
                    
                if j != 0 and i != Nmax-1: 
                    axs[i,j].set(ylabel=None)
                    axs[i,j].set(xlabel=None)
                    axs[i,j].set(xticklabels=[])
                    axs[i,j].set(yticklabels=[])
                if j!=0 and i == Nmax-1:
                    axs[i,j].set(ylabel=None)
                    axs[i,j].set(yticklabels=[])
                if j==0 and i != Nmax-1:
                    axs[i,j].set(xlabel=None)
                    axs[i,j].set(xticklabels=[])
                    
        plt.tight_layout()
        OutputFilename='Correlations_' + column +'.pdf'
        plt.savefig(OutputFilename)
        Command="open " + " "+OutputFilename
        os.system(Command)

    #Label encoding for Y
    
    #df_r = df_new.copy()
    le = LabelEncoder()
    LabelEncodedY = le.fit_transform(df_new['Class'])
    df_new['Class'] = LabelEncodedY
                
    #from sklearn.model_selection import train_test_split
    #Create training data set by converting pandas data frame into numpy arrary
    #which are input to SVM function
    #Select all columns except ID and class
    df_x = df_new.iloc[:,0:-1]
    if column !='all':
        df_x = df_new.loc[:,df_new.columns!='UnifSize']

    print("X df is \n ", df_x.head())
    
    # independent variables as numpy array
    X = np.asarray(df_x)
    print("X array is \n", X)
    
    #dependent variables as numpy array
    y = np.asarray(df_new['Class'])
    
    #Import library for splitting data sets into training and test set
    #from sklearn.model_selection import train_test_split
    
    #Training data set X_train, y_train
    #Test data set X_test, y_test
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4)
    print("X train shape=", X_train.shape)
    print("X test shape=", X_test.shape)
    print(type(X))
    print(type(y), '\n')
    print("dType (X train, y test) = ", type(X_train), type(y_test))

    return X_train, X_test, y_train, y_test

