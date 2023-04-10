### Here, we apply statistical methods to understand features/correlations present in the data
    The projects can be viewed by clicking on the file with extension "*.ipynb" (JupyterNotebook)
    
### Numerical regression and classification models: 
### (1) CNNTensorFlow- Deep learning 
    Classify brain tumor 2D scanned images into tumorus or healthy 
    sample using Convolutional neural network.
    We use keras library and train the model using 'Sequential method'. 
    The original images of brain are converted into a matrix data using
    function "ImageDataGenerator" defined in keras library. 
    We study the accuracy at each epoch of the model training. In the end, 
    we test the model againt known data and compute the confusion matrix 
    to determine the validity of the model.
    To view the project click on "main_cnn.ipynb" or "main_cnn_resnet50.ipynb" 
    
### (2) Support vector Machine- SVM 
    Introduction to SVM (Supervised machine learning technique) 
    Useful for classification, Regression, Outlier/Anomaly detection 
    Relevant Library to Import 
    Modeling Cancer data to predict whether a cell sample is benign or malignant.
    To view the project click on "main_svm.ipynb"
    
### (3) BankCreditCardChurn data
     We apply statistical models: 
     (a) Decision trees
     (b) Random Forest method
     (c) Logistic Regression model
     (d) Support vector regression (SVR)
     (e) KNeighbors regression
     (f) xGBoost regression
     to study feature and corrleations present 
     in the Customer chrun data at a Bank. We convert the predicted probabilities into
     binary group and compute the confusion matrix to determine the performance
     of the machine learning model.
     To view the project click on "CustomerChurn.ipynb"

### (4) NLP_automation_resume for HR management
     Natural language processing to classify a resume into a job category
     Read reasumes in pdf format using pdf2image ocr tool and convert it to
     PIL image, then use pytesseract.image_to_string to convert the image into strings objects.
     Then, use natural language processing tools such as nltk stopwords, regex, string replace
     to get important words from the string.
     We also remove prefix and suffix using PorterStemmer in nltk library.
     Then, we use CountVectorizer method and TfidfVectorizer method from sci-kit learn
     to create a independent feature vector.
     We use explore NaiveBayes model to train the data set. Compute confusion matrix.
     To view the project click on "Scrapping_resume.ipynb" and "NLP_modeling.ipynb"
     
### (5) NLP_DetectSpamEmail text data
     We apply Natural Language Processing (NLP) techniques to classify a text message 
     as a spam or good email (ham) using Python. The text preprocessing is done using 
     "stopwords" from NLTK corpus library and then, "CountVectorizer" function is used 
     to construct a 2d-matrix from the text messages. The 2d-matrix is used to build the model
     for the following different cases: 
     (a) Decision trees
     (b) Random Forest method
     (c) Logistic Regression model
     (d) Support vector classification (SVC)
     (e) xGBoost classifier
     We  compute the confusion matrix to determine the performance
     of the machine learning model.
     To view the project click on "NLP_DetectSpamEmail.ipynb"
     
### (6) TitanicData_DecisionTree_RandomForest 
     We apply statistical models: 
     (a) Decision trees
     (b) Random Forest method
     (c) Logistic Regression model
     (d) Support vector classifier (SVC)
     (e) xGBoost classifier 
     to study feature and corrleations present 
     in the titanic shipwreck data. We compute the confusion matrix to 
     determine the validity of the model.
     To view the project click on "Titanic_DecisionTree_RandomForest.ipynb"
     
### (7) MultiProjectRegressionModel 
    In this project, we do exploratry study of the data 
    Things to check: shape/size of data, nulls for missing values, types of coloumns 
    Use sci-kit library's SimpleImputer and OneHotEncoder to do preprocessing of the data 
    Perform analysis to look for a trend in the data 
    Build Linear or polynomial regression model on the training data
    To view the project click on "Project0_KaggleProject_ExploreData.ipynb"
    or "Project1_PreprocessingAndRegression.ipynb" or "Project2_RegressionLinearPoly.ipynb"
    
### (8) WebScrappingBS4  
     We use python library called "BeautifulSoup" to do webscrapping. 
     In the end we use seaborn library to make a bar graph.
     To view the project click on "WebScrappingBS4.ipynb"
