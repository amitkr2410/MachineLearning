This directory contains the following project:
(0) Project0_KaggleProject_ExploreData.ipynb
    Project0_kaggle_survey_2020_responses.csv
    ### Kaggle project (Basic exploratory data analysis)
    ### The data is taken from kaggle: https://www.kaggle.com/competitions/kaggle-survey-2020/data
    ### Things to do
    """ 
    Steps to approach data
    1. High level understanding of data (look at coloumns, nulls, shape etc.)
       First step is to understand the data and see if it is ready for analysis
       Things to check are shape/size of data, nulls for missing values, types of coloumns 
    2. Understand and make a overall strategy to tackle different question types (multiple answer vs single answer)
    3. Here, I want to experiment with plotly basics px and go (Not using matplotlib here)
    4. Final goal of the analysis is to look at the trend in the data
    """

(1) Project1_PreprocessingAndRegression.ipynb
    Project1_Data_PreprocessingAndRegression.csv
    ## Data preprocessing and regression model for a retail company where we recorded customer's country name, age, salary, and wether they bought the product or not \
    Use sci-kit library's SimpleImputer and OneHotEncoder to do preprocessing of the data \
    In the data set: Features are independent variable, dependent variable vector \
    Dependent variable we would keep as the last coloumn \
    We build the machine learning model using Linear and Polynomial regression method 
    
(2) Project2_RegressionLinearPoly.ipynb
    Project2_RankInCompany_Vs_Salary.csv
    ### Build a linear and polynomial regression model using sci-kit learn library \
    This is regression model to predict the expected salary in a company for a given position/level
