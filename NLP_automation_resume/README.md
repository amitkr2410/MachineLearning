### NLP on resumes for HR management
    In this project, we develop scripts to scrap the resume data from 
    https://www.livecareer.com/resume-search/search?jt=hr&bg=85&eg=100&comp=&mod=&pg=1
    for various job categories such as 
    job_list = ['HR', 'Data-Scientist', 'Machine-Learning-Engineer',  'TEACHER', 'ADVOCATE',
    'Dental-Hygienist', 'Doctor', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 'SALES',
    'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 'CHEF', 'FINANCE', 'APPAREL',
    'ENGINEERING', 'ACCOUNTANT', 'CONSTRUCTION', 'PUBLIC-RELATIONS', 'BANKING',
    'ARTS', 'AVIATION']
    Then, the resume data is saved in pdf format to create a catalog. 
    The generated pdf are converted into text which are then analyzed using NLP techniques
    and used to build a machine learning model for classification.    
    The projects can be viewed by clicking on the file with extension "*.ipynb" (JupyterNotebook)
    
### (1) Generate raw resume pdf files by scrapping  website 
             We create python script to download resumes from livecareer.com
             for example :
    	     https://www.livecareer.com/resume-search/search?jt=hr&bg=85&eg=100&comp=&mod=&pg=1
 	     using webscrapping tools. The extracted html code is converted using pdfkit and
	     saved into pdf files
 	     The extracted html code is also saved into a csv file
    	     To view the project click on "Scrapping_resume.ipynb"
    
### (2) Use NLP and sci-kit learn to build model for resume classification
    	     Natural language processing to classify a resume into a job category 
 	     Read reasumes in pdf format using pdf2image ocr tool and convert it to  
	     PIL image, then use pytesseract.image_to_string to convert the image into strings objects.
	     Then, use natural language processing tools such as nltk stopwords, regex, string replace 
	     to get important words from the string.
	     We also remove prefix and suffix using PorterStemmer in nltk library.
 	     Then, we use CountVectorizer method and TfidfVectorizer method from sci-kit learn
 	     to create a independent feature vector. 
 	     We use explore NaiveBayes model to train the data set.
 	     Compute confusion matrix.
    	     To view the project click on "NLP_modeling.ipynb"
    

### The two approaches used to create independent column vectors from 
###  resume texts are CountVectorizer method and TfidfVectorizer method. Both methods yield a similar 
###  accuracy score, i.e. 83%
