import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from bs4 import BeautifulSoup
from urllib.request import urlopen
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#
#stopWords = set(stopwords.words('english')) 
#print(stopWords)
#
#stemmer = PorterStemmer()
#lemmatizer = WordNetLemmatizer()

true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

true['label'] = 1
fake['label'] = 0

data = pd.concat([fake,true],axis = 0)
data = data.drop(['subject','date'],axis = 1)

data = data.sample(frac=1) #shuffles the dataset
data.reset_index(inplace=True)
data.drop(['index'], axis = 1, inplace = True)


def preProcess(text): #make the text suitable for the machine to read, eliminate irrelevant bits

    #lowercase everything
    text = text.lower()
    #removing html embeds
    text = re.sub(r'https?://\S+\www\.\S+','',text)
    #remove punctuation
    text = re.sub(r'<.*?>','', text)
    #remove digits
    text = re.sub(r'\d','',text)
    #remove newlines
    text = re.sub(r'\n',' ', text)

    #words = word_tokenize(text)
#
    ##removing stopwords
    #words = [word for word in words if word not in stopWords]
#
    ##lemmatizing text
    #words = [stemmer.stem(word) for word in words]
#
    #text = ''.join(words)
#
    return text


def scrape(link):
    url = link
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    return soup.get_text()


#print('preprocessing.....')
    
#data['text'] = data['text'].apply(preProcess)

#print(data['text'])

x = data['text']
y = data['label']

def trainModel(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3) #creates training and test sets from dataset
    vectorizer = TfidfVectorizer()
    xvTrain = vectorizer.fit_transform(x_train)
    xvTest = vectorizer.transform(x_test)

    #creating an instance of the model and training it using dataset
    LRModel = LogisticRegression()
    LRModel.fit(xvTrain,y_train) 

    #assessing the model
    prediction = LRModel.predict(xvTest)
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)




trainModel(x,y)

#print(scrape('https://www.smh.com.au/national/at-28-jason-struggles-to-breathe-and-doesn-t-know-what-s-next-20190530-p51sva.html'))




