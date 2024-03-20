import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string



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

print('preprocessing.....')
    
data['text'] = data['text'].apply(preProcess)

print(data['text'])

x = data['text']
y = data['label']


