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

import customtkinter as ctk
import threading
import pyglet


pyglet.font.add_file('SF-Pro.ttf')
pyglet.font.add_file('SF-Pro-Text-Heavy.otf')
pyglet.font.add_file('SF-Pro-Rounded-Heavy.otf')


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

modelTrained = False
x = data['text']
y = data['label']

#print(scrape('https://www.smh.com.au/national/at-28-jason-struggles-to-breathe-and-doesn-t-know-what-s-next-20190530-p51sva.html'))

#use threading to load pre process text and train the model while the GUI is showing
#use global variables to update the GUI when model has been trained


def switchToNewScreen(oldFrame,newFrame): #general purpose switch screen function
    oldFrame.forget()
    newFrame.pack()
    return


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1200x700")
        self.title("AI Fake News Detection")

        self.columnconfigure(0, weight = 1)
        self.rowconfigure(0, weight= 1)


        screen1 = mainScreen(self)
        screen1.grid(row = 0, column = 0, sticky = 'nw')

ctk.set_appearance_mode('light')


class mainScreen(ctk.CTkFrame):
    def __init__(self,master):
        global modelTrained
        global x 
        global y
        global data
        global LRModel

        self.bigFont = ctk.CTkFont(family='SF-Pro',size =42, weight = 'bold')
        self.normalFont = ctk.CTkFont(family='SF-Pro',size = 15)

        super().__init__(master, width=900, height= 700,fg_color='transparent')

        def testFunc():
            print('testing')

        self.columnconfigure(0, weight = 1)
        self.rowconfigure(0, weight= 2)

        self.helloLabel = ctk.CTkLabel(self, text = 'Hello there!',
                                       fg_color='transparent',
                                       font = self.bigFont,
                                       )
        self.helloLabel.grid(column = 0, row = 0, padx = 10, pady = 10, sticky = 'sw')

        self.aiLabel = ctk.CTkLabel(self, text = 'Model Information',
                                    fg_color='transparent',
                                    font = ('SF-Pro', 30)
                                    )
        self.aiLabel.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'sw')

        modelsTab = modelTabView(self)
        modelsTab.grid(row = 2, column = 0, columnspan = 2, padx = 10, pady = 10)

        
class modelTabView(ctk.CTkTabview):
    def __init__(self,master):
        super().__init__(master, width = 600,)

        self.normalFont = ctk.CTkFont(family='SF-Pro',size = 15)
        self.semiBold = ctk.CTkFont(family='SF-Pro', size = 22, weight='bold')

        self.add('Model 1')
        self.add('Model 2')
        self.add('Model 3')
        self.logisticModel = Model(LogisticRegression())

        def trainSelectedModel(x,y,model,tabNo):

            if tabNo == 1:
                self.model1StatusLabel.configure(text = 'training model')
                self.trainModel1Button.configure(state = 'disabled')
            elif tabNo == 2:
                pass #add code here when other 2 tabs are functional
            elif tabNo == 3:
                pass

            model.trainModel(x,y)
            i = 1
            for metric in model.metrics:
                self.metricLabel = ctk.CTkLabel(self.tab(f'Model {tabNo}'),
                                            text = f'{metric}: {model.metrics[metric]}')
                self.metricLabel.grid(column = 1, row = i)
                i+=1

            if tabNo == 1:
                self.model1StatusLabel.configure(text = 'model trained')
            elif tabNo == 2:
                pass
            elif tabNo == 3:
                pass

        #Tab 1 -> LOGISTIC REGRESSION
        self.model1NameLabel = ctk.CTkLabel(self.tab('Model 1'),
                                            text = 'Logistic Regression',
                                            font = self.semiBold)
        self.model1NameLabel.grid(column = 0, row = 0, columnspan = 3, pady = 5, padx = 5)

        self.model1StatusLabel = ctk.CTkLabel(self.tab('Model 1'),
                                             text = 'Model not trained',
                                             font = self.normalFont
                                             )
        self.model1StatusLabel.grid(column = 0, row = 1, pady = 5, padx = 5)

        self.trainModel1Button = ctk.CTkButton(self.tab('Model 1'),
                                    text = 'train model', 
                                    command = lambda: threading.Thread(target=trainSelectedModel,args=(x,y,self.logisticModel,1)).start()
                                    )
        
        self.trainModel1Button.grid(column = 0, row = 2, padx = 10, pady = 10)



class Model():
    def __init__(self, model):
        self.model = model
        self.metrics = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1 Score': 0}

    def trainModel(self,x,y):
        print(f'training')

        data['text'] = data['text'].apply(preProcess)
        x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3) #creates training and test sets from dataset
        vectorizer = TfidfVectorizer()
        xvTrain = vectorizer.fit_transform(x_train)
        xvTest = vectorizer.transform(x_test)
        #creating an instance of the model and training it using dataset
        LRModel = self.model
        LRModel.fit(xvTrain,y_train) 
        #assessing the model
        prediction = LRModel.predict(xvTest)
        self.accuracy = accuracy_score(y_test, prediction)
        self.precision = precision_score(y_test, prediction)
        self.recall = recall_score(y_test, prediction)
        self.f1 = f1_score(y_test, prediction)
        self.metrics['Accuracy'] = self.accuracy
        self.metrics['Precision'] = self.precision
        self.metrics['Recall'] = self.recall
        self.metrics['F1 Score'] = self.f1

        


# FOR TRAINING MODEL
        #self.button = ctk.CTkButton(self,text = 'train model', command = lambda: threading.Thread(target=trainModel,args=(x,y)).start())
        #self.button.grid(column = 0, row = 1)

        #self.button2 = ctk.CTkButton(self,text = 'press me2', command = lambda: testFunc())
        #self.button2.grid(column = 0, row = 2)
        
        

def exitProgram(): #ends program
    app.quit()


app = App()
app.mainloop()
