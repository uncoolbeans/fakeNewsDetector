import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

#logistic regression model
from sklearn.linear_model import LogisticRegression

#Random forest model
from sklearn.ensemble import RandomForestClassifier

#Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import newspaper
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
from newspaper import Article
from newspaper import Config
import threading
import pyglet
import time

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

predictions = None


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

#pre-processing and vectorising text
xvTrain = None
xvTest = None
y_test = None
y_train = None

vectorizer = None

class article_info():
    def __init__(self, text, title = '',model = '', date = '', verdict = 1):
        text = re.sub(r'\n',' ', text)
        if len(title) > 30:
            title = title[0:25]+'...'
            self.title = title
        else:
            self.title = title
        if len(text) > 55:
            text = text[0:55]+'\n'+text[55:115]+'...'
            self.text = text
        else:

            self.text = text

        if date == None:
            self.date = ''
        elif date != '':
            self.date = f"{date.day}/{date.month}/{date.year}"
        else:
            self.date = date

        self.model = model

        if verdict == 1:
            self.verdict = 'Real'
        else:
            self.verdict = 'Fake'

predictedArticles = []

class timerError(Exception):
    """timer errors"""

class timer():
    def __init__(self):
        self.start_time = None

    def start(self):
            if self.start_time is not None:
                raise timerError(f'Timer is already running')
            self.start_time = time.perf_counter()

    def stop(self):
            if self.start_time is None:
                raise timerError(f"Timer is not running. Use .start() to start it")

            self.elapsed_time = time.perf_counter() - self.start_time
            self.start_time = None
            print(f"Elapsed time: {self.elapsed_time:0.4f} seconds")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.columnconfigure(0, weight = 1)
        self.rowconfigure(0, weight= 1)

        self.loadScreen = loadingScreen(self)
        self.loadScreen.grid(row = 0, column = 0)

        def vectoriseData():
            global vectorizer

            print('vectorising data')
            global xvTrain, xvTest, y_test, y_train

            #apply pre processing steps
            data['text'] = data['text'].apply(preProcess)

            #split data into testing sets and training sets randomly
            x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3) 

            #vectorise data to use to train model
            vectorizer = TfidfVectorizer(stop_words='english')
            xvTrain = vectorizer.fit_transform(x_train)
            xvTest = vectorizer.transform(x_test)

            self.loadScreen.after(10, self.loadScreen.destroy())

            self.screen1 = mainScreen(self)
            self.screen1.grid(row = 0, column = 0, sticky = 'nw')

        self.geometry("1150x600")
        self.title("AI Fake News Detection")

        #carry out pre=processing and vectorise data before model is trained
        vectoriserThread = threading.Thread(target=vectoriseData)
        vectoriserThread.start()



        #screen1 = mainScreen(self)
        #screen1.grid(row = 0, column = 0, sticky = 'nw')

ctk.set_appearance_mode('light')

class loadingScreen(ctk.CTkFrame):
    def __init__(self,master):
        super().__init__(master=master,width = 900, height = 700)

        self.bigFont = ctk.CTkFont(family='SF-Pro',size =42, weight = 'bold')

        self.loadingLabel = ctk.CTkLabel(self, 
                                         text = 'Preparing the data for training, please wait...',
                                         font = self.bigFont,
                                         corner_radius=10)
        self.columnconfigure(0,weight=1)
        self.rowconfigure(0, weight= 1)

        self.loadingLabel.grid(row = 0, column = 0, padx = 0, pady = 0)
        

class mainScreen(ctk.CTkFrame):
    def __init__(self,master):
        global modelTrained
        global x 
        global y
        global data
        global LRModel
        global predictions

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
        self.helloLabel.grid(column = 0, row = 0, padx = 20, pady = 10, sticky = 'sw')

        self.aiLabel = ctk.CTkLabel(self, text = 'Model Information',
                                    fg_color='transparent',
                                    font = ('SF-Pro', 30)
                                    )
        self.aiLabel.grid(row = 1, column = 0, padx = 20, pady = 10, sticky = 'sw')

        modelsTab = modelTabView(self)
        modelsTab.grid(row = 2, column = 0, columnspan = 2, padx = 20, pady = 10)

        predictionsLabel = ctk.CTkLabel(self, text = 'Recent Predictions',
                                        fg_color='transparent',
                                        font = ('SF-Pro', 25)
                                        )
        predictionsLabel.grid(column = 2, row = 0, padx = 10, sticky = 'sw')

        predictions = predictionsFrame(self)
        predictions.grid(row = 1, column = 2, padx = 10, pady = 5, rowspan = 3)

        tips = tipsFrame(self)
        tips.grid(row = 3, column = 0, columnspan = 2, padx = 20, pady = 10)

        
class modelTabView(ctk.CTkTabview):
    def __init__(self,master):
        super().__init__(master, width = 600)

        self.normalFont = ctk.CTkFont(family='SF-Pro',size = 15)
        self.semiBold = ctk.CTkFont(family='SF-Pro', size = 22, weight='bold')

        self.model1Trained = False
        self.model2Trained = False
        self.model3Trained = False

        self.add('Model 1')
        self.add('Model 2')
        self.add('Model 3')
        self.logisticModel = Model(LogisticRegression())

        frame1 = modelFrame(self.tab('Model 1'),1)
        frame1.grid(row = 0, column = 0, sticky = 'nw')

        frame2 = modelFrame(self.tab('Model 2'),2)
        frame2.grid(row = 0, column = 0, sticky = 'nw')

        frame3 = modelFrame(self.tab('Model 3'),3)
        frame3.grid(row = 0, column = 0, sticky = 'nw')


class modelFrame(ctk.CTkFrame):
    def __init__(self, master, tabNo):
        super().__init__(master, fg_color='transparent')
        self.article = None
        #creating an instance of the AI model
        if tabNo == 1:
            self.model = Model(LogisticRegression()) 
            self.modelName = 'Logisitic Regression'
        elif tabNo == 2:
            self.modelName = 'Random Forest Classifier'
            self.model = Model(RandomForestClassifier()) 
        elif tabNo == 3:
            self.modelName =  'Naive Bayes Classifier'
            self.model = Model(MultinomialNB())

        def trainSelectedModel(x,y,model,tabNo):

            self.modelStatusLabel.configure(text = 'Training model, please wait...')
            self.trainModelButton.configure(state = 'disabled')

            
            #train model
            model.trainModel(x,y) 

            self.modelStatusLabel.after(10,self.modelStatusLabel.destroy())
            i = 1
            for metric in model.metrics:
                self.metricLabel = ctk.CTkLabel(self.metricsFrame,
                                        text = f'{metric}: {model.metrics[metric]}',
                                        fg_color= 'lightgray',
                                        corner_radius=5,
                                        height = 35,
                                        width = 250,
                                        justify = 'left',
                                        anchor = 'w'
                                
                                        )
                self.metricLabel.grid(column = 0, row = i, padx = 5, sticky = 'w', pady = 5)
                i+=1
            self.predictButton.configure(state = 'enabled')
            return

        def scrape(link): #extracts body text from a news article URL
            url = link
            url = url.strip()

            #check if URL is empty
            if url == '':
                msg = CTkMessagebox(title="Error", message="URL box must not be empty if you wish to use an URL!", icon="cancel")
                return
            self.article = Article(url)

            try:
                self.article.download()
                self.article.parse()
            except newspaper.article.ArticleException:
                CTkMessagebox(title = 'Error', message = 'Unable to extract text from link. Please try a new link or copy paste text directly into the textbox.', icon='cancel' )
                return
            
            #extracting text from article
            text = self.article.text

            if text.strip() == '':
                CTkMessagebox(title = 'Error', message = 'Unable to extract text from link. Please try a new link or copy paste text directly into the textbox. Please ensure it is a news article.', icon='cancel' )

            #clear textbox of text
            self.textbox.delete(0.0,'end')
            #add scraped text to textbox
            self.textbox.insert(0.0, text)

            done = CTkMessagebox(title = 'Text extracted', message = 'Successfully scraped news article for body text. Click "get prediction" to run the text through the model.', icon = 'check')
            return 

        def predict():
            global predictions
            global vectorizer
            global predictedArticles

            text = {'text': [self.textbox.get(0.0,'end')]}
            test = pd.DataFrame(text)

            test['text'] = test['text'].apply(preProcess)
            processedText = test['text']

            vectorisedText = vectorizer.transform(processedText)

            prediction = self.model.predict(vectorisedText)

            print(prediction)

            if prediction == 0:
                print('It is fake news')
                CTkMessagebox(title = 'Prediction',
                              message=f'{self.modelName} predicts that the article contains FAKE news.',
                              icon='cancel')

            elif prediction == 1:
                print('It is real news')
                CTkMessagebox(title = 'Prediction',
                              message=f'{self.modelName} predicts that the article contains REAL news.',
                              icon='check')

            if self.article is not None:
                msg = CTkMessagebox(title = 'URL Detected', 
                                    message=f'You have recently scraped an URL, do you want to use the following information obtained from the URL?\nTitle: {self.article.title}\nDate: {self.article.publish_date}',
                                    icon = 'question',
                                    option_1='Yes',
                                    option_2='No')
                if msg.get() == 'Yes':
                    predictedArticles.insert(0,article_info(self.article.text, self.article.title, self.modelName, self.article.publish_date, verdict=prediction))
                    predictions.draw()
                else:
                    predictedArticles.insert(0,article_info(self.article.text, model = self.modelName, verdict = prediction))
                    predictions.draw()

            
            return
        
                

        self.normalFont = ctk.CTkFont(family='SF-Pro',size = 15)
        self.semiBold = ctk.CTkFont(family='SF-Pro', size = 22, weight='bold')

        self.modelNameLabel = ctk.CTkLabel(self,
                                            text = self.modelName,
                                            font = self.semiBold)
        self.modelNameLabel.grid(column = 0, row = 0, columnspan = 3, pady = 5, padx = 5, sticky = 'w')

        self.metricsFrame = ctk.CTkFrame(self,width=250, height = 190)
        self.metricsFrame.grid(row = 1, column = 0, padx = 5, pady = 5, rowspan = 2)

        self.metricsLabel = ctk.CTkLabel(self.metricsFrame,
                                          text = 'Model Metrics',
                                          font = self.normalFont
                                          )
        self.metricsLabel.grid(row = 0, column = 0, sticky = 'w',padx = 5)

        self.modelStatusLabel = ctk.CTkLabel(self.metricsFrame,
                                             text = 'Model not trained.\nPress [Train Model] to begin training.',
                                             font = self.normalFont,
                                             width=250, height = 150
                                             )
        self.modelStatusLabel.grid(column = 0, row = 1, pady = 5, padx = 5)

        self.trainModelButton = ctk.CTkButton(self,
                                    text = 'Train Model', 
                                    command = lambda: threading.Thread(target=trainSelectedModel,args=(x,y,self.model,tabNo)).start()
                                    )
        self.trainModelButton.grid(column = 0, row = 3, padx = 5, pady = 5)

        self.URLbox = ctk.CTkEntry(self,
                                    placeholder_text='Paste URL here',
                                    width=250
                                    )
        self.URLbox.grid(row = 0, column = 1, padx = 5)

        self.textbox = ctk.CTkTextbox(self,
                                       width=400,
                                       height = 175)
        self.textbox.insert('0.0', 'Paste body text of article here. (delete this text when pasting)')
        self.textbox.grid(row = 1, column = 1,columnspan = 2, sticky = 'w')

        self.scrapeButton = ctk.CTkButton(self, text =  'Scrape URL text',
                                         command = lambda: scrape(self.URLbox.get())
                                         )
        self.scrapeButton.grid(row = 0, column = 2, padx = 5)

        self.predictButton = ctk.CTkButton(self,
                                           text = 'Get prediction from text',
                                           command = lambda: predict(),
                                           state='disabled'
                                           )
        self.predictButton.grid(row = 2, column = 1, columnspan = 2)

        

class Model():
    def __init__(self, model):
        self.model = model
        self.metrics = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1 Score': 0}

    def trainModel(self,x,y):

        print(f'training')

        global xvTrain
        global xvTest
        t = timer()
        t.start()
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

        t.stop()
        self.trainingTime = t.elapsed_time
        print(self.trainingTime)

    def predict(self,vectorisedText):

        prediction = self.model.predict(vectorisedText)

        return prediction[0]

class predictionsFrame(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master, width = 365, height = 500)
        self.bigFont = ctk.CTkFont(family='SF-Pro',size =13, weight = 'bold')
        self.draw()

    def draw(self):
        global predictedArticles

        if len(predictedArticles) ==  0:
                self.label = ctk.CTkLabel(self, text = 'No past predictions, start training models to begin!',
                                     fg_color='transparent',bg_color='transparent',
                                     font=self.bigFont
                                     )
                self.label.grid(row = 0, column = 0)
                return

        for i,article in enumerate(predictedArticles):
            self.frame = ctk.CTkFrame(self,
                                          fg_color='light grey',
                                          corner_radius=10
                                          )
            self.frame.columnconfigure(1,weight = 2)
            self.frame.columnconfigure(0, weight = 1)
            if article.title == '':
                    titleLabel = ctk.CTkLabel(self.frame,text = f"Title: None",fg_color='transparent',bg_color='transparent',width=210, anchor='w', font=self.bigFont)
                    titleLabel.grid(row = 0, column = 0, padx = 5, pady = 0, sticky = 'w')
            else:
                    titleLabel = ctk.CTkLabel(self.frame,text = f"Title: {article.title}",fg_color='transparent',bg_color='transparent',width=210, anchor='w', font=self.bigFont)
                    titleLabel.grid(row = 0, column = 0, padx = 5, pady = 0, sticky = 'w')

            if article.date == '':
                    dateLabel = ctk.CTkLabel(self.frame,text = f"Date: None",fg_color='transparent',bg_color='transparent',width=210, anchor='w', font=self.bigFont)
                    dateLabel.grid(row = 1, column = 0, padx = 5, pady = 5, sticky = 'nw')
            else:
                    dateLabel = ctk.CTkLabel(self.frame,text = f"Date: {article.date}",fg_color='transparent',bg_color='transparent',width=210, anchor='w', font=self.bigFont)
                    dateLabel.grid(row = 1, column = 0, padx = 5, pady = 5, sticky = 'nw')

            textLabel = ctk.CTkLabel(self.frame,
                                     text = f"Text:\n{article.text}",
                                     fg_color='transparent',bg_color='transparent',
                                     width=100, anchor='w',
                                     justify = 'left'
                                     )
            textLabel.grid(row = 2, column = 0, padx = 5, pady = 5, sticky = 'w', columnspan = 2)
            if article.verdict == 'Real':
                color ='green'
            else:
                color = 'red'
            verdictLabel = ctk.CTkLabel(self.frame,
                                        text = article.verdict,
                                        fg_color= color,
                                        corner_radius=10
                                        )
            verdictLabel.grid(row = 0, column = 1, rowspan = 2,padx=5,pady=5)

            modelLabel = ctk.CTkLabel(self.frame,
                                      text = article.model,
                                      width = 335,
                                      justify = 'left',
                                      font = self.bigFont
                                      )
            modelLabel.grid(row = 3, column = 0, columnspan = 2, padx = 3, pady = 5, sticky = 'w')

            self.frame.grid(row = i, column = 0, padx = 5, pady = 5)
            print(article.title)
        pass

class tipsFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=10)

        quoteLabel = ctk.CTkLabel(self, text = 'uagiagdawigdwiagda',
                                  fg_color='transparent', bg_color='transparent'
                                )
        quoteLabel.grid(row = 0, column = 1, padx = 255, pady = 45)

        self.backButton = ctk.CTkButton(self, text = '<',
                                        width=20
                                        )
        self.backButton.grid(row = 0, column = 0, padx = 5, pady = 45)

        self.fwdButton = ctk.CTkButton(self, text = '>',
                                       width = 20
                                        )
        self.fwdButton.grid(row = 0, column = 2, padx = 5, pady = 45)
        pass

def exitProgram(): #ends program
    app.quit()


app = App()
app.mainloop()


#to get a prediction 