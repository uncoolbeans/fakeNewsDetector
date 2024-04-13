This is a simple software project which is my first venture into implementing machine learning and AI in python.
Libraries and modules used: 
- Scikit-learn machine learning - https://scikit-learn.org/stable/
- Python Natural Language Toolkit - https://www.nltk.org
- BeautifulSoup4 HTML Parseer - https://beautiful-soup-4.readthedocs.io/en/latest/
- urlib Python URL Handler - https://docs.python.org/3/library/urllib.html
- CustomTkinter GUI - https://customtkinter.tomschimansky.com/documentation/windows/
- pandas 2.2.1 - https://pandas.pydata.org/docs/
- numpy 1.26 - https://numpy.org/doc/

Please ensure that these libraries are pre-installed before running the program. Run the follwing code in the terminal line by line (Windows):
```
pip install pandas
pip install numpy
pip install scikit-learn
pip install beautifulsoup4
pip install customtkinter
pip install regex
```

Understanding classification metrics:
There are 4 main metrics the program will provide about the model: accuracy, precision, recall and F-1 score.
To understand these it is important to understand the 4 possible cases of an output:
- True positive (TP): a positive result is predicted as positive
- True negative (TN): a negative result is predicted as negative
- False positive (FP): a negative result is predicted as positive
- False negative (FN): a positive result is predicted as negative

Further reading into score metrics: https://encord.com/blog/f1-score-in-machine-learning/#:~:text=The%20F1%20score%20or%20F,the%20reliability%20of%20a%20model.

Below is a brief explanation of what the metrics indicate.
1. Accuracy:
   This metric measures how 'correct' the model is by taking the number of correct predictions as a percentage of total predictions

![Accuracy metric calculation]('https://images.prismic.io/encord/39632c98-d4c5-4ea7-b573-094c5ef2d608_image1.png?auto=compress,format')

3. Precision:
   This metric measures the quality of positive predictions by evaluating the outcomes that are predicted to be 'positive'. It takes the number of true positives as a percentage of the sum of true positives    true negative predictions.

![Precision metric calculation]('https://images.prismic.io/encord/ab763806-f2b7-4a12-97bb-97076fc6aca9_image3.png?auto=compress,format')

4. Recall:
   This metric evaluates the model's ability to predict positive events correctly. The number of true positive predicted is taken as a percentage of the sum of false negatives and true positives.

![Precision metric calculation]('https://images.prismic.io/encord/de4c1d17-57da-4088-827e-2d09b5700c8f_image5.png?auto=compress,format')
