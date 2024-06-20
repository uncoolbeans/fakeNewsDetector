import newspaper
from newspaper import Article
 
# importing the os module
import os
 
# storing the path of modules file 
# in variable file_path
file_path = newspaper.__file__
 
# storing the directory in dir variable
dir = os.path.dirname(file_path)
 
# printing the directory

article = Article('https://edition.cnn.com/2024/05/12/china/china-xi-jinping-europe-putin-intl-hnk/index.html')
article.parse()

print(article.text)
print(dir)