# Author: Rishabh Panda, KIIT Bhubaneswar

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer()

data = pd.read_csv("C:/Users/KIIT/Documents/IMDBdataset/IMDB Dataset.csv")
data.head()

import re
def preprocessor(review):
             review=re.sub('<[^>]*>','',review)
             return review
         
data['review'] = data['review'].apply(preprocessor)

from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()

def tokenizer(text):
        return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

from nltk.corpus import stopwords
stop=stopwords.words('english')

from wordcloud import WordCloud

positivetext = data[ data['sentiment'] == "positive"]
positivetext = positivetext['review']

negativetext = data[data['sentiment'] == "negative"]
negativetext = negativetext['review']

def wordcloud_draw(data, color = 'white'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                              if(word!='movie' and word!='film')
                            ])
    wordcloud = WordCloud(stopwords=stop, background_color=color,
                          width=2500, height=2000).generate(cleaned_word)
    
    plt.figure(1,figsize=(15, 15))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive Reviews")
wordcloud_draw(positivetext,'pink')
print("Negative Reviews")
wordcloud_draw(negativetext, 'purple')