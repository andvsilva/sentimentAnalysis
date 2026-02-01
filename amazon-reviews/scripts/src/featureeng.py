###########################################################
### step  2 - feature engineering
# 
# Add description for this code.
###########################################################

# libraries for this project
import json
import pandas as pd
import numpy as np
from datetime import datetime
from IPython.display import HTML
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
from matplotlib.backends.backend_pdf import PdfPages
import sys
import gc
import feather
import toolkit as tool
from icecream import ic
from sys import getsizeof
import time
import requests as re
import re # for regex
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
#nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE

# Get start time 
start_time = time.time()

now = datetime.now()
 
print("date..............:", now)

# Here we are going to implement some functions
# redefine score
def sentiment(label):
    if label == 5.0 or label == 4.0:
        return "0"
    #elif label == 3.0:
    #    return "Neutral"
    elif label == 1.0 or label == 2.0:
        return "1"

print('*****************************************************')
print('Starting the feature engineering of the dataset.')
print('*****************************************************')

# Loadind data
print("Loading dataset - cleaned to feature engineering...")

df_featuresel = pd.read_feather('data-feather/cleaned.ftr')

# retirar os neutros.
df_featuresel = df_featuresel[df_featuresel['Score'] != 3]
    
df_featuresel['negative'] = df_featuresel["Score"].apply(sentiment)

# Counting reviews by stars
ax = df_featuresel['Score'].value_counts().sort_index().plot(kind='bar',
                                                  title='Contagem de Reviews por estrelas',
                                                  figsize=(10, 5)
                                                 )

ax.set_xlabel('Review Stars')
ax.set_ylabel('Contagem')
plt.savefig('pngs/counting_reviews_stars.png')
#plt.show()

#Mude df5 para df3 para pegar toda base
texts = df_featuresel['Text'].sum()
texts[0:1000]

stop_pt = nltk.corpus.stopwords.words('portuguese')
stop_en = nltk.corpus.stopwords.words('english')
stopwords_pa = stop_en + stop_pt
stopwords_pa.extend(['-',''])

list_words = texts.split()
list_words = [l.strip().lower() for l in list_words]

# lista de palavras do Text 'reviews'
list_words = [l.strip(string.punctuation) for l in list_words]
list_words = [l for l in list_words if l not in stopwords_pa]
freqdist = Counter(list_words)
#print(dict(freqdist.most_common(10)))

print('Word clouds...Sentiment Analysis: Amazon ')

from wordcloud import WordCloud
cleaned = ' '.join(list_words)
wordcloud = WordCloud().generate(cleaned)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(wordcloud, interpolation='nearest')
plt.grid(False)
plt.tight_layout()
plt.savefig('pngs/word_clouds.png')

# converter string para inteiro
df_featuresel['negative'] = pd.to_numeric(df_featuresel['negative'])

# >>> feature select
df_featuresel = df_featuresel[['Text', 'negative']]

# Define batch size
batch_size = 100

# Split the DataFrame into batches
batches = np.array_split(df_featuresel, len(df_featuresel) // batch_size)

i=0
# Loop over the batches
for batch in batches:
    print(f'>>> #{i} Batching...')
    
    # feature and target
    X = batch['Text']
    y = batch['negative']
    
    # release memory RAM
    tool.release_memory(batch)
    
    print('Transform X(text) to array, please...')
    
    cv = CountVectorizer()
    X = cv.fit_transform(X).toarray()
    #print(X)
    print("Now X is one array, go ahead.")
    
    smote = SMOTE()

    print('imbalance data to balance...')
    
    X, y = smote.fit_resample(X, y)
    
    print("X.shape = ",X.shape)
    print("y.shape = ",y.shape)
    
    print('done, dataset balance. thanks!')
    
    # release memory - array
    tool.release_array(X)
    tool.release_array(y)
    
    #print(batch.head())
    i+=1

print("saving the file format feather...")

# this is important to do before save in feather format.
df_featuresel = df_featuresel.reset_index(drop=True)

# saving in the feather format
df_featuresel.to_feather('data-feather/featureselected.ftr')

# time of execution in minutes
time_exec_min = round( (time.time() - start_time)/60, 4)

print(f'time of execution (preprocessing): {time_exec_min} minutes')
print("the feature engineering is done.")
print("The next step is to do the modeling.")
print("All Done.")

