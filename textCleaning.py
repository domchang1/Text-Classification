import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import tensorflow as tf
from tensorflow import keras
import string

def textCleaning(line):
    stemmer = PorterStemmer()
    #lemmatizer = WordNetLemmatizer()
    line = line.lower()
    stop = set(stopwords.words('english'))
    line = " ".join([stemmer.stem(word) for word in line.split() if word not in stop])
    #line = " ".join([lemmatizer.lemmatize(word) for word in line.split() if word not in stop])
    text = re.sub("@\S+", "", line)
    text = re.sub("\$", "", text)
    text = re.sub("https?:\/\/.*[\r\n]*", "", text)
    text = re.sub("#", "", text)
    punct = set(string.punctuation)
    text = "".join([ch for ch in text if ch not in punct])

    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    return text

#nltk.download('stopwords')
dataset = pd.read_csv('twitter_data.csv', delimiter=',', encoding='latin-1')
dataset.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
dataset.drop('date', inplace=True, axis=1)
dataset.drop('flag', inplace=True, axis=1)
dataset.drop('user', inplace=True, axis=1)
dataset.drop('ids', inplace=True, axis=1)
dataset = dataset[['text', 'target']]
#dataset.plot(x='ids',y='target')
#plt.show()
shuffled_sample = dataset.sample(n=10)
shuffled_sample = shuffled_sample.set_index('text').to_dict()
print(shuffled_sample)
# vectorize_layer = keras.layers.TextVectorization(
#     standardize=textCleaning,
#     max_tokens=1000,
#     output_mode='int',
#     output_sequence_length=250)
# vectorize_layer.adapt(shuffled_sample)
# exit()
inputs = list(shuffled_sample['target'].keys())
labels = list(shuffled_sample['target'].values())

print(inputs)
print(labels)

for index in range(len(inputs)):
    print(textCleaning(inputs[index]))
    inputs[index] = textCleaning(inputs[index])
print(inputs)

#countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
tfIdfVectorizer= TfidfVectorizer(analyzer='word', stop_words='english')
#count_wm = countvectorizer.fit_transform(inputs)
tfIdf = tfIdfVectorizer.fit_transform(inputs)
tokens = tfIdfVectorizer.get_feature_names_out()
print(pd.DataFrame(data=tfIdf.toarray(), columns=tokens))
exit()
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))

    

#print(shuffled_sample.isin({'target': 2}))



#venv\Scripts\activate.bat to activate venv