import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
def textCleaning(line):
    stemmer = PorterStemmer()
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
num_sample = 30000
shuffled_sample = dataset.sample(n=num_sample)
shuffled_sample = shuffled_sample.set_index('text').to_dict()
inputs = list(shuffled_sample['target'].keys())
labels = list(shuffled_sample['target'].values())

# print(inputs)
# print(labels)

for index in range(len(inputs)):
    # print(textCleaning(inputs[index]))
    inputs[index] = textCleaning(inputs[index])
# print(inputs)

#countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
tfIdfVectorizer= TfidfVectorizer(analyzer='word', stop_words='english')
#count_wm = countvectorizer.fit_transform(inputs)
tfIdf = tfIdfVectorizer.fit_transform(inputs)
tokens = tfIdfVectorizer.get_feature_names_out()
data = pd.DataFrame(data=tfIdf.toarray(), columns=tokens)
# data['label'] = labels
# print(data)
# data = data.values.tolist()
# data = np.stack(data)
x = data.to_numpy()
print(data.shape)
y = np.stack(labels)
y[y > 1] = 1
# np.set_printoptions(threshold=sys.maxsize)
# print(y)
# exit()
# print(x.shape)
# print(y.shape)
# exit()

train_inp = x[:int(num_sample*0.8)]
train_label = y[:int(num_sample*0.8)]
valid_inp = x[int(num_sample*0.8):]
valid_label = y[int(num_sample*0.8):]
# print(train_inp.shape)
# print(valid_label)
model = RandomForestClassifier(max_depth = 20, n_estimators= 500)
model.fit(train_inp, train_label)
predictions = model.predict(valid_inp)
results = (predictions == valid_label)
print(results.astype(int).mean())



#train test split, scikit learn random forest classifier
exit()

#venv\Scripts\activate.bat to activate venv