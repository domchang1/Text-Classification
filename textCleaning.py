import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def textCleaning(line):
    line.lower()
    stop = stopwords.words('english')
    line = " ".join([word for word in line.split() if word not in stop])
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", line)
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
shuffled_sample.to_dict()
print(shuffled_sample)
for index,row in shuffled_sample.iterrows():
    print(textCleaning(row['text']))

tfIdfVectorizer= TfidfVectorizer(analyzer='word', stop_words='english')
tfIdf = tfIdfVectorizer.fit_transform(list(shuffled_sample.values()))
tokens = tfIdfVectorizer.get_feature_names_out()
print(pd.DataFrame(data=tfIdf.toarray(), columns=tokens))
exit()
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))

    

#print(shuffled_sample.isin({'target': 2}))



#venv\Scripts\activate.bat to activate venv