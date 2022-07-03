import nltk
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('twitter_data.csv', delimiter=',', encoding='latin-1')
dataset.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
dataset.plot(x='ids',y='target')
plt.show()
exit()
shuffled_sample = dataset.sample(n=1000)

print(shuffled_sample.to_string())
print(type(shuffled_sample))
#print(shuffled_sample.isin({'target': 2}))



#venv\Scripts\activate.bat to activate venv