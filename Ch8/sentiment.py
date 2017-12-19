import re
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text).lower() + \
        ' '.join(emoticons).replace('-', '')
    return text


df = pd.read_csv('./movie_data.csv')

df['review'] = df['review'].apply(preprocessor)

print(df.head(3))


def tokenizer(text):
    return text.split()


run = 'runners like running and thus they run'

print(tokenizer(run))

porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


print(tokenizer_porter(run))

stop = stopwords.words('english')
print([w for w in tokenizer_porter(run)[-10:] if w not in stop])
