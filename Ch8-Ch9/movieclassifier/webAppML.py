import pickle
import re
import os
import numpy as np
from vectorizer import vect

clf = pickle.load(open(os.path.join('pk1_objects', 'classifier.pk1'), 'rb'))

label = {0: 'negative', 1: 'positive'}
example = ['I hate how much I love this movie']
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))