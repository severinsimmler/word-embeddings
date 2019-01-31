from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

data = pd.read_csv("final-corpus.csv",sep=",")
vec = TfidfVectorizer().fit_transform(data["text"])
Y = LabelEncoder().fit_transform(data["category"])

X_train, X_test, y_train, y_test = train_test_split(vec, Y, test_size=0.2, 
													random_state=42, 
													shuffle=True)

clf = MultinomialNB(alpha=.01)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
f1_score(y_test, pred, average='macro')
## 0.8936604149404189

clf = LogisticRegression(solver="liblinear", C=1, penalty="l2", multi_class="ovr")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
f1_score(y_test, pred, average='macro')
## 0.9006840926994731

cross_val = cross_val_score(MultinomialNB(), vec, Y, cv=10)
cross_val_mean = np.mean(cross_val)
##[0.85092127 0.83082077 0.82747069 0.86195286 0.85353535 0.87521079
## 0.83979764 0.85398981 0.85908319 0.84957265]
## Mean: 0.8502355032175706
