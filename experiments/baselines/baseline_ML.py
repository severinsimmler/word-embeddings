import pandas as pd
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

## load dataset
data = pd.read_csv("final-corpus.csv",sep=",")

## preprocess features
# Vectorize
vec = CountVectorizer(dtype=np.int32, analyzer = 'word',
                        lowercase=True, decode_error="ignore",  max_features=100000)
X = vec.fit_transform(data["text"])
X = pd.DataFrame(X.toarray())

# to relative values
X = X.div(X.sum(axis=1), axis=0)
X = np.array(X)

## targets
Y = data["category"]
Y = LabelEncoder().fit_transform(Y)

## split trainigsset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

## classifiers
# Logistic Regression
print("start training")
print("LogReg") 
clf = LogisticRegression(C=1, solver="liblinear", multi_class="ovr", n_jobs=-1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f1_score(y_test,y_pred, average="macro"))
print(f1_score(y_test,y_pred, average="micro"))

# Multinomial Naive Baise
print("MNNB")
clf = MultinomialNB(alpha=1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f1_score(y_test,y_pred, average="macro"))
print(f1_score(y_test,y_pred, average="micro"))

# Support Vector Machine
print("SVM")
clf = SVC(gamma="auto", C=1, coef0=0.0, kernel="poly")
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f1_score(y_test,y_pred, average="macro"))
print(f1_score(y_test,y_pred, average="micro"))

# Gradient Descent
print("Gradient")
clf = SGDClassifier(n_jobs=-1, max_iter=50,tol=1e-3, alpha=0.001)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f1_score(y_test,y_pred, average="macro"))
print(f1_score(y_test,y_pred, average="micro"))

# 5-fold Cross-Validation
print("cross_val_score")
clf = LogisticRegression(C=1, solver="liblinear", multi_class="ovr", n_jobs=-1)
scores = cross_val_score(clf, X, Y, cv=5, scoring="f1_macro")
print(np.mean(scores))
## Save results
scores = pd.DataFrame(scores)
scores.to_csv("res2.csv",sep="\t")

