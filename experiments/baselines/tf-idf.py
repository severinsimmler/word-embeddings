import itertools
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def save_classification_report(algorithm, y_pred, y_true):
    with open(f"report-{algorithm}.txt", "w", encoding="utf-8") as file:
        report = classification_report(y_true, y_pred)
        file.write(report)


def plot_confusion_matrix(cm, classes, algorithm, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"{algorithm}.svg")


def save_cross_val(clf, algorithm, data, labels, cv=10):
    cross_val = pd.Series(cross_val_score(clf, data, labels, cv=10))
    cross_val.to_csv(f"{algorithm}-cross-val.csv", index=False)


if __name__ == "__main__":
    data = pd.read_csv("../../data/classification-corpus/final-corpus.csv")
    classes = data["category"].drop_duplicates().tolist()
    vec = TfidfVectorizer().fit_transform(data["text"])
    Y = LabelEncoder().fit_transform(data["category"])

    X_train, X_test, y_train, y_test = train_test_split(vec,
                                                        Y,
                                                        test_size=0.2, 
													    random_state=42, 
													    shuffle=True)

    ###########################
    # MULTINOMIAL NAIVE BAYES #
    ###########################
    algorithm = "multinomial-naive-bayes"
    clf = MultinomialNB(alpha=.01)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    save_classification_report(algorithm, pred, y_test)
    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes, algorithm)
    # cross val
    clf = MultinomialNB(alpha=.01)
    save_cross_val(clf, algorithm, vec, Y)
    

    #######################
    # LOGISTIC REGRESSION #
    #######################
    algorithm = "logistic-regression"
    clf = LogisticRegression(solver="liblinear",
                             C=1,
                             penalty="l2",
                             multi_class="ovr")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    save_classification_report(algorithm, pred, y_test)
    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes, algorithm)
    # cross val
    clf = LogisticRegression(solver="liblinear",
                             C=1,
                             penalty="l2",
                             multi_class="ovr")
    save_cross_val(clf, algorithm, vec, Y)


    ##########################
    # SUPPORT VECTOR MACHINE #
    ##########################
    algorithm = "support-vector-machine"
    clf = SVC(gamma="auto", C=1, coef0=0.0, kernel="poly")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    save_classification_report(algorithm, pred, y_test)
    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes, algorithm)
    # cross val
    clf = SVC(gamma="auto", C=1, coef0=0.0, kernel="poly")
    save_cross_val(clf, algorithm, vec, Y)



    ####################
    # GRADIENT DESCENT #
    ####################
    algorithm = "gradient-descent"
    clf = SGDClassifier(n_jobs=-1, max_iter=50,tol=1e-3, alpha=0.001)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    save_classification_report(algorithm, pred, y_test)
    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes, algorithm)
    # cross val
    clf = SGDClassifier(n_jobs=-1, max_iter=50,tol=1e-3, alpha=0.001)
    save_cross_val(clf, algorithm, vec, Y)

