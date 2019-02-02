import itertools
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns


def save_classification_report(algorithm, y_pred, y_true, labels):
    with open(f"report-{algorithm}.txt", "w", encoding="utf-8") as file:
        report = classification_report(y_true, y_pred, target_names=labels)
        file.write(report)


def plot_confusion_matrix(cm, classes, algorithm):
    df = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 9))
    sns.heatmap(df, annot=True, cmap=sns.color_palette("Blues"))
    plt.tight_layout()
    plt.savefig(f"{algorithm}.svg")
    df.to_csv(f"{algorithm}-cm.csv")


def save_cross_val(clf, algorithm, data, labels, cv=10):
    cross_val = pd.Series(cross_val_score(clf, data, labels, cv=10))
    cross_val.to_csv(f"{algorithm}-cross-val.csv", index=False)


if __name__ == "__main__":
    f1_scores = list()
    accuracies = list()
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
    print(algorithm)
    clf = MultinomialNB(alpha=.01)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    save_classification_report(algorithm, pred, y_test, classes)
    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes, algorithm)
    # cross val
    clf = MultinomialNB(alpha=.01)
    save_cross_val(clf, algorithm, vec, Y)
    f1_scores.append({"algorithm": algorithm, "score": f1_score(pred, y_test, average="macro")})
    accuracies.append({"algorithm": algorithm, "score": cross_val_score(clf, vec, Y, cv=10)})
    

    #######################
    # LOGISTIC REGRESSION #
    #######################
    algorithm = "logistic-regression"
    print(algorithm)
    clf = LogisticRegression(solver="liblinear",
                             C=1,
                             penalty="l2",
                             multi_class="ovr")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    save_classification_report(algorithm, pred, y_test, classes)
    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes, algorithm)
    # cross val
    clf = LogisticRegression(solver="liblinear",
                             C=1,
                             penalty="l2",
                             multi_class="ovr")
    save_cross_val(clf, algorithm, vec, Y)
    f1_scores.append({"algorithm": algorithm, "score": f1_score(pred, y_test, average="macro")})
    accuracies.append({"algorithm": algorithm, "score": cross_val_score(clf, vec, Y, cv=10)})


    ##########################
    # SUPPORT VECTOR MACHINE #
    ##########################
    algorithm = "support-vector-machine"
    print(algorithm)
    clf = SVC(gamma="auto", C=1, coef0=0.0, kernel="poly")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    save_classification_report(algorithm, pred, y_test, classes)
    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes, algorithm)
    # cross val
    clf = SVC(gamma="auto", C=1, coef0=0.0, kernel="poly")
    save_cross_val(clf, algorithm, vec, Y)
    f1_scores.append({"algorithm": algorithm, "score": f1_score(pred, y_test, average="macro")})
    accuracies.append({"algorithm": algorithm, "score": cross_val_score(clf, vec, Y, cv=10)})


    ####################
    # GRADIENT DESCENT #
    ####################
    algorithm = "gradient-descent"
    print(algorithm)
    clf = SGDClassifier(n_jobs=-1, max_iter=50,tol=1e-3, alpha=0.001)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    save_classification_report(algorithm, pred, y_test, classes)
    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes, algorithm)
    # cross val
    clf = SGDClassifier(n_jobs=-1, max_iter=50,tol=1e-3, alpha=0.001)
    save_cross_val(clf, algorithm, vec, Y)
    f1_scores.append({"algorithm": algorithm, "score": f1_score(pred, y_test, average="macro")})
    accuracies.append({"algorithm": algorithm, "score": cross_val_score(clf, vec, Y, cv=10)})

    with open("f1-scores.json", "w", encoding="utf-8") as f:
        import json
        f.write(json.dumps(f1_scores))
    
    with open("accuracies.json", "w", encoding="utf-8") as f:
        import json
        f.write(json.dumps(accuracies))
    plt.figure()
    ax = pd.DataFrame(accuracies).plot.box(vert=False, color="black")
    ax.set_ylabel("Category")
    ax.set_xlabel("Accuracy")
    plt.savefig("accuracies.svg")
    
