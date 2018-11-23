import os
import json
import pickle
import collections

import nltk
import regex as re


def load_pickle(filepath):
    return pickle.load(filepath)


def tokenize(text, pattern=r"\p{L}+\p{P}?\p{L}+"):
    for match in re.compile(pattern).finditer(text):
        yield match.group(0)


def create_matrix(filepath, mfw, window_size, stopwords):
    sparse_matrix = defaultdict(lambda: defaultdict(lambda: 0))
    for root, dirs, files in os.walk(filepath):
        for file_ in files:
            with open(os.path.join(root, file_), "r", encoding="utf-8") as f:
                data = f"[{f.read()[:-2]}]"
                json_data = json.loads(data)
                for row in json_data:
                    for sentence in nltk.tokenize.sent_tokenize(row["text"]):
                        tokens = list(tokenize(sentence.lower()))
                        for token in tokens:
                            if token in mfw:
                                for i in [x for x in range(-window_size, window_size + 1) if x != 0]:
                                    if tokens.index(token) + i >= 0:
                                        try:
                                            if tokens[tokens.index(token) + i] not in stopwords:
                                                sparse_matrix[token][tokens[tokens.index(token) + i]] += 1
                                        except IndexError:
                                            pass
                                        continue


def save_matrix(matrix, filepath):
    pass
