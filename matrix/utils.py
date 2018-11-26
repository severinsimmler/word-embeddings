import json
import os
import collections

import nltk
import regex as re
import pandas as pd
import scipy.sparse


STOPWORDS = set(nltk.corpus.stopwords.words("german"))


def tokenize(text, pattern=r"\p{L}+\p{P}?\p{L}+"):
    for match in re.compile(pattern).finditer(text):
        yield match.group(0)


def create_frequency_list(filepath, limit, stopwords=STOPWORDS):
    freq = collections.Counter()
    for root, dirs, files in os.walk(filepath):
        for file_ in files:
            with open(os.path.join(root, file_), "r", encoding="utf-8") as f:
                freq.update(list([token.lower() for token in tokenize(f.read()) if token.lower() not in stopwords]))
    return {k for k, v in freq.most_common(limit)}


def create_cooccurrence_matrix(filepath, mfw, window_size, stopwords=STOPWORDS):
    if isinstance(mfw, str):
        with open(mfw, "r", encoding="utf-8") as f:
            mfw = json.load(f)
    voc = {}
    row = []
    col = []
    data = []
    for root, dirs, files in os.walk(filepath):
        for file_ in files:
            with open(os.path.join(root, file_), "r", encoding="utf-8") as f:
                # Does not load the whole file:
                for line in f:
                    sentences = nltk.sent_tokenize(line)
                    for sentence in sentences:
                        # Lowering _one_ string is more efficient than lowering _n_ tokens _n_ times:
                        tokens = [token for token in tokenize(sentence.lower())]
                        for pos, token in enumerate(tokens):
                            if token in stopwords or token not in mfw:
                                continue
                            i = voc.setdefault(token, len(voc))
                            start = max(0, pos-window_size)
                            end = min(len(tokens), pos+window_size+1)
                            for pos2 in range(start, end):
                                if pos2 == pos or tokens[pos2] in stopwords or tokens[pos2] not in mfw:
                                    continue
                                j = voc.setdefault(tokens[pos2], len(voc))
                                data.append(1)
                                row.append(i)
                                col.append(j)
                    break
    csr = scipy.sparse.coo_matrix((data, (row, col))).tocsr()
    df = pd.SparseDataFrame(csr)
    voc = dict((v, k) for k, v in voc.items())
    df = df.rename(index=voc); df = df.rename(columns=voc)
    return df


def save_matrix(matrix, filepath):
    matrix.to_csv(filepath)
