import logging
import json
from pathlib import Path

import nltk
import pandas as pd
import scipy.sparse
import sklearn

from . import utils


class Wikipedia:
    def __init__(self, path, suffix=".txt", pattern=r"\p{L}+\p{P}?\p{L}+",
                 lowercase=True):
        self.path = Path(path)
        self.suffix = suffix
        self.pattern = pattern
        self.lowercase = lowercase

    def mfw(self, n, stopwords=utils.STOPWORDS):
        return utils.create_frequency_list(self.path,
                                           n,
                                           stopwords)

    @staticmethod
    def load_mfw(filepath):
        with Path(filepath).open("r", encoding="utf-8") as mfw:
            return json.load(mfw)

    @property
    def lines(self):
        for file in self.path.glob(f"*{self.suffix}"):
            logging.info(f"Processing '{file}'...")
            with file.open("r", encoding="utf-8") as textfile:
                # Lazy reading:
                for n, line in enumerate(textfile):
                    logging.debug(f"Processing line {n}...")
                    yield line.lower() if self.lowercase else line

    @property
    def sentences(self):
        for line in self.lines:
            for sentence in nltk.sent_tokenize(line):
                yield sentence

    def tokens(self, sentences=False):
        for text in self.sentences if sentences else self.lines:
            yield list(utils.tokenize(text, self.pattern))

    def sparse_coo_matrix(self, mfw, stopwords=utils.STOPWORDS, sentences=False, window_size=2):
        voc = {}
        row = []
        col = []
        data = []
        for tokens in self.tokens(sentences):
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
        return scipy.sparse.coo_matrix((data, (row, col))).tocsr(), voc

    def sparse_coo_dataframe(self, mfw, stopwords=utils.STOPWORDS, sentences=False, window_size=2, sparse=True):
        csr, voc = self.sparse_coo_matrix(stopwords, mfw, sentences, window_size)
        return self._sparse2dataframe(csr, voc, sparse)

    @staticmethod
    def _sparse2dataframe(matrix, voc, sparse=True):
        if sparse:
            dataframe = pd.SparseDataFrame
        else:
            dataframe = pd.DataFrame
        df = dataframe(matrix)
        voc = dict((v, k) for k, v in voc.items())
        df = df.rename(index=voc)
        df = df.rename(columns=voc)
        return df

    @staticmethod
    def similarities(matrix, voc):
        similarities = sklearn.metrics.pairwise.cosine_similarity(matrix, dense_output=False)
        df = pd.SparseDataFrame(similarities)
        voc = dict((v, k) for k, v in voc.items())
        df = df.rename(index=voc)
        df = df.rename(columns=voc)
        return df

    @staticmethod
    def tfidf(matrix, sublinear_tf=True):
        transformer = sklearn.feature_extraction.text.TfidfTransformer(sublinear_tf=sublinear_tf)
        return transformer.fit_transform(matrix)
