#!/usr/bin/env python3

import argparse
import logging
import os

import sklearn

import matrix


logging.basicConfig(level=logging.DEBUG,
                    filename="matrix-tool.log",
                    filemode="w")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="matrix-tool",
                                     description="CLI tool to process a Wikipedia "
                                                 "dump to a word-word matrix.")
    parser.add_argument("--corpus", help="Path to corpus directory.")
    parser.add_argument("--suffix", help="Suffix of the text files.")
    parser.add_argument("--lowercase", help="Use this parameter to lowercase all letters.", action="store_true")
    parser.add_argument("--mfw", help="Path to JSON file with most frequent words.")
    parser.add_argument("--n-mfw", help="Count tokens and use the n most frequent words.", type=int)
    parser.add_argument("--window", help="Context window size.", type=int)
    parser.add_argument("--sentences", help="Use sentences instead of lines.", action="store_true")
    parser.add_argument("--output", help="Path to output directory.")
    parser.add_argument("--stopwords", help="Optional external stopwords list.")

    args = parser.parse_args()

    wikipedia = matrix.corpus.Wikipedia(path=args.corpus,
                                        suffix=args.suffix,
                                        lowercase=args.lowercase)

    if args.stopwords:
        with Path(args.stopwords).open("r", encoding="utf-8") as textfile:
            stopwords = textfile.read().split("\n")

    if args.mfw:
        mfw = wikipedia.load_mfw(args.mfw)
    elif args.n_mfw:
        mfw = wikipedia.mfw(args.n_mfw, matrix.utils.STOPWORDS if not args.stopwords else stopwords)
    else:
        raise ValueError("You have to set either a threshold for the most frequent words, "
                         "or pass a path to a JSON file with most frequent words.")

    csr, vocab = wikipedia.sparse_coo_matrix(mfw,
                                             stopwords=matrix.utils.STOPWORDS if not args.stopwords else stopwords,
                                             sentences=args.sentences,
                                             window_size=args.window)

    df = wikipedia.sparse_coo_dataframe(mfw,
                                        stopwords=matrix.utils.STOPWORDS if not args.stopwords else stopwords,
                                        sentences=args.sentences,
                                        window_size=args.window)

    print(df)
    print(wikipedia.similarities(csr, vocab))

