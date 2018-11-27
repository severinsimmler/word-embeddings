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
    parser.add_argument("--term", help="Get top 50 nearest neighbors for this term.")
    parser.add_argument("--sublinear_tf", help="Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).",
                        action="store_true")
    parser.add_argument("--doc_tfidf", help="Calculate tf-idf from article 'documents' as a weight for the"
                                            "word-word matrix.",
                        action="store_true")
    parser.add_argument("--global_tfidf", help="Normalize word-word matrix with tf-idf transformation",
                        action="store_true")

    args = parser.parse_args()

    # Construct Wikipedia corpus object:
    wikipedia = matrix.corpus.Wikipedia(path=args.corpus,
                                        suffix=args.suffix,
                                        lowercase=args.lowercase)

    # Read stopwords from file, if any (using NLTK otherwise):
    if args.stopwords:
        with Path(args.stopwords).open("r", encoding="utf-8") as textfile:
            stopwords = textfile.read().split("\n")

    # Load most frequent words from a file:
    if args.mfw:
        mfw = wikipedia.load_mfw(args.mfw)
    # Or count them:
    elif args.n_mfw:
        mfw = wikipedia.mfw(args.n_mfw,
                            matrix.utils.STOPWORDS if not args.stopwords else stopwords)
    else:
        raise ValueError("You have to set either a threshold for the most frequent words, "
                         "or pass a path to a JSON file with most frequent words.")

    logging.info("Creating sparse matrix...")
    # Create sparse scipy matrix from corpus:
    if args.doc_tfidf:
        tfidf_weights = wikipedia.create_tfidf_features(mfw=mfw)
        csr, vocab = wikipedia.sparse_coo_matrix(mfw,
                                                 stopwords=matrix.utils.STOPWORDS if not args.stopwords else stopwords,
                                                 sentences=args.sentences,
                                                 window_size=args.window, tfidf_weights=tfidf_weights)
    else:
        csr, vocab = wikipedia.sparse_coo_matrix(mfw,
                                                 stopwords=matrix.utils.STOPWORDS if not args.stopwords else stopwords,
                                                 sentences=args.sentences,
                                                 window_size=args.window)

    # Normalize with tf-idf:
    if args.global_tfidf:
        csr = wikipedia.tfidf(csr)

    logging.info("Calculating similarities...")
    # Calculate cosine similarity:
    similarities = wikipedia.similarities(csr, vocab)

    logging.info("Sorting similarities...")
    # Sorting ascending (the higher the value, the more similar a vector):
    most_similar_stadt = similarities[args.term].sort_values(ascending=False)[:50]
    logging.info("Saving to file...")
    most_similar_stadt.to_csv(f"most-similar-{args.term}.csv")

    logging.info("Scipy matrix to pandas matrix...")
    # Scipy sparse matrix to pandas SparseDataFrame:
    df = wikipedia._sparse2dataframe(csr, vocab, sparse=True)
    logging.info("Saving to file...")
    df.to_csv("coo-matrix.csv")
