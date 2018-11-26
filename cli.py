#!/usr/bin/env python3

import argparse
import os
import matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="matrix-tool",
                                     description="CLI tool to process a Wikipedia "
                                                 "dump to a word-word matrix.")
    parser.add_argument('--corpus', help="Path to corpus directory.")
    parser.add_argument('--vocab', help="Path to vocabulary file.")
    parser.add_argument('--window', help="Context window size.", type=int)
    parser.add_argument('--output', help="Path to output directory.")

    args = parser.parse_args()

    coo_matrix = matrix.utils.create_cooccurrence_matrix(args.corpus,
                                                         args.vocab,
                                                         args.window)
    #similarities = matrix.utils.create_similarity_matrix(coo_matrix)

    matrix.utils.save_matrix(coo_matrix, os.path.join(args.output, "coo.csv"))
    #matrix.utils.save_matrix(similarities, os.path.join(args.output, "sim.csv"))
