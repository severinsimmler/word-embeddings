#!/usr/bin/env python3

import argparse
import matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="matrix-tool",
                                     description="CLI tool to process a Wikipedia "
                                                 "dump to a word-word matrix.")
    parser.add_argument('--corpus', help="Path to corpus directory.")
    parser.add_argument('--vocab', help="Path to vocabulary file.")
    parser.add_argument('--window', help="Context window size.")
    parser.add_argument('--output', help="Path to output file.")

    args = parser.parse_args()

    matrix = create_matrix(args["corpus_dir"],
                           args["mfw"],
                           args["window_size"])

    save_matrix(matrix, args["output"])
