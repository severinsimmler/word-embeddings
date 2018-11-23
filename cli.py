#!/usr/bin/env python3

import argparse
import matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="matrix-tool",
                                     description="CLI tool to process a Wikipedia "
                                                 "dump to a word-word matrix.")
    parser.add_argument('--corpus-dir', help="Path to corpus directory.")

    args = parser.parse_args()
    
    matrix = create_matrix(filepath, mfw, window_size, stopwords)
