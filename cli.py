#!/usr/bin/env python3

import argparse
import matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="fancy-name",
                                     description="Coole Beschreibung.")
    parser.add_argument('--wiki-dump', help="Filepath to Wikipedia dump.")

    args = parser.parse_args()
    
    # hier dann die funktionsaufrufe
