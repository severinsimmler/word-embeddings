## Installation
[Pipenv](https://pipenv.readthedocs.io/en/latest/) automatically creates and manages a virtualenv for this project. Installation as usual:

```
$ pip install pipenv
```

To install the _project’s dependencies_:

```
$ pipenv install
```

You can spawn a shell:

```
$ pipenv shell
```

or a command installed into the virtual environment, for example:

```
$ pipenv run python cli.py --help
```


## Getting started

```
$ python cli.py --help
usage: matrix-tool [-h] [--corpus CORPUS] [--suffix SUFFIX] [--lowercase]
                   [--mfw MFW] [--n-mfw N_MFW] [--window WINDOW] [--sentences]
                   [--output OUTPUT] [--stopwords STOPWORDS] [--term TERM]

CLI tool to process a Wikipedia dump to a word-word matrix.

optional arguments:
  -h, --help            show this help message and exit
  --corpus CORPUS       Path to corpus directory.
  --suffix SUFFIX       Suffix of the text files.
  --lowercase           Use this parameter to lowercase all letters.
  --mfw MFW             Path to JSON file with most frequent words.
  --n-mfw N_MFW         Count tokens and use the n most frequent words.
  --window WINDOW       Context window size.
  --sentences           Use sentences instead of lines.
  --output OUTPUT       Path to output directory.
  --stopwords STOPWORDS
                        Optional external stopwords list.
  --term TERM           Get top 50 nearest neighbors for this term.
```
