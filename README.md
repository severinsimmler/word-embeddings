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
usage: matrix-tool [-h] [--corpus CORPUS] [--vocab VOCAB] [--window WINDOW]
                   [--output OUTPUT]

CLI tool to process a Wikipedia dump to a word-word matrix.

optional arguments:
  -h, --help       show this help message and exit
  --corpus CORPUS  Path to corpus directory.
  --vocab VOCAB    Path to vocabulary file.
  --window WINDOW  Context window size.
  --output OUTPUT  Path to output file.
```
